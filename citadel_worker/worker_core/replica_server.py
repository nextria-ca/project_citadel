# citadel_worker/worker_core/replica_server.py
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import pickle
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any

from citadel_shared.citadel_model import CitadelModel
from citadel_shared.os_utils import ipc_address, WIN, gpu_mem_info
from citadel_shared.logging_setup import (
    setup_logging,
    get_logger,
    SafeStreamHandler,
    _ColourFormatter,
)

setup_logging(env="beta")

_root = logging.getLogger()
_stdout_hdlr = SafeStreamHandler(sys.stdout)
_stdout_hdlr.setLevel(logging.INFO)
_stdout_hdlr.setFormatter(
    _ColourFormatter(
        "%(asctime)s | %(levelname)-20s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_root.addHandler(_stdout_hdlr)

_LOG = get_logger(__name__)


# ─────────────────────────── minimalist serializer ──────────────────────────
def _dumps(obj: Any) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _loads(buf: bytes):
    try:
        return json.loads(buf.decode("utf-8"))
    except Exception:
        return buf
# ────────────────────────

# ────────────────────────── protocol ─────────────────────────
MSG_LEN   = 4
SENTINEL  = b"__stream_end__"
_TLS_RECV = threading.local()


def _get_buf(size: int) -> bytearray:
    buf: bytearray | None = getattr(_TLS_RECV, "buf", None)
    if buf is None or len(buf) < size:
        buf = bytearray(size)
        _TLS_RECV.buf = buf
    return buf


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    view = memoryview(_get_buf(n))[:n]
    mv = view
    while mv:
        r = sock.recv_into(mv)
        if r == 0:
            raise ConnectionResetError("socket closed")
        mv = mv[r:]
    return view.tobytes()


def _recv_msg(sock: socket.socket) -> Any:
    size = struct.unpack("!I", _recv_exact(sock, MSG_LEN))[0]
    data = _recv_exact(sock, size)
    try:
        return _loads(data)
    except Exception:
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return data


def _send_msg(sock: socket.socket, obj: Any) -> None:
    if isinstance(obj, memoryview):
        obj = obj.tobytes()

    if isinstance(obj, (bytes, bytearray)):
        data = obj
    elif isinstance(obj, (dict, list, str)):
        data = _dumps(obj)
    else:
        data = str(obj).encode("utf-8")

    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)



# ─────────────────────── model loading ───────────────────────
def _load_impl(model_dir: Path) -> CitadelModel:
    import importlib.util as iu
    import json

    # 1) config is optional – still try to read it (for time-outs etc.)
    cfg_file = model_dir / "config.json"
    try:
        with cfg_file.open(encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        cfg = {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {cfg_file}: {exc}") from exc

    # 2) decide which *.py to load
    wrapper_name = os.getenv("CITADEL_WRAPPER_NAME", "").strip()
    cand_py      = model_dir / f"{wrapper_name}.py" if wrapper_name else None
    if cand_py and cand_py.exists():
        target_py = cand_py
    else:
        target_py = model_dir / "model.py"
        if not target_py.exists():
            # last-chance: first *.py in the folder
            try:
                target_py = next(p for p in model_dir.glob("*.py") if p.name != "__init__.py")
            except StopIteration as exc:
                raise FileNotFoundError(
                    f"No model.py and no matching wrapper in {model_dir}"
                ) from exc

    # 3) import & instantiate
    spec = iu.spec_from_file_location("user_model", target_py)
    mod  = iu.module_from_spec(spec)           # type: ignore[arg-type]
    assert spec.loader
    spec.loader.exec_module(mod)               # type: ignore[arg-type]

    for obj in vars(mod).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, CitadelModel)
            and obj is not CitadelModel
        ):
            return obj(str(model_dir), cfg)
    raise RuntimeError("No CitadelModel subclass found")


# ─────────────────────────── main ────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--port",      required=True, type=int)
    args = p.parse_args()

    _LOG.info("Replica starting - model-dir=%s  port=%s", args.model_dir, args.port)

    mdl    = _load_impl(Path(args.model_dir))
    gpu_id = getattr(mdl, "gpu_id", None)

    _, pre_free_mb = gpu_mem_info(gpu_id)

    mdl.init()

    _, post_free_mb = gpu_mem_info(gpu_id)
    vram_used_mb    = max(0, pre_free_mb - post_free_mb)
    mdl.vram_used_mb = vram_used_mb

    _LOG.info(
        "Model initialised from %s - GPU %s  VRAM +%d MB (free %d → %d MB)",
        args.model_dir,
        gpu_id if gpu_id is not None else "N/A",
        vram_used_mb,
        pre_free_mb,
        post_free_mb,
    )

    address = ipc_address(args.port)

    if isinstance(address, tuple):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(address)
    else:
        if not WIN:
            try:
                os.unlink(address)
            except FileNotFoundError:
                pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(address)

    server.listen(1)
    _LOG.info("Listening on %s", address)

    conn, _ = server.accept()
    _LOG.debug("Client connected")
    with server, conn:
        while True:
            msg = _recv_msg(conn)
            if msg == "__shutdown__":
                _LOG.debug("Shutdown signal received")
                break
            
            if isinstance(msg, dict) and msg.get("_type") == "clear_cache":
                try:
                    try:
                        import torch  # type: ignore
                        if getattr(torch, "cuda", None) and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    _send_msg(conn, {"ok": True})
                except Exception as exc:
                    _LOG.debug("clear_cache failed: %s", exc)
                    _send_msg(conn, {"ok": False, "error": str(exc)})
                continue

            if isinstance(msg, dict) and msg.get("_type") == "batch_execute":
                try:
                    reply = mdl.execute_batch(msg["payload"])
                except Exception as exc:
                    _LOG.warning("execute_batch failed (%s); falling back to execute()", exc)
                    reply = [mdl.execute(x) for x in msg["payload"]]
                _send_msg(conn, reply)
                continue

            if isinstance(msg, dict) and msg.get("_type") == "stream_execute":
                gen = mdl.stream_execute(msg["payload"])
                if inspect.isasyncgen(gen):

                    async def _relay():
                        async for part in gen:
                            _send_msg(conn, part)

                    asyncio.run(_relay())
                else:
                    for part in gen:
                        _send_msg(conn, part)
                _send_msg(conn, SENTINEL)
                continue

            # ──────────────────────────────────────────────────────
            #  SAFE one-shot execute (prevents replica crash loops)
            # ──────────────────────────────────────────────────────
            try:
                result = mdl.execute(msg)
            except Exception as exc:
                _LOG.exception("Model raised during execute: %s", exc)
                _send_msg(conn, {"error": str(exc)})
                continue              # stay alive for the next request

            _send_msg(conn, result)

    mdl.finalize()
    if not isinstance(address, tuple) and not WIN:
        try:
            os.unlink(address)
        except OSError:
            pass
    _LOG.info("Replica shut down cleanly")


if __name__ == "__main__":
    main()
