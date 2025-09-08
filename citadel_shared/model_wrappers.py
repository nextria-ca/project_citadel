from __future__ import annotations

import asyncio
import inspect
import json as _std_json
try:
    import orjson as _json 
except ModuleNotFoundError:
    _json = _std_json 
import base64
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import importlib.util as _ilu
from pathlib import Path
from typing import Any, Dict
import base64



from citadel_shared.logging_setup import setup_logging, get_logger
setup_logging(env="beta")
logger = get_logger(__name__)

from citadel_shared.citadel_model import CitadelModel
from citadel_shared.os_utils import conda_cmd, norm_env_path, ipc_address, WIN, gpu_mem_info

def _wrap_bytes(obj: Any):
    """
    Recursively walk *obj* and replace any bytes / bytearray with
    {"__bytes__": "<base64>"} so JSON encoders stay happy.
    """
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _wrap_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wrap_bytes(x) for x in obj]
    return obj


# ───────────────────────────── dumps / loads ───────────────────────────
def _dumps(obj: Any) -> bytes:
    """
    Safe serialiser used by SubprocModel wrappers.
    • bytes / str go through unchanged (fast-path)
    • any other Python value → JSON (orjson when available)
      with bytes auto-wrapped so the decoder can reverse it.
    """
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")

    obj = _wrap_bytes(obj)                      # <── NEW

    if _json is _std_json:                      # stdlib path
        return _json.dumps(
            obj, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    # orjson path – OPT_NON_STR_KEYS replicates stdlib behaviour
    return _json.dumps(obj, option=_json.OPT_NON_STR_KEYS)


def _unwrap_bytes(obj: Any):
    if isinstance(obj, dict):
        if obj.keys() == {"__bytes__"} and isinstance(obj["__bytes__"], str):
            return base64.b64decode(obj["__bytes__"])
        return {k: _unwrap_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_bytes(x) for x in obj]
    return obj


def _loads(buf: bytes):
    try:
        data = (
            _json.loads(buf) if _json is not _std_json else _json.loads(buf.decode("utf-8"))
        )
        return _unwrap_bytes(data)              # reverse the wrapping
    except Exception:
        return buf



# ─────────────────────────── framing helpers ────────────────────────────
_PORTS_IN_USE: set[int] = set()
_PORTS_LOCK            = threading.Lock()

MSG_LEN_SIZE = 4
_TLS_RECV    = threading.local()


def _get_buffer(size: int) -> bytearray:
    buf: bytearray | None = getattr(_TLS_RECV, "buf", None)
    if buf is None or len(buf) < size:
        buf = bytearray(size)
        _TLS_RECV.buf = buf
    return buf


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    view = memoryview(_get_buffer(n))[:n]
    mv   = view
    while mv:
        r = conn.recv_into(mv)
        if r == 0:
            raise ConnectionResetError("socket closed")
        mv = mv[r:]
    return view.tobytes()


def _recv_msg(conn: socket.socket, *, raw: bool = True):
    size = struct.unpack("!I", _recv_exact(conn, MSG_LEN_SIZE))[0]
    data = _recv_exact(conn, size)
    if raw:
        return data

    try:
        return _loads(data)
    except Exception:
        try:
            return _json.loads(data.decode("utf-8"))
        except Exception:
            return data


def _send_msg(conn: socket.socket, obj: Any) -> None:
    """
    Serialise **once** and blast the frame:

    • bytes / bytearray / memoryview → sent untouched  
    • dict / list / str              → compact UTF-8 JSON  
    • everything else                → str(obj).encode('utf-8')
    """
    if isinstance(obj, memoryview):
        obj = obj.tobytes()

    if isinstance(obj, (bytes, bytearray)):
        data = obj
    elif isinstance(obj, (dict, list, str)):
        data = _dumps(obj)
    else:
        data = str(obj).encode("utf-8")

    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)


# ─────────────────────── paths to models / service ──────────────────────
def _resolve_models_root() -> Path:
    for pkg in ("citadel_worker", "citadel_shared"):
        spec = _ilu.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations))) / "models"
            if root.exists():
                return root
    return Path(__file__).resolve().parent.parent / "models"


MODELS_ROOT = _resolve_models_root()


def _resolve_service_script() -> Path:
    spec = _ilu.find_spec("citadel_worker.worker_core")
    if spec and spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
        for name in ("replica_server.py", "model_service.py"):
            cand = base / name
            if cand.exists():
                return cand
    local = Path(__file__).parent
    for name in ("replica_server.py", "model_service.py"):
        cand = local / name
        if cand.exists():
            return cand
    raise FileNotFoundError("Cannot find replica_server.py / model_service.py")


SERVICE_SCRIPT = _resolve_service_script()

# ─────────────────────── stdout / stderr tee helper ─────────────────────
class _TeeStd:
    def __init__(self, orig):
        self._orig = orig

    def write(self, buf: str) -> None:
        if buf:
            self._orig.write(buf)

    def flush(self) -> None:
        self._orig.flush()

    def isatty(self) -> bool:
        return False


def _patch_stdio_once() -> None:
    if getattr(_patch_stdio_once, "_done", False):   # type: ignore[attr-defined]
        return
    sys.stdout = _TeeStd(sys.__stdout__)            # type: ignore[assignment]
    sys.stderr = _TeeStd(sys.__stderr__)            # type: ignore[assignment]
    _patch_stdio_once._done = True                  # type: ignore[attr-defined]

# ───────────────────────────── In-process model ─────────────────────────
class InProcModel(CitadelModel):
    """Loads the user model directly into the worker process."""
    def __init__(self, model_dir: str, config: Dict):
        super().__init__(model_dir, config)
        _patch_stdio_once()

        self.gpu_id       = config.get("gpu_id")
        self.vram_used_mb = 0

        mod_path = Path(model_dir) / "model.py"
        try:
            spec = _ilu.spec_from_file_location("user_model", mod_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {mod_path}")
            mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)       # type: ignore[arg-type]
        except Exception as exc:
            # Import errors should NEVER halt global load — log and bubble up.
            logger.exception("Failed importing model module %s: %s", mod_path, exc)
            raise

        impl_cls = None
        for obj in vars(mod).values():
            if inspect.isclass(obj) and issubclass(obj, CitadelModel) and obj is not CitadelModel:
                impl_cls = obj
                break

        if impl_cls is None:
            # No model class — this is a hard error for THIS model only.
            raise RuntimeError(f"No CitadelModel subclass found in {mod_path}")

        try:
            self.impl = impl_cls(model_dir, config)  # type: ignore[call-arg]
        except Exception as exc:
            logger.exception("Failed constructing model from %s: %s", mod_path, exc)
            raise

    # -- wrapped API -----------------------------------------------------
    def init(self) -> None:
        try:
            pre_free_mb, _ = gpu_mem_info(self.gpu_id)
        except Exception:
            pre_free_mb = 0

        try:
            self.impl.init()
        except Exception as exc:
            logger.exception("Model init() failed for %s: %s", self.model_dir, exc)
            raise

        try:
            post_free_mb, _ = gpu_mem_info(self.gpu_id)
        except Exception:
            post_free_mb = pre_free_mb

        self.vram_used_mb = max(0, pre_free_mb - post_free_mb)
        logger.info(
            "InProcModel %s loaded - GPU %s  VRAM +%d MB (free %d → %d MB)",
            self.model_dir,
            self.gpu_id if self.gpu_id is not None else "N/A",
            self.vram_used_mb,
            pre_free_mb,
            post_free_mb,
        )
        self._run_warmup()
    
    def clear_cache(self) -> None:
        # let the user implementation participate if it exposes a hook
        try:
            hook = getattr(self.impl, "clear_cache", None)
            if callable(hook):
                hook()
        except Exception:
            pass
        # generic CUDA cache clear
        super().clear_cache()


# ─────────────────────────── Sub-process replica ───────────────────────
class SubprocModel(CitadelModel):
    """
    Launches *replica_server.py* inside the requested Conda env and exchanges
    length-prefixed binary frames over an IPC socket.
    """

    CONNECT_RETRIES = int(os.getenv("CITADEL_CONNECT_RETRIES", "3000"))
    RETRY_DELAY     = float(os.getenv("CITADEL_CONNECT_RETRY_DELAY", "0.1"))

    def __init__(self, model_dir: str, config: Dict):
        super().__init__(model_dir, config)
        self.env_path       = config["conda_env"]
        self.gpu_id: int | None = config.get("gpu_id")
        self.port:   int | None = None
        self._conn:  socket.socket | None = None
        self.vram_used_mb = 0

        logger.debug(
            "SubprocModel created: dir=%s  env=%r  gpu_id=%s",
            model_dir, self.env_path, self.gpu_id,
        )

    # ------------------------------ public API ------------------------------
    def init(self) -> None:
        self._connect()
        self._run_warmup()
    
    def clear_cache(self) -> None:
        try:
            self._ensure_conn()
            _send_msg(self._conn, {"_type": "clear_cache"})
            # read small ack; parsed as JSON/dict
            _recv_msg(self._conn, raw=False)
        except Exception as exc:
            logger.debug("clear_cache IPC failed: %s", exc)

    def _run_warmup(self) -> None:
        warm = self.config.get("warmup")
        if not warm:
            return

        def _specs(obj):
            if isinstance(obj, list):
                for spec in obj:
                    yield spec
            elif isinstance(obj, dict):
                yield obj

        for spec in _specs(warm):
            amount = int(spec.get("amount", 1))
            inp    = spec.get("input") or {k: v for k, v in spec.items() if k != "amount"}
            for _ in range(max(1, amount)):
                try:
                    self.execute(inp)
                except Exception as exc:
                    logger.warning("warm-up failed: %s", exc)
                    break

    # ------------------------------ invoke RPC ------------------------------
    def execute(self, data):
        attempt = 0
        while True:
            try:
                self._ensure_conn()
                raw = data if isinstance(data, (bytes, bytearray, memoryview)) else _dumps(data)
                _send_msg(self._conn, raw)
                return _recv_msg(self._conn, raw=True)
            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt >= 1:
                    raise
                attempt += 1
                logger.warning("Replica connection broken – restarting and retrying once")
                self._reset_conn()

    def execute_batch(self, items):
        attempt = 0
        while True:
            try:
                self._ensure_conn()
                _send_msg(self._conn, {"_type": "batch_execute", "payload": items})
                return _recv_msg(self._conn, raw=True)
            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt >= 1:
                    raise
                attempt += 1
                logger.warning("Replica connection broken during batch – restarting and retrying once")
                self._reset_conn()

    async def stream_execute(self, data):
        attempt = 0
        while True:
            try:
                self._ensure_conn()
                await asyncio.to_thread(
                    _send_msg, self._conn, {"_type": "stream_execute", "payload": data}
                )
                sentinel = b"__stream_end__"
                while True:
                    part = await asyncio.to_thread(_recv_msg, self._conn, raw=True)
                    if part == sentinel:
                        break
                    yield part
                return
            except (BrokenPipeError, ConnectionResetError, OSError):
                if attempt >= 1:
                    raise
                attempt += 1
                logger.warning("Replica connection broken during stream – restarting and retrying once")
                self._reset_conn()

    def finalize(self):
        try:
            if self._conn:
                _send_msg(self._conn, "__shutdown__")
        finally:
            self._reset_conn()

    # ---------------------------- internal I/O ------------------------------
    def _ensure_conn(self):
        if self._conn is None:
            self._connect()
            return
        try:
            self._conn.send(b"")
        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.warning("Replica connection unhealthy – re-establishing")
            self._reset_conn()
            self._connect()

    def _reset_conn(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        if self.port is not None:
            with _PORTS_LOCK:
                _PORTS_IN_USE.discard(self.port)
            self.port = None

    # ----------------------- establishing the replica -----------------------
    

    # --------------------- launching the replica process --------------------
    def _run_in_env(self, env_path: Path, *args: str) -> None:
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run([conda_cmd(), "run", "-p", env_path, *args], check=True, env=env)

    def _launch_subprocess(self):
        env_path = norm_env_path(self.env_path)

        if WIN:
            python_exe = Path(env_path, "python.exe")
            if not python_exe.exists():
                python_exe = Path(env_path, "Scripts", "python.exe")
        else:
            python_exe = Path(env_path, "bin", "python")
        if not python_exe.exists():
            raise FileNotFoundError(f"python executable not found at {python_exe}")

        cmd = [
            str(python_exe),
            "-u",
            str(SERVICE_SCRIPT),
            "--model-dir", str(self.model_dir),
            "--port",      str(self.port),
        ]

        # Ensure the child prefers its env (Windows DLLs depend on PATH order)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"]   = "1"
        env["PYTHONNOUSERSITE"]   = "1"
        env["CITADEL_WRAPPER_NAME"] = self.config["name"]

        # Child writes to stdout (console) only; parent owns file + optional tee.
        env["CITADEL_LOG_CONSOLE"] = "1"
        env["CITADEL_LOG_FILES"]   = "0"

        env["PATH"] = f"{env_path}{os.pathsep}{env.get('PATH','')}"

        log_path = Path(self.model_dir, f"replica_{self.port}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path  # used by _connect() for error tail

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._proc = proc  # so _connect() can detect early exit

        # ─────────────── tee behaviour (default: echo to console) ───────────────
        def _parse_tee_mode(value: str | None):
            """
            Returns (to_console, to_logger) from CITADEL_REPLICA_TEE.
            off/0/false/no  -> (False, False)
            console         -> (True,  False)   [default]
            logger          -> (False, True)
            both            -> (True,  True)
            1/true/yes      -> (True,  False)
            """
            v = (value or "console").strip().lower()
            if v in ("off", "0", "false", "no"):
                return False, False
            if v in ("1", "true", "yes", "console"):
                return True, False
            if v == "logger":
                return False, True
            if v == "both":
                return True, True
            return True, False

        def _safe_console_write(prefix: str, line: str) -> None:
            # Single, direct write to the parent's real stdout; no extra formatting passes.
            try:
                sys.__stdout__.write(f"{prefix}{line}\n")
                sys.__stdout__.flush()
            except Exception:
                pass

        tee_console, tee_logger = _parse_tee_mode(os.getenv("CITADEL_REPLICA_TEE"))
        prefix = f"[replica {self.port}] "

        def _relay_output(pipe, dest_path: Path) -> None:
            with open(dest_path, "a", encoding="utf-8") as fh:
                for line in iter(pipe.readline, ''):
                    if not line:
                        continue
                    line = line.rstrip("\r\n")

                    # always write to file
                    fh.write(line + "\n")
                    fh.flush()

                    # optionally echo to terminal (fast path, no double-formatting)
                    if tee_console:
                        _safe_console_write(prefix, line)

                    # optionally mirror into the parent logger (kept for completeness)
                    if tee_logger:
                        logger.info("%s%s", prefix, line)

        threading.Thread(
            target=_relay_output,
            args=(proc.stdout, log_path),
            daemon=True,
            name=f"relay-{self.port}",
        ).start()

        # compact summary of tee mode
        if tee_console and tee_logger:
            tee_summary = "both"
        elif tee_console:
            tee_summary = "console"
        elif tee_logger:
            tee_summary = "logger"
        else:
            tee_summary = "off"

        logger.info(
            "Replica launched (%s) – live logs → %s (tee=%s)",
            self.model_dir,
            log_path,
            tee_summary,
        )


    # ----------------------- establishing the replica -----------------------
    def _connect(self):
        pre_free_mb = gpu_mem_info(self.gpu_id)[1] if self.gpu_id not in (None, -1) else 0

        if self.port is None:
            self.port = _pick_free_port()
            logger.debug(
                "Picked free port %s for %s (gpu_id=%s)",
                self.port, self.model_dir, self.gpu_id,
            )
            self._launch_subprocess()

        addr = ipc_address(self.port)

        def _log_tail(n: int = 80) -> str:
            lp = getattr(self, "_log_path", None)
            if not lp:
                return ""
            try:
                with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
                tail = "".join(lines[-n:])
                return f"\n----- replica log (tail) -----\n{tail}-------------------------------"
            except Exception:
                return ""

        for _ in range(self.CONNECT_RETRIES):
            proc = getattr(self, "_proc", None)
            if proc is not None:
                rc = proc.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"Replica process exited early with code {rc}.{_log_tail()}"
                    )

            try:
                if isinstance(addr, tuple):           # loop-back TCP
                    self._conn = socket.create_connection(addr)
                else:                                  # AF_UNIX (non-Windows)
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(addr)
                    self._conn = s

                logger.debug("Connected to replica socket %s", addr)

                post_free_mb      = gpu_mem_info(self.gpu_id)[1] if self.gpu_id not in (None, -1) else pre_free_mb
                self.vram_used_mb = max(0, pre_free_mb - post_free_mb)
                logger.info(
                    "SubprocModel %s ready – GPU %s  VRAM +%d MB (free %d → %d MB)",
                    self.model_dir,
                    self.gpu_id if self.gpu_id is not None else "N/A",
                    self.vram_used_mb,
                    pre_free_mb,
                    post_free_mb,
                )
                break
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                time.sleep(self.RETRY_DELAY)
        else:
            # Final timeout without a socket – include log tail for clues
            raise RuntimeError(f"Replica on {addr} never became reachable.{_log_tail()}")


# ────────────────────────────── port allocator ─────────────────────────
def _pick_free_port() -> int:
    port_range_str = os.getenv("CITADEL_PORT_RANGE", "9501-9599")
    try:
        lo, hi = map(int, port_range_str.strip().split("-", 1))
        assert 0 < lo < hi <= 65535
    except Exception as exc:
        raise RuntimeError(f"Invalid CITADEL_PORT_RANGE '{port_range_str}': {exc}") from exc

    with _PORTS_LOCK:
        for port in range(lo, hi + 1):
            if port in _PORTS_IN_USE:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _PORTS_IN_USE.add(port)
            return port

    raise RuntimeError(f"No free ports available in range {lo}-{hi}")
