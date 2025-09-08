from __future__ import annotations
from citadel_master.metrics.core.metrics_dashboard import calculate_mesh

"""Async gRPC front-end on the master – updated for protobuf one-of.

▪ Change log 2025-07-30 (allow no-knowledge runs)
    * Legacy single-message & streaming training paths now accept empty
      `knowledge_items` without raising the NO_KNOWLEDGE_RECEIVED error.
    * A new boolean flag `no_files` (mirrors the upload helper) is
      propagated through metadata but **is not required** – an empty list
      alone is enough.

▪ Change log 2025-08-22 (trainer ingest passthrough for T2T)
    * Merge top-level T2T/ingest fields from struct_data into `hyperparams`
      before launching the trainer so ingest-only jobs can be sent via
      the trainer route. Fields: action, source, mode, src_acronym,
      src_description, tgt_acronym, tgt_description.
"""

import asyncio
import base64
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator, Dict, List, Tuple

import grpc
from grpc import aio
from google.protobuf.json_format import MessageToDict
from proto.python import (
    inference_pb2,
    inference_pb2_grpc,
    trainer_pb2,
    trainer_pb2_grpc,
)

from citadel_shared.logging_setup import get_logger
from citadel_db.db_helpers import run_query, exec_sql
from citadel_master.master_core import dispatcher
from master_core.worker_registry import REGISTRY

_GRPC_OPTS = [
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 60_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.http2.min_time_between_pings_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# ───────────────────────────── local light-weight serializer ────────────────


def _dumps(obj) -> bytes:  # bytes | str | dict → bytes
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _loads(buf: bytes):
    try:
        return json.loads(buf.decode("utf-8"))
    except Exception:  # UnicodeDecodeError | JSONDecodeError
        return buf


_LOG = get_logger(__name__)

_DB_MODEL_NAME = "__db_query__"
_ENV_TOKEN_KEY = "CITADEL_DB_QUERY_TOKEN"
_ADMIN_PREFIX = "special_"
_METRICS_PREFIX = "metrics_"

# ───────────────────────── helper: restore real extension ───────────────────


def _maybe_fix_extension(path: str) -> str:
    """
    Upload clients often save every temp file as “.json” even when the
    bytes are actually DOCX / PDF / TXT.  That confuses DocumentLoader later.
    """
    if path.startswith("b64:"):
        return path

    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".json":
        return path

    try:
        with p.open("rb") as fh:
            magic = fh.read(4)
    except Exception:
        return path

    if magic.startswith(b"PK\x03\x04"):
        new_path = p.with_suffix(".docx")
    elif magic.startswith(b"%PDF"):
        new_path = p.with_suffix(".pdf")
    else:
        return path  # genuine JSON or unknown – leave as-is

    try:
        p.rename(new_path)  # fast path (same filesystem)
    except OSError:
        shutil.copy2(p, new_path)
        p.unlink(missing_ok=True)

    return str(new_path)


async def _maybe_reassemble_audio_stream(
    first_req: inference_pb2.InferenceRequest,
    rest_iterator: AsyncIterator[inference_pb2.InferenceRequest],
) -> AsyncIterator[inference_pb2.InferenceRequest]:
    """
    If the client is streaming JSON frames like {"audio_data":"<b64>", "task_type": "..."}
    with a session_id and last_chunk flags, merge them into ONE request whose
    `raw` contains the full base64 audio. Otherwise, pass through unchanged.
    """
    # must be the streaming pattern we care about
    if not first_req.stream or not first_req.session_id:
        # no change – yield first then tail
        yield first_req
        async for r in rest_iterator:
            yield r
        return

    sid = first_req.session_id
    buf = bytearray()
    meta: dict | None = None
    seen_any = False
    parse_failed = False

    async def _iter_all():
        yield first_req
        async for r in rest_iterator:
            yield r

    async for req in _iter_all():
        # only merge frames that belong to the same session_id
        if req.session_id != sid or not req.HasField("raw") or not req.raw:
            # different session or no raw – bail out to passthrough
            parse_failed = True
            break

        try:
            frame = _loads(req.raw)
        except Exception:
            parse_failed = True
            break

        if not isinstance(frame, dict) or "audio_data" not in frame:
            parse_failed = True
            break

        if meta is None:
            # keep anything except the chunk itself (e.g., task_type)
            meta = dict(frame)
            meta.pop("audio_data", None)

        try:
            import base64 as _b64

            buf.extend(_b64.b64decode(frame["audio_data"]))
        except Exception:
            parse_failed = True
            break

        seen_any = True

        # keep accumulating until last_chunk; do not yield intermediate frames
        if not req.last_chunk:
            continue

        # last chunk for this session – emit a SINGLE merged request
        merged = dict(meta or {})
        merged["audio_data"] = _b64.b64encode(bytes(buf)).decode("ascii")

        # clone the first request but as non-streaming single payload
        final_req = inference_pb2.InferenceRequest()
        final_req.CopyFrom(first_req)
        final_req.stream = False
        final_req.session_id = ""
        final_req.last_chunk = False
        final_req.ClearField("struct_data")
        final_req.raw = _dumps(merged)

        yield final_req
        break  # finished our session

    if not seen_any or parse_failed:
        # fall back to passthrough: re-yield original frames (unchanged)
        yield first_req
        async for r in rest_iterator:
            yield r


# ─────────────────────────────  Admin helper  ───────────────────────────────


class _AdminManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tx_history: Dict[str, List[Tuple[str, str]]] = {}

    def list_enabled(self) -> List[Dict]:
        with self._lock:
            reps: List[Dict] = []
            for wid, info in REGISTRY._db.items():
                for model in info.models:
                    reps.append(
                        {
                            "model": model,
                            "worker": wid,
                            "gpu_id": self._guess_gpu_id(wid),
                            "endpoint": info.endpoint,
                        }
                    )
            return reps

    def unload(self, sel: Dict) -> Dict:
        tx_id = str(uuid.uuid4())
        disabled: List[Tuple[str, str]] = []

        with self._lock:
            for wid, info in REGISTRY._db.items():
                if sel.get("worker") and sel["worker"] != wid:
                    continue
                for model in list(info.models):
                    if sel.get("model") and sel["model"] != model:
                        continue
                    info.models.remove(model)
                    disabled.append((model, wid))

        if not disabled:
            return {"error": "Nothing matched the selection"}

        self._tx_history[tx_id] = disabled
        return {"transaction": tx_id, "disabled": disabled}

    def load(self, sel: Dict) -> Dict:
        restored: List[Tuple[str, str]] = []

        with self._lock:
            for wid, info in REGISTRY._db.items():
                if sel.get("worker") and sel["worker"] != wid:
                    continue
                wanted = sel.get("model")
                if not wanted:
                    continue
                if wanted not in info.models:
                    info.models.append(wanted)
                    restored.append((wanted, wid))

        return {"restored": len(restored), "details": restored}

    def undo(self, tx_id: str) -> Dict:
        with self._lock:
            if tx_id not in self._tx_history:
                return {"error": "Unknown transaction"}
            restored: List[Tuple[str, str]] = []
            for model, wid in self._tx_history.pop(tx_id):
                inf = REGISTRY._db.get(wid)
                if inf and model not in inf.models:
                    inf.models.append(model)
                    restored.append((model, wid))
            return {"restored": len(restored), "details": restored}

    @staticmethod
    def _guess_gpu_id(worker_id: str) -> int | None:
        stats = REGISTRY.replica_stats(worker_id)
        return stats[0].get("gpu_id") if stats else None


_ADMIN = _AdminManager()

# ─────────────────────────── Inference gRPC servicer ─────────────────────────


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    async def Infer(self, request_iterator, context):  # type: ignore[override]
        aiter = request_iterator.__aiter__()

        try:
            first = await aiter.__anext__()
        except StopAsyncIteration:
            return

        # ---------------- TRAIN ----------------
        if (first.level or "").lower() == "train":
            async for part in self._handle_train(first, request_iterator):
                yield part
            return

        # ---------------- ADMIN ----------------
        if first.model.startswith(_ADMIN_PREFIX):
            resp = await asyncio.to_thread(self._handle_admin, first)
            yield resp
            async for _ in request_iterator:  # drain iterator
                pass
            return

        # --------------- METRICS -------------
        if first.model.startswith(_METRICS_PREFIX):
            resp = await asyncio.to_thread(self._handle_metrics, first)
            yield resp
            async for _ in request_iterator:
                pass
            return

        # ---------------- DB QUERY -------------
        if first.model == _DB_MODEL_NAME and first.HasField("db_query"):
            resp = await asyncio.to_thread(self._handle_db, first.db_query)
            yield resp
            async for _ in request_iterator:
                pass
            return

        # --------------- INFERENCE -------------
        async def _maybe_stream():
            async for r in _maybe_reassemble_audio_stream(first, request_iterator):
                yield r

        async for rsp in dispatcher.forward_stream_async(_maybe_stream()):
            yield rsp

    # ------------------------------------------------------------------ #
    # TRAINING HANDLER
    # ------------------------------------------------------------------ #

    async def _handle_train(
        self,
        first_req: inference_pb2.InferenceRequest,
        rest_iterator: AsyncIterator[inference_pb2.InferenceRequest],
    ) -> AsyncIterator[inference_pb2.InferenceResponse]:
        """Legacy single-message path and streaming multi-message path."""

        def _to_dict(req: inference_pb2.InferenceRequest) -> Dict:
            if req.HasField("struct_data"):
                return MessageToDict(req.struct_data, preserving_proto_field_name=True)
            try:
                return _loads(req.raw)
            except Exception:
                return {}

        # NEW: merge helper — copy ingest/T2T hints into hyperparams
        def _merge_t2t_fields(raw_h: Dict, d: Dict) -> Dict:
            """
            Ensure trainer receives ingest-only intent even on the TRAIN route by
            merging top-level fields into hyperparams (without clobbering existing keys).
            """
            if not isinstance(raw_h, dict):
                raw_h = {}
            for k in (
                "action",
                "source",
                "mode",
                "src_acronym",
                "src_description",
                "tgt_acronym",
                "tgt_description",
            ):
                v = d.get(k)
                if v is not None and v != "":
                    raw_h.setdefault(k, v)
            return raw_h

        first_dict = _to_dict(first_req)
        is_streaming = "chunk_b64" in first_dict  # heuristic

        # ========== LEGACY SINGLE-MESSAGE ==========
        if not is_streaming:
            raw_items: List[str] = list(first_dict.get("knowledge", []))
            knowledge_items = [_maybe_fix_extension(x) for x in raw_items]
            # ── MOD: accept empty knowledge (no_files flag optional) ──
            if not knowledge_items and not first_dict.get("no_files", False):
                knowledge_items = []  # blank run – allowed

            # Merge T2T fields into hyperparams for trainer CLI
            raw_h = first_dict.get("hyperparams", {})
            raw_h = _merge_t2t_fields(raw_h, first_dict)
            hyperparams = {str(k): str(v) for k, v in raw_h.items()}

            async for resp in self._start_train_job(
                model_name=first_req.model,
                knowledge_items=knowledge_items,
                meta={
                    "name": first_dict.get("name", ""),
                    "description": first_dict.get("description", ""),
                    "instruction": first_dict.get("instruction", ""),
                    "hyperparams": hyperparams,
                },
            ):
                yield resp

            async for _ in rest_iterator:  # consume any leftovers
                pass
            return

        # ========== STREAMING MULTI-MESSAGE ==========
        chunks_by_file: Dict[str, Dict[int, str]] = defaultdict(dict)
        metadata: Dict[str, str] = {}

        async def _process(req: inference_pb2.InferenceRequest) -> None:
            d = _to_dict(req)

            fname = d.get("file_name")
            if not fname:
                # also capture any job-level fields sent on non-chunk messages
                if not metadata and d:
                    metadata.update(
                        name=d.get("name", ""),
                        description=d.get("description", ""),
                        instruction=d.get("instruction", ""),
                        no_files=d.get("no_files", False),
                    )
                # Always (re)merge hyperparams with any top-level T2T keys we see
                raw_h = d.get("hyperparams", {})
                raw_h = _merge_t2t_fields(raw_h, d)
                metadata["hyperparams"] = {
                    str(k): str(v) for k, v in (raw_h if isinstance(raw_h, dict) else {}).items()
                }
                return

            try:
                idx = int(d.get("chunk_index", 0))
            except ValueError:
                idx = 0
            b64 = d.get("chunk_b64")
            if not b64:
                return

            chunks_by_file[fname][idx] = b64

            if not metadata:  # capture once, from first chunk
                metadata.update(
                    name=d.get("name", ""),
                    description=d.get("description", ""),
                    instruction=d.get("instruction", ""),
                    no_files=d.get("no_files", False),
                )
                raw_h = d.get("hyperparams", {})
                raw_h = _merge_t2t_fields(raw_h, d)
                metadata["hyperparams"] = {
                    str(k): str(v) for k, v in (raw_h if isinstance(raw_h, dict) else {}).items()
                }

        await _process(first_req)
        async for r in rest_iterator:
            await _process(r)

        knowledge_items: List[str] = []

        for fname, parts in chunks_by_file.items():
            ordered_bytes = b"".join(base64.b64decode(parts[idx]) for idx in sorted(parts))
            b64_full = base64.b64encode(ordered_bytes).decode("ascii")
            knowledge_items.append(f"b64:{fname}:{b64_full}")

            tmp_path = Path(tempfile.gettempdir()) / fname
            tmp_path.write_bytes(ordered_bytes)
            _LOG.info("Re-assembled %s (%d chunks, %d bytes)", fname, len(parts), tmp_path.stat().st_size)

        knowledge_items = [_maybe_fix_extension(x) for x in knowledge_items]

        # ── MOD: allow blank runs – skip error when no knowledge ──
        if not knowledge_items and not metadata.get("no_files"):
            knowledge_items = []

        async for resp in self._start_train_job(
            model_name=first_req.model,
            knowledge_items=knowledge_items,
            meta=metadata,
        ):
            yield resp

    # ------------------------------------------------------------------ #
    # INTERNAL: launch trainer and stream progress
    # ------------------------------------------------------------------ #

    async def _start_train_job(
        self,
        model_name: str,
        knowledge_items: List[str],
        meta: Dict,
    ) -> AsyncIterator[inference_pb2.InferenceResponse]:
        train_req = trainer_pb2.TrainRequest(
            model_name=model_name,
            knowledge=knowledge_items,
            hyperparams=meta.get("hyperparams", {}),
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            instruction=meta.get("instruction", ""),
        )

        chan = aio.insecure_channel("localhost:9000", options=_GRPC_OPTS)
        stub = trainer_pb2_grpc.TrainerStub(chan)

        try:
            async for prog in stub.StartJob(train_req):
                yield inference_pb2.InferenceResponse(
                    raw=_dumps(
                        {
                            "job_id": prog.job_id,
                            "stage": trainer_pb2.TrainProgress.Stage.Name(prog.stage),
                            "step": prog.step,
                            "total_steps": prog.total_steps,
                            "loss": prog.loss,
                            "metric": prog.metric,
                            "message": prog.message,
                        }
                    ),
                    error="",
                )
        except grpc.RpcError as exc:
            yield inference_pb2.InferenceResponse(raw=b"", error=exc.details() or "trainer failed")

    # ------------------------------------------------------------------ #
    # DB QUERY HANDLER
    # ------------------------------------------------------------------ #

    def _handle_db(self, q: inference_pb2.DBQuery) -> inference_pb2.InferenceResponse:
        if q.token != os.getenv(_ENV_TOKEN_KEY, ""):
            return inference_pb2.InferenceResponse(error="UNAUTHENTICATED")

        sql_lower = q.sql.lstrip().lower()
        try:
            if sql_lower.startswith("select"):
                rows = run_query(q.sql, q.params)
                payload = {"rows": rows}
            elif sql_lower.startswith(("insert", "update", "delete")):
                affected = exec_sql(q.sql, q.params)
                payload = {"rows_affected": affected}
            else:
                raise ValueError("Unsupported SQL command")

            return inference_pb2.InferenceResponse(raw=_dumps(payload), error="")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("DB query failed: %s", exc)
            return inference_pb2.InferenceResponse(error=len(str(exc)) and str(exc) or "DB query failed")

    # ------------------------------------------------------------------ #
    # ADMIN HANDLER
    # ------------------------------------------------------------------ #

    def _handle_admin(self, req: inference_pb2.InferenceRequest) -> inference_pb2.InferenceResponse:
        action = req.model[len(_ADMIN_PREFIX) :]

        # ––– parse admin payload (tiny) ––––––––––––––––––––––––––––––––
        payload_bytes = req.raw if req.HasField("raw") else b""
        try:
            payload = _loads(payload_bytes) if payload_bytes else {}
        except Exception:
            try:
                payload = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
            except Exception:
                payload = {}

        try:
            if action == "list":
                result = _ADMIN.list_enabled()
            elif action == "unload":
                result = _ADMIN.unload(payload)
            elif action == "load":
                if "transaction" in payload:
                    result = _ADMIN.undo(payload["transaction"])
                else:
                    result = _ADMIN.load(payload)
            else:
                raise ValueError(f"Unsupported admin action {action}")

            return inference_pb2.InferenceResponse(raw=_dumps(result), error="")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Admin action failed: %s", exc)
            return inference_pb2.InferenceResponse(error=str(exc))

    # ------------------------------------------------------------------ #
    # METRICS HANDLER
    # ------------------------------------------------------------------ #
    def _handle_metrics(self, req: inference_pb2.InferenceRequest) -> inference_pb2.InferenceResponse:

        try:
            result = calculate_mesh()
            return inference_pb2.InferenceResponse(raw=_dumps(result), error="")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Metrics action failed: %s", exc)
            return inference_pb2.InferenceResponse(error=str(exc))



def register(server: aio.Server) -> None:
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
