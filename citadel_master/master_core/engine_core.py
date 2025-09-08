# citadel_worker/worker_core/model_hub.py
"""Adaptive batching worker (ModelHub) – protobuf one-of edition
   • struct-bytes fast-path (skip JSON parse unless validation requires it)
"""

from __future__ import annotations

import asyncio
import json as _std_json
import os
import pickle
import time
from collections import deque
from pathlib import Path
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    List,
    Tuple,
)

try:
    import orjson as _json                 # ultra-fast JSON if available
except ModuleNotFoundError:
    _json = _std_json

from google.protobuf.json_format import MessageToDict
from proto.python import inference_pb2

from citadel_shared.citadel_model import CitadelModel
from citadel_shared.logging_setup import get_logger
from citadel_shared.model_wrappers import InProcModel, SubprocModel
from citadel_shared.priority import Priority
from citadel_shared.schema_validator import compile_schema, validate

# ─────────────────────── local serializer (str/bytes only) ───────────────────
def _dumps(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")

    if _json is _std_json:                                  # stdlib path
        return _json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    return _json.dumps(obj, option=_json.OPT_NON_STR_KEYS)


def _loads(buf: bytes):
    try:
        return _json.loads(buf) if _json is not _std_json else _json.loads(buf.decode("utf-8"))
    except Exception:
        return buf                                            # leave as-is on failure


MODELS_ROOT             = Path(__file__).parent / "models"
_MAX_BACKLOG_PER_MODEL  = int(os.getenv("MAX_BACKLOG_PER_MODEL", 512))
_EWMA_ALPHA             = 0.2
_LAT_TARGET_MS          = 200
_BATCH_GROWTH_FACTOR    = 1

_LOG = get_logger(__name__)

PRIORITY_NAME_TO_VALUE = {
    "immediate":   Priority.IMMEDIATE,
    "high":        Priority.HIGH,
    "medium":      Priority.MEDIUM,
    "normal":      Priority.NORMAL,
    "system_slow": Priority.SYSTEM_SLOW,
}

# --------------------------------------------------------------------------- #
#  Multi-level priority queue                                                 #
# --------------------------------------------------------------------------- #
class RankedRequestQueue:
    def __init__(self) -> None:
        self._queues: Dict[
            Priority,
            Deque[Tuple[inference_pb2.InferenceRequest, bytes, asyncio.Queue[bytes | None]]],
        ] = {p: deque() for p in Priority}
        self._condition = asyncio.Condition()

    @staticmethod
    def _priority_of(req: inference_pb2.InferenceRequest | dict) -> Priority:
        if hasattr(req, "level") and req.level:
            return PRIORITY_NAME_TO_VALUE.get(str(req.level).lower(), Priority.NORMAL)
        if isinstance(req, dict) and "level" in req:
            return PRIORITY_NAME_TO_VALUE.get(str(req["level"]).lower(), Priority.NORMAL)
        return Priority.NORMAL

    def depth(self) -> int:
        return sum(len(q) for q in self._queues.values())

    async def put(
        self,
        req_msg: inference_pb2.InferenceRequest,
        req_bytes: bytes,
        resp_q: asyncio.Queue[bytes | None],
    ) -> None:
        prio = self._priority_of(req_msg)
        async with self._condition:
            self._queues[prio].append((req_msg, req_bytes, resp_q))
            self._condition.notify()

    async def get(
        self,
    ) -> Tuple[inference_pb2.InferenceRequest, bytes, asyncio.Queue[bytes | None]]:
        async with self._condition:
            while True:
                for prio in Priority:
                    if self._queues[prio]:
                        return self._queues[prio].popleft()
                await self._condition.wait()

# --------------------------------------------------------------------------- #
#  Model discovery                                                            #
# --------------------------------------------------------------------------- #
def _scan_models() -> Dict[str, List[CitadelModel]]:
    registry: Dict[str, List[CitadelModel]] = {}

    for d in MODELS_ROOT.iterdir():
        if not d.is_dir():
            continue
        cfg_file, model_py = d / "config.json", d / "model.py"
        if not (cfg_file.exists() and model_py.exists()):
            continue

        cfg_master = _json.loads(cfg_file.read_text())
        name       = cfg_master.get("name", d.name)
        gpu_ids    = cfg_master.get("gpu_ids") or [None]
        per_gpu    = cfg_master.get("model_per_gpu", 1)

        if (inp_schema := cfg_master.get("inputs")):
            try:
                cfg_master["_inputs_validator"] = compile_schema(inp_schema)
            except Exception as exc:                                  # noqa: BLE001
                _LOG.error("✖ Failed to compile schema for %s: %s", name, exc)
                cfg_master["_inputs_validator"] = None

        replicas: List[CitadelModel] = []
        for gpu_id in gpu_ids:
            for replica_idx in range(per_gpu):
                cfg           = dict(cfg_master)
                cfg["gpu_id"] = gpu_id
                try:
                    mdl = (
                        SubprocModel(str(d), cfg)
                        if cfg.get("conda_env")
                        else InProcModel(str(d), cfg)
                    )
                    mdl.init()
                    replicas.append(mdl)
                    _LOG.info("Loaded %s replica #%d on GPU %s", name, replica_idx, gpu_id)
                except Exception as exc:                              # noqa: BLE001
                    _LOG.exception("Failed to load %s (GPU %s): %s", name, gpu_id, exc)

        if not replicas:
            _LOG.error("No working replicas for model %s", name)

        registry[name] = replicas
    return registry

# --------------------------------------------------------------------------- #
#  Helpers so legacy inference_servicer_engine keeps compiling                #
# --------------------------------------------------------------------------- #
def enqueue_stream(req_it: Iterator[inference_pb2.InferenceRequest]) -> Iterator[inference_pb2.InferenceRequest]:
    return req_it


async def enqueue_stream_async(req_it: AsyncIterator[inference_pb2.InferenceRequest]) -> AsyncIterator[inference_pb2.InferenceRequest]:
    async for r in req_it:
        yield r

# --------------------------------------------------------------------------- #
#  ModelHub – one instance per gRPC server                                    #
# --------------------------------------------------------------------------- #
class ModelHub:
    def __init__(self) -> None:
        self.registry: Dict[str, List[CitadelModel]] = _scan_models()
        self._queues: Dict[str, RankedRequestQueue] = {
            m: RankedRequestQueue() for m in self.registry
        }
        self._workers_started = False

    # ........................................................................ #
    #  Public entry-point                                                     #
    # ........................................................................ #
    async def infer(self, req_bytes: bytes) -> AsyncGenerator[bytes, None]:
        await self._maybe_start_workers()

        req_msg = inference_pb2.InferenceRequest.FromString(req_bytes)
        model   = req_msg.model

        if model not in self.registry or not self.registry[model]:
            yield self._make_error_rsp(model, "Unknown model")
            return

        if self._queues[model].depth() >= _MAX_BACKLOG_PER_MODEL:
            yield self._make_error_rsp(
                model,
                "UNAVAILABLE: model queue overloaded (back-off & retry)",
            )
            return

        resp_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        await self._queues[model].put(req_msg, req_bytes, resp_q)

        while True:
            part = await resp_q.get()
            if part is None:
                break
            yield part

    async def _maybe_start_workers(self) -> None:
        if self._workers_started:
            return

        self._workers_started = True
        loop = asyncio.get_running_loop()

        for model in self.registry:
            loop.create_task(self._worker_loop(model))
            _LOG.info("Started worker for %s", model)

    # ----------------------------------------------------------------------- #
    #  Worker loop – adaptive batching                                        #
    # ----------------------------------------------------------------------- #
    async def _worker_loop(self, model_name: str) -> None:
        q        = self._queues[model_name]
        replicas = self.registry[model_name]

        cycle: asyncio.Queue[CitadelModel] = asyncio.Queue()
        for r in replicas:
            cycle.put_nowait(r)

        cfg                = replicas[0].config
        static_max_bs      = max(int(cfg.get("batch_size", 1) or 1), 1)
        timeout_ms         = int(cfg.get("batch_timeout_ms", 0) or 0)
        batching_on        = static_max_bs > 1

        dynamic_bs      = 1 if not batching_on else static_max_bs
        ewma_latency_ms = 0.0

        _LOG.info(
            "Worker %s – batching=%s (max=%d, timeout=%d ms)",
            model_name,
            "ON" if batching_on else "OFF",
            static_max_bs,
            timeout_ms,
        )

        while True:
            req_msg, req_bytes, resp_q = await q.get()
            first_is_stream = bool(req_msg.stream)

            batch: List[
                Tuple[inference_pb2.InferenceRequest, bytes, asyncio.Queue[bytes | None]]
            ] = [(req_msg, req_bytes, resp_q)]

            if batching_on and dynamic_bs > 1 and not first_is_stream:
                deadline = time.perf_counter() + timeout_ms / 1000
                while len(batch) < dynamic_bs:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(q.get(), timeout=remaining)
                        if nxt[0].stream:                  # don’t batch streams
                            await q.put(*nxt)
                            break
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

            t0 = time.perf_counter()
            try:
                if len(batch) == 1 or first_is_stream or not batching_on:
                    await self._process_single(batch[0], cycle)
                else:
                    await self._process_batch(batch, cycle)
            except Exception as exc:                       # noqa: BLE001
                _LOG.exception("Worker for %s blew up: %s", model_name, exc)
                for _, _, rq in batch:
                    await rq.put(
                        inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                    )
                    await rq.put(None)
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if ewma_latency_ms == 0.0:
                    ewma_latency_ms = latency_ms
                else:
                    ewma_latency_ms = (1 - _EWMA_ALPHA) * ewma_latency_ms + _EWMA_ALPHA * latency_ms

                backlog = q.depth()
                if (
                    ewma_latency_ms < _LAT_TARGET_MS
                    and backlog > dynamic_bs
                    and dynamic_bs < static_max_bs
                ):
                    dynamic_bs = min(static_max_bs, dynamic_bs + _BATCH_GROWTH_FACTOR)
                elif ewma_latency_ms > _LAT_TARGET_MS * 1.5 and dynamic_bs > 1:
                    dynamic_bs = max(1, dynamic_bs - _BATCH_GROWTH_FACTOR)

    # ----------------------------------------------------------------------- #
    #  Internal helpers                                                       #
    # ----------------------------------------------------------------------- #
    async def _process_single(
        self,
        item: Tuple[inference_pb2.InferenceRequest, bytes, asyncio.Queue[bytes | None]],
        cycle: asyncio.Queue,
    ) -> None:
        req_msg, req_bytes, resp_q = item

        async for part in self._process_request(req_bytes, cycle):
            await resp_q.put(part)
        await resp_q.put(None)

    async def _process_batch(
        self,
        batch: List[
            Tuple[inference_pb2.InferenceRequest, bytes, asyncio.Queue[bytes | None]]
        ],
        cycle: asyncio.Queue,
    ) -> None:
        replica: CitadelModel = await cycle.get()
        await cycle.put(replica)

        validator = replica.config.get("_inputs_validator")
        schema    = replica.config.get("inputs") if validator is None else None
        needs_dict = validator is not None or schema is not None

        inputs: List[object]        = []
        queues: List[asyncio.Queue] = []

        for req_msg, _, resp_q in batch:
            try:
                # ── struct fast-path unless validation needs JSON dict ──
                if req_msg.HasField("struct_data"):
                    user_input = (
                        MessageToDict(req_msg.struct_data, preserving_proto_field_name=True)
                        if needs_dict
                        else req_msg.struct_data.SerializeToString()
                    )
                else:
                    user_input = (
                        _loads(req_msg.raw) if needs_dict else bytes(req_msg.raw)
                    )
            except Exception:
                user_input = pickle.loads(req_msg.raw)       # unsafe fallback

            if needs_dict:
                try:
                    if validator:
                        validator(user_input)
                    elif schema:
                        validate(user_input, schema)
                except Exception as exc:
                    await resp_q.put(
                        inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                    )
                    await resp_q.put(None)
                    continue

            inputs.append(user_input)
            queues.append(resp_q)

        if not inputs:
            return

        try:
            results = (
                replica.execute_batch(inputs)
                if hasattr(replica, "execute_batch")
                else [replica.execute(x) for x in inputs]
            )
        except Exception as exc:
            for q in queues:
                await q.put(
                    inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                )
                await q.put(None)
            return

        for res, q in zip(results, queues):
            try:
                bin_out = res if isinstance(res, (bytes, bytearray)) else _dumps(res)
                await q.put(
                    inference_pb2.InferenceResponse(raw=bin_out, error="").SerializeToString()
                )
            except Exception as exc:
                await q.put(
                    inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                )
            finally:
                await q.put(None)

    async def _process_request(
        self,
        req_bytes: bytes,
        cycle: asyncio.Queue,
    ) -> AsyncGenerator[bytes, None]:
        req = inference_pb2.InferenceRequest.FromString(req_bytes)
        replica: CitadelModel = await cycle.get()
        await cycle.put(replica)

        validator = replica.config.get("_inputs_validator")
        schema    = replica.config.get("inputs") if validator is None else None
        needs_dict = validator is not None or schema is not None

        try:
            if req.HasField("struct_data"):
                user_input = (
                    MessageToDict(req.struct_data, preserving_proto_field_name=True)
                    if needs_dict
                    else req.struct_data.SerializeToString()
                )
            else:
                user_input = _loads(req.raw) if needs_dict else bytes(req.raw)
        except Exception:
            try:
                user_input = _json.loads(req.raw.decode("utf-8"))
            except Exception as exc:                             # noqa: BLE001
                yield inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                return

        if needs_dict:
            try:
                if validator:
                    validator(user_input)
                elif schema:
                    validate(user_input, schema)
            except Exception as exc:
                yield inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
                return

        if req.stream:
            try:
                gen = replica.stream_execute(user_input)
                if not hasattr(gen, "__aiter__"):
                    raise RuntimeError("stream_execute did not return an async generator")
                async for part in gen:
                    bin_out = part if isinstance(part, (bytes, bytearray)) else _dumps(part)
                    yield inference_pb2.InferenceResponse(raw=bin_out, error="").SerializeToString()
            except Exception as exc:                             # noqa: BLE001
                yield inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()
        else:
            try:
                result  = replica.execute(user_input)
                bin_out = result if isinstance(result, (bytes, bytearray)) else _dumps(result)
                yield inference_pb2.InferenceResponse(raw=bin_out, error="").SerializeToString()
            except Exception as exc:                             # noqa: BLE001
                yield inference_pb2.InferenceResponse(raw=b"", error=str(exc)).SerializeToString()

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _make_error_rsp(model: str, msg: str) -> bytes:
        return inference_pb2.InferenceResponse(raw=b"", error=f"{msg} {model}").SerializeToString()

    def __del__(self) -> None:
        for reps in self.registry.values():
            for m in reps:
                m.finalize()
