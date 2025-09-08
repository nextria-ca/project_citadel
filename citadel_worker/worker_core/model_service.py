# citadel_worker/worker_core/model_service.py
# ============================================================================
#  Async-native ModelService for gRPC-aio – protobuf one-of edition
#  • env-tuned thread / process pools (CITADEL_MAX_THREADS / CITADEL_MAX_PROCS)
#  • struct-bytes fast-path (skip MessageToDict unless the model needs dict)
#  • 1-second cache for gpu_mem_info() so streams don’t hammer NVML
# ============================================================================

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import inspect
import json as _std_json
import os
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Tuple

from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from proto.python import inference_pb2

try:
    import orjson as _json
except ModuleNotFoundError:  # fallback: stdlib json
    _json = _std_json

from citadel_shared.citadel_model import CitadelModel
from citadel_shared.logging_setup import get_logger
from citadel_shared.model_wrappers import InProcModel, SubprocModel, MODELS_ROOT
from citadel_shared.os_utils import gpu_mem_info

# ─────────────────────────── globals / constants ────────────────────────────
_LOG                = get_logger(__name__)
_EWMA_ALPHA         = 0.10
_DEFAULT_CAPACITY   = 1
_DEFAULT_SCORE_A    = 1.0
_DEFAULT_SCORE_B    = 0.5
_DEFAULT_TIMEOUT_MS = int(os.getenv("CITADEL_DEFAULT_TIMEOUT_MS", "0"))
_BYTES_PER_UNIT     = 32_000            # heuristic “work-unit” size
_DEFAULT_SPU        = 0.25              # secs per work-unit baseline

# ─────────────────────────── env-tuned executors ────────────────────────────
_MAX_THREADS = int(os.getenv("CITADEL_MAX_THREADS", "0") or 0) or (os.cpu_count() or 4)
_MAX_PROCS   = int(os.getenv("CITADEL_MAX_PROCS", "0") or 0)   or max(1, (os.cpu_count() or 4) // 2)

_MAX_THREADS = max(2, min(_MAX_THREADS, 32))
_MAX_PROCS   = max(1, min(_MAX_PROCS, 32))

_EXECUTOR         = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_THREADS,
                                                          thread_name_prefix="citadel-io")
_PROCESS_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=_MAX_PROCS)

# ─────────────────────────── helpers ----------------------------------------
def _dumps_str(obj) -> str:
    data = (
        _json.dumps(obj, option=_json.OPT_NON_STR_KEYS)
        if _json is not _std_json
        else _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    )
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data


async def _acall_with_timeout(
    fn, *args, timeout_ms: int = 0, cpu_bound: bool = False
):
    """
    Run *fn* in the appropriate pool:

    • cpu_bound=False (default) → shared ThreadPool (good for GPU/I-O work).
    • cpu_bound=True            → shared ProcessPool (true CPU parallelism).

    Call-sites opt-in to ProcessPool by passing cpu_bound=True.
    """
    loop     = asyncio.get_running_loop()
    executor = _PROCESS_EXECUTOR if cpu_bound else _EXECUTOR
    fut      = loop.run_in_executor(executor, fn, *args)
    return (
        await asyncio.wait_for(fut, timeout_ms / 1000)
        if timeout_ms > 0
        else await fut
    )


def _guess_work_units(size_bytes: int | None) -> int:
    return 1 if not size_bytes else max(1, size_bytes // _BYTES_PER_UNIT)

# ────────────────────── cached GPU-mem helper (1-s TTL) ─────────────────────
_GPU_CACHE: dict[int, Tuple[float, Tuple[int, int]]] = {}
def _gpu_mem_info_cached(gpu_id: int | None) -> Tuple[int, int]:
    if gpu_id is None:
        return (0, 0)
    now = time.time()
    ts, val = _GPU_CACHE.get(gpu_id, (0.0, (0, 0)))
    if now - ts < 1.0:
        return val
    val = gpu_mem_info(gpu_id)
    _GPU_CACHE[gpu_id] = (now, val)
    return val

# ─────────────────────────── replica bookkeeping ────────────────────────────
@dataclasses.dataclass
class _ReplicaState:
    replica: CitadelModel
    gpu_id: int | None
    capacity: int
    score_weights: dict
    timeout_ms: int
    queue_depth: int = 0
    in_flight: int = 0
    ewma_latency_ms: float = 0.0
    queued_work_units: int = 0
    running_work_units: int = 0
    processing_spu: float = _DEFAULT_SPU
    token_pool: asyncio.Semaphore = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.token_pool = asyncio.Semaphore(self.capacity)

    @property
    def free_tokens(self) -> int:
        return self.token_pool._value  # type: ignore[attr-defined]

    @property
    def score(self) -> float:
        a = self.score_weights.get("lat", _DEFAULT_SCORE_A)
        b = self.score_weights.get("queue", _DEFAULT_SCORE_B)
        lat_q = a * self.ewma_latency_ms + b * self.queue_depth
        backlog = (self.queued_work_units + self.running_work_units) * self.processing_spu
        return lat_q + backlog

# ─────────────────────────── replica discovery ──────────────────────────────
def _load_replicas() -> Dict[str, List[CitadelModel]]:
    """
    Discover replicas in two places:

    1. Classic layout:   models/<model-dir>/{config.json,model.py}
    2. Third-party hub:  models/thirdparty/config.json
       → either {"thirdparty": [ ... ]} or
         { "openai_thirdparty": { ... }, "ollama_thirdparty": { ... }, ... }
    """
    registry: Dict[str, List[CitadelModel]] = {}
    futures: List[concurrent.futures.Future] = []

    def _load_one(name, d, cfg, idx, gpu_id):
        try:
            _LOG.info("↻ Loading %s replica #%d (GPU %s)…", name, idx, gpu_id)
            cls = SubprocModel if cfg.get("conda_env") else InProcModel
            mdl = cls(str(d), cfg)
            mdl.init()
            mdl.gpu_id = gpu_id
            _LOG.info("✓ Loaded %s replica #%d (GPU %s)", name, idx, gpu_id)
            return name, mdl, None
        except Exception as exc:
            return name, None, exc

    gpu_ids_seen: set[int | None] = set()
    model_dirs: List[Tuple] = []

    # ── 1) classic directories ────────────────────────────────────────
    for d in MODELS_ROOT.iterdir():
        if not d.is_dir() or d.name == "thirdparty":
            continue
        cfg_file, model_py = d / "config.json", d / "model.py"
        if not (cfg_file.exists() and model_py.exists()):
            continue
        cfg_master = _json.loads(cfg_file.read_text())
        name       = cfg_master.get("name", d.name)
        gpu_ids    = cfg_master.get("gpu_ids") or [None]
        per_gpu    = cfg_master.get("model_per_gpu", 1)
        model_dirs.append((name, d, cfg_master, gpu_ids, per_gpu))
        gpu_ids_seen.update(g for g in gpu_ids if g is not None)

    # ── 2) shared third-party wrappers ─────────────────────────────────
    tp_root   = MODELS_ROOT / "thirdparty"
    tp_config = tp_root / "config.json"
    if tp_config.exists():
        try:
            tp_spec = _json.loads(tp_config.read_text())
        except Exception as exc:
            _LOG.error("Failed to parse third-party config %s: %s", tp_config, exc)
            tp_spec = {}

        # support both {"thirdparty":[...]} and direct map {name: {...}, ...}
        raw_entries = tp_spec.get("thirdparty")
        if isinstance(raw_entries, list):
            entries = raw_entries
        else:
            # treat every top-level key as an entry
            entries = []
            for key, entry in tp_spec.items():
                if not isinstance(entry, dict):
                    continue
                e = dict(entry)  # copy so we can inject name
                e.setdefault("name", key)
                entries.append(e)

        for entry in entries:
            name    = entry.get("name")
            if not name:
                continue
            gpu_ids = entry.get("gpu_ids") or [None]
            per_gpu = entry.get("model_per_gpu", 1)
            model_dirs.append((name, tp_root, entry, gpu_ids, per_gpu))
            gpu_ids_seen.update(g for g in gpu_ids if g is not None)

    # ── spin up replicas in parallel ──────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids_seen) or 4) as tp:
        for name, d, cfg_master, gpu_ids, per_gpu in model_dirs:
            for gpu in gpu_ids:
                for idx in range(per_gpu):
                    cfg = dict(cfg_master)
                    cfg["gpu_id"] = gpu
                    futures.append(tp.submit(_load_one, name, d, cfg, idx, gpu))

        temp: Dict[str, List[CitadelModel]] = {}
        for fut in concurrent.futures.as_completed(futures):
            name, mdl, exc = fut.result()
            temp.setdefault(name, [])
            if mdl:
                temp[name].append(mdl)
            else:
                _LOG.exception("✗ Failed to load %s: %s", name, exc)

        for name, reps in temp.items():
            if not reps:
                _LOG.error("No working replicas for model %s", name)
            registry[name] = reps

    return registry


_REGISTRY = _load_replicas()

def available_models() -> List[dict]:
    return [{"name": n, "gpu_id": getattr(r, "gpu_id", None)}
            for n, reps in _REGISTRY.items() for r in reps]

# ─────────────────────────── async ModelService ─────────────────────────────
class ModelService:
    """Async replacement for the old sync ModelService, with latency tricks."""

    def __init__(self) -> None:
        self._states: Dict[str, List[_ReplicaState]] = {}
        self._blobs: Dict[str, bytearray] = {}  # session assemblies

        for name, reps in _REGISTRY.items():
            lst: List[_ReplicaState] = []
            for rep in reps:
                cfg = getattr(rep, "config", {})
                lst.append(
                    _ReplicaState(
                        replica       = rep,
                        gpu_id        = getattr(rep, "gpu_id", None),
                        capacity      = int(cfg.get("concurrency", _DEFAULT_CAPACITY)),
                        score_weights = dict(cfg.get("score_weights", {"lat": 1.0, "queue": 0.5})),
                        timeout_ms    = int(cfg.get("timeout", _DEFAULT_TIMEOUT_MS) or 0),
                    )
                )
            self._states[name] = lst
        
    async def cache_clear_loop(self, interval_s: int | None = None) -> None:
        """
        Best-effort: once per *interval_s* (default 3600s), clear CUDA caches
        on idle replicas. Skips replicas with in-flight work.
        """
        try:
            interval = int(interval_s or int(os.getenv("CITADEL_CUDA_CACHE_INTERVAL_S", "3600")))
        except Exception:
            interval = 3600
        interval = max(60, interval)  # safety floor

        _LOG.info("CUDA cache clear loop enabled – every %d s", interval)

        while True:
            try:
                for states in self._states.values():
                    for st in states:
                        # avoid interfering with active work
                        if st.in_flight > 0 or st.queue_depth > 0:
                            continue
                        try:
                            await _acall_with_timeout(st.replica.clear_cache, timeout_ms=3000)
                            _LOG.debug("Cleared CUDA cache on replica (gpu=%s)", st.gpu_id)
                        except Exception:
                            _LOG.debug("clear_cache skipped/failed on replica (gpu=%s)", st.gpu_id)
            except Exception:
                _LOG.debug("cache_clear_loop iteration failed", exc_info=True)

            await asyncio.sleep(interval)

    # ────────────────────────── stats (cached GPU) ───────────────────────────
    def get_replica_stats(self) -> List[dict]:
        stats: List[dict] = []
        for states in self._states.values():
            for st in states:
                tot, free = _gpu_mem_info_cached(st.gpu_id)
                stats.append(
                    {
                        "queue_depth"      : st.queue_depth,
                        "in_flight"        : st.in_flight,
                        "ewma_latency_ms"  : st.ewma_latency_ms,
                        "queued_work_units": st.queued_work_units + st.running_work_units,
                        "processing_spu"   : st.processing_spu,
                        "gpu_id"           : st.gpu_id or 0,
                        "gpu_total_mb"     : tot,
                        "gpu_free_mb"      : free,
                    }
                )
        return stats

    # ---------------------------------------------------------------- helpers
    def _pick_state(self, model: str) -> _ReplicaState:
        states = self._states[model]
        live   = [s for s in states if s.free_tokens > 0]
        return min(live or states, key=lambda s: s.score)

    def _update_latency(self, st: _ReplicaState, ms: float) -> None:
        st.ewma_latency_ms = ms if st.ewma_latency_ms == 0 else (1 - _EWMA_ALPHA) * st.ewma_latency_ms + _EWMA_ALPHA * ms

    # ---------------------------------------------------------------- handler
    async def handle_stream(
        self,
        req_it: AsyncIterator[inference_pb2.InferenceRequest],
    ) -> AsyncIterator[inference_pb2.InferenceResponse]:
        _LOG.debug("handle_stream[aio]: iterator opened")
        frame_idx = 0

        async for req in req_it:
            # session re-assembly (raw-bytes mode)
            if req.session_id:
                buf = self._blobs.setdefault(req.session_id, bytearray())
                if req.HasField("raw") and req.raw:
                    buf.extend(req.raw)
                if not req.last_chunk:
                    frame_idx += 1
                    continue
                req.raw = bytes(buf)
                del self._blobs[req.session_id]

            if req.model not in self._states:
                err = f"Unknown model '{req.model}'"
                _LOG.error(err)
                yield inference_pb2.InferenceResponse(raw=b"", error=err)
                frame_idx += 1
                continue

            st  = self._pick_state(req.model)
            rep = st.replica

            payload_bytes = (
                req.struct_data.SerializeToString()
                if req.HasField("struct_data")
                else bytes(req.raw)
            )
            work_units = _guess_work_units(len(payload_bytes))

            # queue bookkeeping
            st.queue_depth += 1
            st.queued_work_units += work_units
            await st.token_pool.acquire()
            st.queue_depth      -= 1
            st.queued_work_units -= work_units
            st.running_work_units += work_units
            st.in_flight += 1

            t_start = time.time()
            try:
                # subproc direct call (no streaming)
                if isinstance(rep, SubprocModel) and not req.stream:
                    try:
                        result = await _acall_with_timeout(
                            rep.execute, payload_bytes,
                            timeout_ms=st.timeout_ms
                        )
                        yield _make_rsp(result, st.gpu_id, t_start)
                    except Exception as exc:
                        err = _json.dumps(
                            {"gpu_id": st.gpu_id, "latency": time.time() - t_start, "error": str(exc)}
                        )
                        yield inference_pb2.InferenceResponse(raw=b"", error=err)
                    frame_idx += 1
                    continue

                # decide input form (bytes vs dict)
                expects_dict = getattr(rep, "input_format", "bytes") == "dict"
                user_input   = (
                    MessageToDict(req.struct_data, preserving_proto_field_name=True)
                    if expects_dict and req.HasField("struct_data")
                    else payload_bytes
                )

                # STREAMING path
                if req.stream:
                    gen     = rep.stream_execute(user_input)
                    timeout = st.timeout_ms

                    async def deadline_passed() -> bool:
                        return timeout > 0 and (time.time() - t_start) * 1000 > timeout

                    agen = gen if inspect.isasyncgen(gen) else _sync_to_async(gen)
                    part_idx = 0
                    async for part in agen:
                        if await deadline_passed():
                            yield inference_pb2.InferenceResponse(
                                raw=b"", error=f"TIMEOUT after {timeout} ms"
                            )
                            break
                        yield _make_rsp(part, st.gpu_id, t_start)
                        part_idx += 1
                # single-shot path
                else:
                    result = await _acall_with_timeout(
                        rep.execute, user_input, timeout_ms=st.timeout_ms
                    )
                    yield _make_rsp(result, st.gpu_id, t_start)

            finally:
                lat_ms = (time.time() - t_start) * 1000.0
                self._update_latency(st, lat_ms)

                st.running_work_units -= work_units
                st.in_flight          -= 1
                st.token_pool.release()

                elapsed_s = lat_ms / 1000.0
                st.processing_spu = (
                    (1 - _EWMA_ALPHA) * st.processing_spu
                    + _EWMA_ALPHA * (elapsed_s / work_units)
                )
                frame_idx += 1

def _sync_to_async(gen):
    async def _wrap():
        for part in gen:
            yield part
    return _wrap()

def _make_rsp(obj, gpu_id, t0):
    tot_mb, free_mb = _gpu_mem_info_cached(gpu_id)
    meta = _dumps_str(
        {
            "gpu_id"      : gpu_id,
            "latency"     : time.time() - t0,
            "gpu_total_mb": tot_mb,
            "gpu_free_mb" : free_mb,
        }
    )

    if isinstance(obj, (bytes, bytearray)):
        return inference_pb2.InferenceResponse(raw=bytes(obj), meta_json=meta)
    if isinstance(obj, str):
        return inference_pb2.InferenceResponse(raw=obj.encode("utf-8"), meta_json=meta)
    if isinstance(obj, (dict, list)):
        struct_msg = Struct()
        json_format.ParseDict(obj, struct_msg, ignore_unknown_fields=False)
        return inference_pb2.InferenceResponse(struct_data=struct_msg, meta_json=meta)

    return inference_pb2.InferenceResponse(
        raw=str(obj).encode("utf-8"), meta_json=meta
    )
