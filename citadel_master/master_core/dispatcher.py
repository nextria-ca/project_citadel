"""Request router on the master – updated for protobuf one-of."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
from typing import AsyncIterator, List, Set

import grpc
from grpc import aio

from citadel_master.grpc_services.metrics_servicer import (
    MetricsJobMonitorService
)
from proto.python import inference_pb2, registry_pb2_grpc
from master_core.worker_registry import REGISTRY
from citadel_master.stats import AGGREGATOR

_LOG = logging.getLogger(__name__)

_MAX_FAILOVER_ATTEMPTS   = int(os.getenv("MAX_FAILOVER_ATTEMPTS", "3"))
_FORWARD_RPC_TIMEOUT_SEC = float(os.getenv("FORWARD_RPC_TIMEOUT_SEC", "200"))
_MAX_CHANNELS            = 64

# ────────────────── optimistic backlog accounting ──────────────────
_BYTES_PER_UNIT = 32_000            # 32 kB ≈ 1 audio-second

metrics_job_monitor_service = MetricsJobMonitorService()

def _guess_work_units(size_bytes: int | None) -> int:
    """32 kB ⇒ 1 work-unit."""
    if not size_bytes:
        return 1
    return max(1, size_bytes // _BYTES_PER_UNIT)


def _payload_size(req: inference_pb2.InferenceRequest) -> int:
    if req.HasField("raw"):
        return len(req.raw)
    if req.HasField("struct_data"):
        return len(req.struct_data.SerializeToString())
    return 0


@functools.lru_cache(maxsize=_MAX_CHANNELS)
def _get_aio_channel(endpoint: str) -> aio.Channel:
    return aio.insecure_channel(endpoint)


def _healthy_workers_for(model: str, exclude: Set[str]) -> List[str]:
    if hasattr(REGISTRY, "inference_workers"):
        all_workers = [
            w
            for w in REGISTRY.inference_workers()
            if model in REGISTRY._db[w].models
        ]
    else:  # legacy fallback
        all_workers = REGISTRY.workers_for(model)
    return [w for w in all_workers if w not in exclude]


_rr_ptr: dict[str, int] = {}  # per-model round-robin pointer


def _score_worker(wid: str) -> float:
    """Estimate **seconds** of remaining work for *wid* (lower is better)."""
    reps = REGISTRY.replica_stats(wid)
    if not reps:
        return float("inf")
    units = sum(r.get("queued_work_units", 0) for r in reps)
    spu   = max(r.get("processing_spu", 1.0) for r in reps)
    return units * spu


def _predicted_score(wid: str, inc_units: int) -> float:
    """Score if *this* request (inc_units) were added to worker *wid*."""
    base   = _score_worker(wid)
    reps   = REGISTRY.replica_stats(wid)
    spu    = max(r.get("processing_spu", 1.0) for r in reps) if reps else 1.0
    return base + inc_units * spu


def _next_p2c(model: str, exclude: Set[str], inc_units: int) -> str | None:
    """
    Weighted Power-of-Two-Choices using **predicted** backlog
    (seconds-of-work after accepting *this* request).
    """
    pool = _healthy_workers_for(model, exclude)
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    w1, w2 = random.sample(pool, 2)
    return w1 if _predicted_score(w1, inc_units) <= _predicted_score(w2, inc_units) else w2


# ───────────────────────── streaming forwarder ─────────────────────────
async def _forward_stream(
    wid: str,
    model: str,
    requests: List[inference_pb2.InferenceRequest],
) -> AsyncIterator[inference_pb2.InferenceResponse]:
    """
    Stream *requests* to worker *wid* and yield its responses one-by-one.
    Raises on failure so the caller can retry elsewhere.
    """
    chan = _get_aio_channel(REGISTRY.endpoint(wid))
    stub = registry_pb2_grpc.WorkerForwardStub(chan)

    level = requests[0].level if requests and hasattr(requests[0], "level") else None
    stat = AGGREGATOR.record_queued(
        model,
        wid,
        sum(_payload_size(r) for r in requests),
        level=level,
    )

    async def _iter() -> AsyncIterator[inference_pb2.InferenceRequest]:
        for r in requests:
            yield r

    # ── set deadline only for non-streaming calls ───────────────────────
    if requests and getattr(requests[0], "stream", False):
        timeout_kw = {}
    else:
        timeout_kw = (
            {"timeout": _FORWARD_RPC_TIMEOUT_SEC}
            if _FORWARD_RPC_TIMEOUT_SEC > 0
            else {}
        )

    try:
        rsp_stream = stub.Forward(_iter(), **timeout_kw)
        if stat is not None:
            AGGREGATOR.record_started(stat)

        gpu_stats: dict | None = None

        async for rsp in rsp_stream:
            yield rsp

            # footer-style GPU telemetry
            if getattr(rsp, "meta_json", ""):
                try:
                    gpu_stats = json.loads(rsp.meta_json)
                except (TypeError, json.JSONDecodeError):
                    gpu_stats = None

        if stat is not None:
            AGGREGATOR.record_completed(stat, gpu_stats=gpu_stats or {})
    except Exception:
        if stat is not None:
            AGGREGATOR.record_completed(stat)
        raise


async def _forward_stream_iter(
    wid: str,
    model: str,
    first: inference_pb2.InferenceRequest,
    rest_it: AsyncIterator[inference_pb2.InferenceRequest],
) -> AsyncIterator[inference_pb2.InferenceResponse]:
    """
    Forward a *streaming iterator* (first + rest) to worker *wid* and yield responses.
    Mirrors _forward_stream() but without buffering the entire request stream up front.
    """
    chan = _get_aio_channel(REGISTRY.endpoint(wid))
    stub = registry_pb2_grpc.WorkerForwardStub(chan)

    level = getattr(first, "level", None)
    stat  = AGGREGATOR.record_queued(model, wid, _payload_size(first), level=level)

    async def _iter():
        # pass frames through immediately
        yield first
        async for r in rest_it:
            yield r

    # only set a deadline for non-streaming requests
    timeout_kw = {} if getattr(first, "stream", False) else (
        {"timeout": _FORWARD_RPC_TIMEOUT_SEC} if _FORWARD_RPC_TIMEOUT_SEC > 0 else {}
    )

    try:
        rsp_stream = stub.Forward(_iter(), **timeout_kw)
        if stat is not None:
            AGGREGATOR.record_started(stat)

        gpu_stats: dict | None = None
        async for rsp in rsp_stream:
            yield rsp
            if getattr(rsp, "meta_json", ""):
                try:
                    gpu_stats = json.loads(rsp.meta_json)
                except (TypeError, json.JSONDecodeError):
                    gpu_stats = None

        if stat is not None:
            AGGREGATOR.record_completed(stat, gpu_stats=gpu_stats or {})
    except Exception:
        if stat is not None:
            AGGREGATOR.record_completed(stat)  # ensure queue accounting decrements
        raise


async def forward_stream_async(
    req_it: AsyncIterator[inference_pb2.InferenceRequest],
) -> AsyncIterator[inference_pb2.InferenceResponse]:
    """
    Entry-point called by the master’s gRPC server.
    Now *pass-throughs* frames as they arrive, avoiding head-of-line blocking.
    For streaming calls, failover is attempted only until the first response is produced.
    """
    aiter = req_it.__aiter__()
    model = None
    wid = None
    try:
        first = await aiter.__anext__()
    except StopAsyncIteration:
        return

    model     = first.model
    inc_units = _guess_work_units(_payload_size(first))
    attempts  = 0
    tried: Set[str] = set()

    while attempts < _MAX_FAILOVER_ATTEMPTS:
        wid = _next_p2c(model, tried, inc_units)
        if wid is None:
            break

        _LOG.info(
            "P-2-C → dispatching %s to %s (attempt %d, inc_units=%d)",
            model, wid, attempts + 1, inc_units,
        )
        attempts += 1
        tried.add(wid)

        produced = False
        try:
            metrics_job_monitor_service.WorkerStartedJob(wid, model)
            async for rsp in _forward_stream_iter(wid, model, first, aiter):
                produced = True
                yield rsp
            return  # success
        except Exception:
            _LOG.exception("Worker %s failed", wid)
            if produced:
                # We can’t safely replay a partially-consumed client stream.
                # Propagate a clear error and stop retries.
                yield inference_pb2.InferenceResponse(
                    error=f"Worker failed mid-stream for model '{model}' (attempt {attempts}); cannot fail over safely"
                )
                return
        finally:
            metrics_job_monitor_service.WorkerFinishedJob(wid, model)
            # else: try next worker

    _LOG.error("All replicas failed for model %s for reason: %s", model)
    yield inference_pb2.InferenceResponse(
        error=f"All replicas failed for model '{model}' after {attempts} attempts"
    )