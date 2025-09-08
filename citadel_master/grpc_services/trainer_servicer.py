from __future__ import annotations
"""
citadel_master/grpc_services/trainer_servicer.py

Smarter trainer router:
  • Scores candidates by predicted backlog (seconds of work) using live
    worker heartbeats (queued_work_units × processing_spu) + an increment
    for *this* job.
  • Soft-leases (worker_id, gpu_id) so two concurrent jobs don't start
    on the same GPU.
  • Failover: if the first target fails before producing any progress,
    try the next best slot.
  • Preserves DB job book-keeping and protobuf surface exactly as-is.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import grpc
from grpc import aio, StatusCode
from google.protobuf.json_format import MessageToDict
from google.protobuf import empty_pb2

from citadel_db.db_helpers import register_job, update_job_status
from master_core.worker_registry import REGISTRY
from proto.python import trainer_pb2, trainer_pb2_grpc

_LOG = logging.getLogger(__name__)

# If a worker exposes a dedicated trainer port, you can still override per-node
# by setting TRAINER_PORT on that worker; otherwise we use the registered port.
_TRAIN_PORT_ENV = "TRAINER_PORT"
MAX_MSG = 64 * 1024 * 1024
_GRPC_OPTS = [
    ("grpc.max_send_message_length", MAX_MSG),
    ("grpc.max_receive_message_length", MAX_MSG),
    ("grpc.keepalive_time_ms", 60_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.http2.min_time_between_pings_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]
_TRAINER_TEE = str(os.getenv("CITADEL_TRAINER_TEE", "0")).lower() not in ("0", "false", "no", "off")

def _chan(endpoint: str) -> aio.Channel:
    # single place to tweak transport; avoids per-call construction differences
    return aio.insecure_channel(endpoint, options=_GRPC_OPTS)
# Tunables (env-overridable)
_MIN_FREE_MB           = int(os.getenv("TRAINER_MIN_FREE_MB", "1024"))      # skip GPUs < this free VRAM
_WORK_UNIT_BYTES       = int(os.getenv("TRAINER_WORK_UNIT_BYTES", "200000"))# ≈ cost hint for scoring
_MAX_FAILOVER_ATTEMPTS = int(os.getenv("TRAINER_FAILOVER_ATTEMPTS", "3"))

# Internal soft-leases to avoid double-scheduling same GPU (router-local)
_LEASE_LOCK = asyncio.Lock()
_LEASED: set[Tuple[str, int]] = set()   # {(worker_id, gpu_id)}


# ───────────────────────── utilities ─────────────────────────

def _estimate_inc_units(req: trainer_pb2.TrainRequest) -> int:
    """
    Heuristic: number of "work-units" implied by the knowledge payload.
    • For b64:… items → approximate bytes = 3/4 * len(b64)
    • For file paths → os.path.getsize
    • Otherwise       → 1 small unit
    """
    total_bytes = 0

    for item in req.knowledge:
        if isinstance(item, str) and item.startswith("b64:"):
            # format "b64:<fname>:<payload>"
            try:
                b64 = item.split(":", 2)[2]
                total_bytes += (len(b64) * 3) // 4
            except Exception:
                total_bytes += _WORK_UNIT_BYTES
        else:
            try:
                if os.path.isfile(item):
                    total_bytes += os.path.getsize(item)
                else:
                    total_bytes += _WORK_UNIT_BYTES
            except Exception:
                total_bytes += _WORK_UNIT_BYTES

    # Always at least one unit
    return max(1, total_bytes // max(1, _WORK_UNIT_BYTES))


def _score_snapshot(s: Dict, inc_units: int) -> float:
    """
    Lower is better.
    Base backlog (seconds) = queued_work_units * processing_spu
    Predicted with this job = base + inc_units * processing_spu
    If free VRAM is low, bias the score upwards (avoid borderline OOM).
    """
    spu     = max(0.001, float(s.get("processing_spu") or 1.0))
    base_s  = float(s.get("queued_work_units") or 0) * spu
    pred_s  = base_s + inc_units * spu

    free_mb = int(s.get("gpu_free_mb") or 0)
    if free_mb < _MIN_FREE_MB:
        pred_s += 60.0  # big penalty, effectively skip unless nothing else

    return pred_s


async def _lease_slot(wid: str, gpu_id: int) -> bool:
    async with _LEASE_LOCK:
        key = (wid, gpu_id)
        if key in _LEASED:
            return False
        _LEASED.add(key)
        return True


async def _release_slot(wid: str, gpu_id: int) -> None:
    async with _LEASE_LOCK:
        _LEASED.discard((wid, gpu_id))


def _best_slots(req: trainer_pb2.TrainRequest) -> List[Tuple[str, int, float]]:
    """
    Return candidate (worker_id, gpu_id, score) sorted by predicted score.
    Uses REGISTRY.replica_stats() which contains gpu_id, queued_work_units, processing_spu,
    gpu_* counters populated by the worker heartbeats.
    """
    inc = _estimate_inc_units(req)
    ranked: List[Tuple[str, int, float]] = []

    for wid in REGISTRY.training_workers():
        for s in REGISTRY.replica_stats(wid):
            gpu_id = int(s.get("gpu_id") or 0)
            score  = _score_snapshot(s, inc)
            ranked.append((wid, gpu_id, score))

    ranked.sort(key=lambda t: t[2])
    return ranked


def _resolve_trainer_endpoint(wid: str) -> str:
    """
    By default use the worker's registered endpoint. If that worker has a dedicated
    trainer port in its environment, prefer it (same host, other port).
    """
    host, port_str = REGISTRY.endpoint(wid).split(":", 1)
    env_port = os.getenv(_TRAIN_PORT_ENV, "")
    port     = int(env_port or port_str)
    return f"{host}:{port}"


# ───────────────────────── gRPC Servicer ─────────────────────────

class TrainerRouterServicer(trainer_pb2_grpc.TrainerServicer):
    """
    Forwards TrainRequest / CancelJob calls to an available trainer daemon
    with backlog-aware selection and early-failover.
    """

    async def StartJob(  # type: ignore[override]
        self,
        request: trainer_pb2.TrainRequest,
        context: aio.ServicerContext,
    ) -> AsyncIterator[trainer_pb2.TrainProgress]:
        _LOG.info(
            "[TRAINER_ROUTER] StartJob – model=%s  job_id=%s  kn=%d",
            request.model_name,
            request.job_id or "auto",
            len(request.knowledge),
        )

        job_id = request.job_id or f"{request.model_name}:{int(time.time())}"
        try:
            register_job(job_id, json.dumps(MessageToDict(request)))
        except Exception as exc:
            _LOG.warning("register_job failed (continuing): %s", exc)

        attempts = 0
        produced_any = False
        last_err: Optional[grpc.RpcError] = None

        for wid, gpu_id, _score in _best_slots(request):
            if attempts >= _MAX_FAILOVER_ATTEMPTS:
                break
            attempts += 1

            if not await _lease_slot(wid, gpu_id):
                continue

            try:
                endpoint = _resolve_trainer_endpoint(wid)
                chan     = _chan(endpoint)
                stub     = trainer_pb2_grpc.TrainerStub(chan)

                fwd = trainer_pb2.TrainRequest()
                fwd.CopyFrom(request)
                fwd.gpu_id = gpu_id
                fwd.job_id = job_id

                _LOG.info("[TRAINER_ROUTER] → %s gpu=%s (attempt %d)", endpoint, gpu_id, attempts)

                async for prog in stub.StartJob(fwd):
                    produced_any = True

                    # 1) yield to the client IMMEDIATELY for live logs
                    yield prog

                    # 2) optional tee to router logs (accept either log_line or message)
                    if _TRAINER_TEE:
                        line = getattr(prog, "log_line", "") or getattr(prog, "message", "")
                        if line:
                            _LOG.info("[trainer %s] %s", job_id, line)

                    # 3) keep DB updates, but don’t block the stream
                    asyncio.create_task(
                        asyncio.to_thread(
                            update_job_status,
                            job_id,
                            status=trainer_pb2.TrainProgress.Stage.Name(prog.stage),
                            progress={
                                "step": prog.step,
                                "total_steps": prog.total_steps,
                                "loss": prog.loss,
                                "metric": prog.metric,
                            },
                            message=prog.message,
                        )
                    )

                return  # finished successfully

            except grpc.RpcError as exc:
                last_err = exc
                _LOG.warning("Trainer %s (gpu %s) failed: %s", wid, gpu_id, exc)
                if produced_any:
                    # If any progress already flowed, we can’t safely resume elsewhere
                    update_job_status(job_id, status="FAILED", message=exc.details() or "upstream error")
                    await context.abort(exc.code(), exc.details() or "upstream error")
                # else: try next candidate
            finally:
                await _release_slot(wid, gpu_id)

        msg = last_err.details() if isinstance(last_err, grpc.RpcError) else "No training GPU available"
        code = last_err.code() if isinstance(last_err, grpc.RpcError) else StatusCode.RESOURCE_EXHAUSTED
        update_job_status(job_id, status="FAILED", message=msg or "trainer routing failed")
        await context.abort(code, msg or "trainer routing failed")

    async def CancelJob(  # type: ignore[override]
        self,
        request: trainer_pb2.TrainRequest,
        context: aio.ServicerContext,
    ) -> empty_pb2.Empty:
        """Broadcast job-cancellation to every trainer daemon we know about."""
        async def _broadcast() -> None:
            tasks = []
            for wid in REGISTRY.training_workers():
                chan = _chan(_resolve_trainer_endpoint(wid))
                stub = trainer_pb2_grpc.TrainerStub(chan)
                tasks.append(stub.CancelJob(request))
            await asyncio.gather(*tasks, return_exceptions=True)

        await _broadcast()
        return empty_pb2.Empty()
