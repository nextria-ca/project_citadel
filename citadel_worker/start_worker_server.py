from __future__ import annotations

from citadel_worker.metrics.metrics_service import MetricsClient
from citadel_shared.env_loader import load_role_env

load_role_env("worker")

MAX_MSG = 64 * 1024 * 1024  # 64 MiB
GRPC_OPTS = [
    ("grpc.max_send_message_length",    MAX_MSG),
    ("grpc.max_receive_message_length", MAX_MSG),
]


from citadel_shared.logging_setup import setup_logging

setup_logging()

import argparse
import asyncio
import contextlib
import logging
import os
import platform
import socket
import time
from pathlib import Path
from typing import AsyncIterator, List

import grpc.aio as aio

from proto.python import inference_pb2, registry_pb2, registry_pb2_grpc
from citadel_worker.worker_core import model_service

_LOG = logging.getLogger(__name__)

ENGINE = model_service.ModelService()
_SENTINEL = object()
_IMMEDIATE_HB = os.getenv("CITADEL_IMMEDIATE_HEARTBEAT", "1") not in ("0", "false")


def _detect_host(master_host: str, master_port: int) -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((master_host, master_port))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _in_docker() -> bool:
    return os.path.exists("/.dockerenv") or os.getenv("DOCKER") == "true"


def _cleanup_replica_logs() -> None:
    logging.shutdown()
    for log_file in Path(".").rglob("replica_*.log"):
        try:
            log_file.unlink()
            _LOG.debug("Removed %s", log_file)
        except Exception as exc:
            _LOG.warning("Could not delete %s: %s", log_file, exc)


async def _heartbeat_loop(
    stub: registry_pb2_grpc.WorkerAdminStub,
    wid: str,
    engine: model_service.ModelService,
    interval: int = 30,
) -> None:
    while True:
        try:
            stats_proto = [
                registry_pb2.HeartbeatSnapshot(
                    queue_depth=s["queue_depth"],
                    in_flight=s["in_flight"],
                    ewma_latency_ms=s["ewma_latency_ms"],
                    queued_work_units=s["queued_work_units"],
                    processing_spu=s["processing_spu"],
                    gpu_id=s["gpu_id"],
                    gpu_total_mb=s["gpu_total_mb"],
                    gpu_free_mb=s["gpu_free_mb"],
                )
                for s in engine.get_replica_stats()
            ]
            await stub.Heartbeat(
                registry_pb2.HeartbeatRequest(worker_id=wid, stats=stats_proto)
            )
            _LOG.debug("Heartbeat sent (%d replicas)", len(stats_proto))
        except Exception as exc:
            _LOG.warning("Heartbeat failed: %s", exc)
        await asyncio.sleep(interval)


class AsyncWorkerForwardServicer(registry_pb2_grpc.WorkerForwardServicer):
    """
    Streams requests straight to the **async-native** ModelService and, if
    configured, fires an immediate heartbeat so the master registry sees the
    updated queue/in-flight numbers without waiting for the 5-second beat.
    """

    def __init__(
        self,
        engine: model_service.ModelService,
        *,
        admin_stub: registry_pb2_grpc.WorkerAdminStub,
        worker_id: str,
        monitor: MetricsClient
    ) -> None:
        self._engine = engine
        self._admin  = admin_stub
        self._wid    = worker_id
        self._monitor = monitor

    # ────────────────────────── heartbeat helper ──────────────────────────
    async def _send_heartbeat_now(self) -> None:
        stats_proto = [
            registry_pb2.HeartbeatSnapshot(
                queue_depth       = s["queue_depth"],
                in_flight         = s["in_flight"],
                ewma_latency_ms   = s["ewma_latency_ms"],
                queued_work_units = s["queued_work_units"],
                processing_spu    = s["processing_spu"],
                gpu_id            = s["gpu_id"],
                gpu_total_mb      = s["gpu_total_mb"],
                gpu_free_mb       = s["gpu_free_mb"],
            )
            for s in self._engine.get_replica_stats()
        ]
        try:
            await self._admin.Heartbeat(
                registry_pb2.HeartbeatRequest(worker_id=self._wid, stats=stats_proto)
            )
            _LOG.debug("Immediate heartbeat sent (%d replicas)", len(stats_proto))
        except Exception as exc:
            _LOG.debug("Immediate heartbeat failed: %s", exc)

    # ──────────────────────────── main RPC ────────────────────────────────
    async def Forward(
        self,
        request_iterator: AsyncIterator[inference_pb2.InferenceRequest],
        context: aio.ServicerContext,
    ) -> AsyncIterator[inference_pb2.InferenceResponse]:
        """
        Pass the caller's async iterator straight to the engine’s
        async-stream handler – zero thread hops.

        We only need to peek at the **first** frame to decide whether to send
        an immediate heartbeat, then re-yield it so the engine sees every
        request exactly once.
        """
        try:
            first_req = await anext(request_iterator)
        except StopAsyncIteration:
            return  # nothing to do

        if _IMMEDIATE_HB:
            await self._send_heartbeat_now()

        async def _chain() -> AsyncIterator[inference_pb2.InferenceRequest]:
            yield first_req
            async for req in request_iterator:
                yield req

        async for rsp in self._engine.handle_stream(_chain()):
            self._monitor.collect_data()
            yield rsp


async def _amain() -> None:
    p = argparse.ArgumentParser("Citadel Worker")
    p.add_argument("--master", default=os.getenv("MASTER_HOST"))
    p.add_argument("--registry-port", type=int, default=int(os.getenv("REGISTRY_PORT")))
    p.add_argument("--serve-port", type=int, default=int(os.getenv("SERVE_PORT")))
    p.add_argument("--worker-id", default="worker-wsl")
    p.add_argument("--advertise-host", default=os.getenv("ADVERTISE_HOST"))
    p.add_argument("--role", choices=["infer", "train", "mixed"], default="infer")
    args = p.parse_args()

    role_can_infer = args.role in ("infer", "mixed")
    role_can_train = args.role in ("train", "mixed")

    _LOG.info("Booting %s", args.worker_id)

    if args.advertise_host:
        advertise_host = args.advertise_host
    elif _in_docker() and platform.system() == "Windows":
        advertise_host = "host.docker.internal"
    else:
        advertise_host = _detect_host(args.master, args.registry_port)
    _LOG.info("Advertised host: %s", advertise_host)

    reg_chan = aio.insecure_channel(f"{args.master}:{args.registry_port}")
    admin = registry_pb2_grpc.WorkerAdminStub(reg_chan)

    await admin.Register(
        registry_pb2.RegisterRequest(
            worker_id=args.worker_id,
            models=[
                registry_pb2.ModelInfo(
                    model_name=m["name"],
                    gpu_id=str(m["gpu_id"]),
                    worker_id=args.worker_id,
                    endpoint=advertise_host,
                    batch=registry_pb2.BatchingPolicy(),
                )
                for m in model_service.available_models()
            ],
            serve_port=args.serve_port,
            advertise_host=advertise_host,
            can_infer=role_can_infer,
            can_train=role_can_train,
        )
    )
    heartbeat_interval = int(os.getenv("CITADEL_HEARTBEAT_INTERVAL_S", "5"))
    _LOG.info("Heartbeat interval: %d seconds", heartbeat_interval)
    hb_task = asyncio.create_task(
        _heartbeat_loop(admin, args.worker_id, ENGINE, heartbeat_interval)
    )
    cache_interval = int(os.getenv("CITADEL_CUDA_CACHE_INTERVAL_S", "3600"))
    cache_task = None
    if cache_interval > 0:
        cache_task = asyncio.create_task(ENGINE.cache_clear_loop(cache_interval))

    metrics_reg_chan = aio.insecure_channel(f"{args.master}:{args.registry_port}")
    metrics_grpc_client = registry_pb2_grpc.GpuMetricsStub(metrics_reg_chan)

    metrics_client = MetricsClient(metrics_grpc_client, args.worker_id, ENGINE)
    metrics_task = asyncio.create_task(metrics_client.run_loop())

    server = aio.server(options=GRPC_OPTS)
    registry_pb2_grpc.add_WorkerForwardServicer_to_server(
        AsyncWorkerForwardServicer(
            ENGINE,
            admin_stub=admin,
            worker_id=args.worker_id,
            monitor=metrics_client
        ),
        server,
    )
    server.add_insecure_port(f"0.0.0.0:{args.serve_port}")
    await server.start()
    _LOG.info("Listening on :%d", args.serve_port)

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        _LOG.info("Shutdown requested …")
    finally:
        hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb_task
            await metrics_task
        if cache_task:
            cache_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cache_task
        await server.stop(0)
        _cleanup_replica_logs()
        _LOG.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(_amain())
