"""
Citadel Master - unified gRPC front-door
=======================================

Single **aio.server** instance that multiplexes:

│ 9000 │  Inference  +  Trainer-router   (client-facing)
│ 9100 │  Worker-registry / admin        (internal)
"""
from __future__ import annotations
MAX_MSG = 64 * 1024 * 1024  # 64 MiB
GRPC_OPTS = [
    ("grpc.max_send_message_length",    MAX_MSG),
    ("grpc.max_receive_message_length", MAX_MSG),
]

import asyncio
import socket
from pathlib import Path

from grpc import aio

from citadel_shared.env_loader import load_role_env

load_role_env("master")

from citadel_shared.logging_setup import setup_logging

from inference_servicer_engine import register as register_inference

# ── worker-registry & autoscaler ───────────────────────────────────
from citadel_master.grpc_services.registry_servicer import (
    WorkerAdminServicer,
    WorkerControlServicer,
    MasterStatsServicer,
)
from citadel_master.grpc_services.metrics_servicer import MetricsServicer
from master_core.autoscaler import AutoScaler
from proto.python import registry_pb2_grpc
from citadel_master.grpc_services import trainer_servicer as trainer_router
from proto.python import trainer_pb2_grpc


setup_logging()

INFERENCE_PORT = 9000           # inference + trainer
REGISTRY_PORT  = 9100           # worker registry / control


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------
def _primary_ip() -> str:
    """
    Return the primary non-loopback IPv4 address of this host.

    We open a dummy UDP socket to a well-known external address; no
    traffic is ever sent, but the OS will populate the socket with the
    outbound interface’s IP, which is what we want to display.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _write_worker_env(master_ip: str) -> None:
    """
    Ensure citadel_shared/worker.env contains **exactly one** line
    `MASTER_HOST=<detected-ip>` - either replacing an existing key or
    appending it if the file is missing / malformed.
    """
    shared_root = Path(__file__).resolve().parent.parent / "citadel_shared"
    env_file = shared_root / "worker.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    found = False

    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            if raw.strip().startswith("MASTER_HOST="):
                lines.append(f"MASTER_HOST={master_ip}")
                found = True
            else:
                lines.append(raw.rstrip())
    if not found:                       # add the key if it wasn’t present
        lines.append(f"MASTER_HOST={master_ip}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------
#  Main async entry-point
# -------------------------------------------------------------------
async def main() -> None:
    """Boot the Citadel master gRPC server with all servicers attached."""
    server = aio.server(options=GRPC_OPTS)

    # ── client-facing servicers  (port 9000) ─────────────────────────
    register_inference(server)
    trainer_pb2_grpc.add_TrainerServicer_to_server(
        trainer_router.TrainerRouterServicer(), server
    )

    # ── internal control plane  (port 9100) ──────────────────────────
    registry_pb2_grpc.add_WorkerAdminServicer_to_server(WorkerAdminServicer(), server)
    registry_pb2_grpc.add_WorkerControlServicer_to_server(WorkerControlServicer(), server)
    registry_pb2_grpc.add_MasterStatsServicer_to_server(MasterStatsServicer(), server)

    # ── metrics service ─────────────────────────────
    registry_pb2_grpc.add_GpuMetricsServicer_to_server(MetricsServicer(), server)

    # ── bind sockets ────────────────────────────────────────────────
    server.add_insecure_port(f"0.0.0.0:{INFERENCE_PORT}")
    server.add_insecure_port(f"0.0.0.0:{REGISTRY_PORT}")

    await server.start()

    ip = _primary_ip()
    _write_worker_env(ip)   # ← NEW: persist for workers to read later

    print(f"[MASTER] Detected host IP: {ip}", flush=True)
    print(f"[MASTER] Inference + Trainer router listening on {ip}:{INFERENCE_PORT}", flush=True)
    print(f"[MASTER] Worker-registry service listening on     {ip}:{REGISTRY_PORT}", flush=True)  # :contentReference[oaicite:0]{index=0}

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("[MASTER] Shutdown requested - stopping …")
        await server.stop(grace=None)


# -------------------------------------------------------------------
#  Process entry-point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # The autoscaler runs in a background thread and watches worker stats.
    AutoScaler().start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
