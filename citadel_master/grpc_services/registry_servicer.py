# citadel_master/grpc_services/registry_servicer.py
"""
gRPC servicers that the worker containers call to register themselves with the
master and to send periodic heartbeats. The heartbeat payload is a list of
HeartbeatSnapshot (queue_depth, in_flight, ewma_latency_ms).
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
from typing import Dict, List

import grpc
from google.protobuf import empty_pb2

from master_core.worker_registry import REGISTRY
from citadel_master.stats import AGGREGATOR  # <-- new

from proto.python import registry_pb2, registry_pb2_grpc

_LOG = logging.getLogger(__name__)


def _canonical_host(raw: str) -> str:
    raw = raw.strip()
    try:
        return ipaddress.ip_address(raw).compressed
    except ValueError:
        pass
    try:
        return socket.gethostbyname(raw)
    except socket.gaierror:
        return raw


class WorkerAdminServicer(registry_pb2_grpc.WorkerAdminServicer):
    def Register(
        self,
        request: registry_pb2.RegisterRequest,
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        peer_host = context.peer().split(":")[1]
        advertise_host = request.advertise_host or peer_host
        host = (
            advertise_host
            if advertise_host.lower() == "host.docker.internal"
            else _canonical_host(advertise_host)
        )
        endpoint = f"{host}:{request.serve_port}"
        REGISTRY.register(
            request.worker_id,
            list(request.models),
            endpoint,
            request.can_infer,
            request.can_train,
        )
        _LOG.info(
            "Registered worker %s @ %s (models=%s, infer=%s, train=%s)",
            request.worker_id,
            endpoint,
            list(request.models),
            request.can_infer,
            request.can_train,
        )
        return empty_pb2.Empty()

    def Heartbeat(
        self,
        request: registry_pb2.HeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        stats: List[Dict[str, int | float]] = [
            {
                "queue_depth":        s.queue_depth,
                "in_flight":          s.in_flight,
                "ewma_latency_ms":    s.ewma_latency_ms,
                "queued_work_units":  s.queued_work_units,
                "processing_spu":     s.processing_spu,
                "gpu_id":             s.gpu_id,
                "gpu_total_mb":       s.gpu_total_mb,
                "gpu_free_mb":        s.gpu_free_mb,
            }
            for s in request.stats
        ]

        REGISTRY.heartbeat(request.worker_id, stats)
        AGGREGATOR.record_heartbeat(request.worker_id, stats)  # <-- new

        return empty_pb2.Empty()


class WorkerControlServicer(registry_pb2_grpc.WorkerControlServicer):
    def Shutdown(self, request, context):  # noqa: N802
        return empty_pb2.Empty()

    def ShutdownModel(self, request, context):  # noqa: N802
        _LOG.info("ShutdownModel: worker=%s, model=%s", request.worker_id, request.model_name)
        return empty_pb2.Empty()

    def ShutdownGPU(self, request, context):  # noqa: N802
        _LOG.info("ShutdownGPU: worker=%s, gpu_id=%s", request.worker_id, request.gpu_id)
        return empty_pb2.Empty()


class MasterStatsServicer(registry_pb2_grpc.MasterStatsServicer):
    def GetStats(self, request, context):  # noqa: N802
        from master_core import stats as master_stats

        blob = master_stats.AGGREGATOR.get_stats()
        return registry_pb2.StatsResponse(json=json.dumps(blob, indent=2))

    def ListModels(self, request, context):  # noqa: N802
        models_info = []
        for worker_id, info in REGISTRY._db.items():
            for model in info.models:
                if isinstance(model, dict):
                    model_name = model.get("model_name") or model.get("name", "")
                    gpu_id = model.get("gpu_id", "")
                else:
                    model_name = str(model)
                    gpu_id = ""
                models_info.append(
                    registry_pb2.ModelInfo(
                        model_name=model_name,
                        worker_id=worker_id,
                        endpoint=info.endpoint,
                        gpu_id=gpu_id,
                    )
                )
        return registry_pb2.ListModelsResponse(models=models_info)
