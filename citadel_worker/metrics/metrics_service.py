import asyncio
import json
from typing import List
from citadel_worker.worker_core import model_service
from citadel_shared.logging_setup import get_logger
from citadel_worker.metrics.gpu_metrics import GPUCollector
from proto.python import registry_pb2_grpc, registry_pb2


class MetricsClient:
    def __init__(
        self,
        client: registry_pb2_grpc.GpuMetricsStub,
        worker_id,
        engine: model_service.ModelService,
    ):
        self.worker_id = worker_id
        self.client = client
        self.temp = []
        self.logger = get_logger(f"metrics_service.{worker_id}")
        self.collector = GPUCollector(worker_id)
        self.host = "localhost"
        self.frequency = 5  # seconds
        self.batch_size = 5
        self.engine = engine

    def worker_metrics(self) -> dict:
        qd = 0
        infl = 0
        backlog_units = 0
        eta_s = 0.0
        for states in self.engine._states.values():
            for st in states:
                qd += st.queue_depth
                infl += st.in_flight
                backlog_units += st.queued_work_units + st.running_work_units
                eta_s += (st.queued_work_units + st.running_work_units) * st.processing_spu
        return {"queue_depth": qd, "in_flight": infl, "backlog_units": backlog_units, "eta_s": eta_s}


    def collect_data(self) -> List[registry_pb2.GpuMetricsData]:
        try:
            # t0 = time.perf_counter()

            metric_lines = self.collector.query_nvidia_smi(self.host)
            if metric_lines:

                # t1 = time.perf_counter()
                # self.logger.debug(f"nvidia-smi query took {t1 - t0:.3f} sec")
                data = self.collector.parse_metrics(metric_lines, self.host, self.worker_id)

                # t2 = time.perf_counter()
                # self.logger.debug(f"nvidia-smi parse took {t2 - t1:.3f} sec, got {len(data)} items")

                return data

            # t1 = time.perf_counter()
            # self.logger.debug(f"nvidia-smi query took {t1 - t0:.3f} sec (no data)")
            return []
        except Exception as e:
            self.logger.exception(f"Error collecting GPU metrics {e}")
            return []

    async def send_data(self, data):

        snapshot = json.dumps(self.worker_metrics(), ensure_ascii=False) 
        req = registry_pb2.GpuMetrcsRequest(
            items=data,
            worker_load_snapshot=snapshot,
            worker_id=self.worker_id
        )
        resp = await self.client.AddData(req, timeout=5.0)
        return resp

    async def run_loop(self):
        try:

            while True:
                try:
                    data = self.collect_data()
                    if data:
                        self.temp.extend(data)

                    if len(self.temp) >= self.batch_size:
                        try:
                            await self.send_data(self.temp)
                            self.temp = []
                        except Exception:
                            self.logger.exception("Send failed; keeping batch for retry")

                except Exception:
                    self.logger.exception("Error in metrics loop")

                await asyncio.sleep(self.frequency)
        except asyncio.CancelledError:
            self.logger.info("metrics loop cancelled; flushing %d pending items", len(self.temp))
            if self.temp:
                try:
                    await self.send_data(self.temp)
                    self.logger.info("Final flush sent")
                except Exception:
                    self.logger.exception("Final flush failed")
            raise
