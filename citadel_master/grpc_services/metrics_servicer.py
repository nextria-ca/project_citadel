
import asyncio
import json
import logging
import threading
from google.protobuf import empty_pb2
from datetime import datetime, timezone

from citadel_master.metrics.core.metrics_general import (
    HOURLY_FILE,
    _load_json_or,
    _save_json
)
from citadel_master.metrics.core.metrics_hour import update_hourly
from proto.python import registry_pb2_grpc
from google.protobuf.json_format import MessageToDict


class MetricsServicer(registry_pb2_grpc.GpuMetricsServicer):
    def __init__(self, *, queue_maxsize: int = 50_000, drop_policy: str = "drop_oldest"):
        self.logger = logging.getLogger(__name__)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._worker: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._started = False
        # drop_policy: "drop_oldest" or "drop_newest"
        self._drop_policy = drop_policy

        # Optional micro-batch knobs (set to 1/0 to effectively disable batching)
        self._batch_max_items = 0
        self._batch_timeout_s = 0.0
        self._worker_qd: dict[str, int] = {}
        self._qd_lock = threading.Lock()

    # thread-safe write
    def _set_worker_qd(self, worker_id: str, qd: int) -> None:
        if qd < 0:
            qd = 0
        with self._qd_lock:
            self._worker_qd[worker_id] = int(qd)

    # thread-safe snapshot read (pl. hourly pipeline használhatja)
    def get_worker_qd_snapshot(self) -> dict[str, int]:
        with self._qd_lock:
            return dict(self._worker_qd)

    async def start(self):
        """Start single background consumer (idempotent)."""
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._worker = asyncio.create_task(self._consumer())
        self.logger.info("MetricsServicer started (1 worker, maxsize=%d, policy=%s)",
                         self.queue.maxsize, self._drop_policy)

    async def stop(self):
        """Graceful shutdown."""
        if not self._started:
            return
        self._stop.set()
        # Unblock consumer
        try:
            self.queue.put_nowait(None)  # sentinel
        except asyncio.QueueFull:
            # Drop one and insert sentinel
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                pass
            finally:
                self.queue.put_nowait(None)
        if self._worker:
            await self._worker
            self._worker = None
        self._started = False
        self.logger.info("MetricsServicer stopped")

    async def _consumer(self):
        """Single consumer: processes queued items off-thread; optional micro-batch."""
        self.logger.debug("consumer started")
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            if item is None or self._stop.is_set():
                self.queue.task_done()
                break

            batch = [item]
            # Optional micro-batch drain (off by default)
            if self._batch_max_items > 1 or self._batch_timeout_s > 0:
                t0 = loop.time()
                while len(batch) < self._batch_max_items and (loop.time() - t0) < self._batch_timeout_s:
                    try:
                        nxt = self.queue.get_nowait()
                        if nxt is None:
                            self.queue.task_done()
                            break
                        batch.append(nxt)
                        self.queue.task_done()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0)
                        break

            try:
                for it in batch:
                    d = MessageToDict(it, preserving_proto_field_name=True, use_integers_for_enums=True)
                    await asyncio.to_thread(update_hourly, d, self.get_worker_qd_snapshot())
            except Exception:
                self.logger.exception("consumer: update_hourly failed")
            finally:
                self.queue.task_done()

        self.logger.debug("consumer stopped")

    def _enqueue_nowait(self, item) -> None:
        """Non-blocking enqueue with drop policy when full."""
        try:
            self.queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            if self._drop_policy == "drop_oldest":
                try:
                    _ = self.queue.get_nowait()  # drop oldest
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    # Still full → drop newest (this one)
                    self.logger.warning("ingress queue full; dropped newest after evicting oldest")
            else:  # drop_newest
                self.logger.warning("ingress queue full; dropped newest")

    async def AddData(self, request, context):
        """Non-blocking ingress: enqueue and return immediately."""
        wid = request.worker_id if hasattr(request, "worker_id") else "unknown"
        if not self._started:
            await self.start()
        count = 0
        try:
            for it in request.items:
                self._enqueue_nowait(it)
                count += 1
        except Exception:
            self.logger.exception("AddData: enqueue failed")
        try:
            raw = getattr(request, "worker_load_snapshot", "") or ""
            if raw:
                snap = json.loads(raw)
                qd = int(snap.get("queue_depth", 0) or 0)
                if qd < 0:
                    qd = 0
                self._set_worker_qd(wid, qd)
        except Exception:
            self.logger.exception("AddData: worker_load_snapshot parse/update failed")
        return empty_pb2.Empty()


class MetricsJobMonitorService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def WorkerStartedJob(self, worker_id: str, model: str) -> None:
        """Increment in_flight and set current_model for the worker (sync)."""
        try:
            with self._lock:
                doc = _load_json_or(HOURLY_FILE, None) or {
                    "schema_version": 2,
                    "day": "",
                    "hours": [],
                    "latest": {"timestamp": None, "by_gpu": {}, "by_worker": {}},
                }
                latest = doc.setdefault("latest", {"timestamp": None, "by_gpu": {}, "by_worker": {}})
                wb = latest.setdefault("by_worker", {})
                w = wb.setdefault(worker_id, {
                    "in_flight": 0,
                    "tasks_served": 0,
                    "queue_depth": 0,
                    "current_model": ""
                })

                w["in_flight"] = int(w.get("in_flight", 0)) + 1
                w["current_model"] = model or w.get("current_model", "")
                latest["timestamp"] = self._now_iso()

                _save_json(HOURLY_FILE, doc)

            self.logger.info("Worker %s started %s", worker_id, model)
        except Exception as e:
            self.logger.exception("WorkerStartedJob failed: %s", e)

    def WorkerFinishedJob(self, worker_id: str, model: str) -> None:
        """Decrement in_flight; increment tasks_served; clear model if none in flight (sync).
        PLUS: bump per-hour finishes (job_count_sum) for this worker in the current hour."""
        try:
            with self._lock:
                doc = _load_json_or(HOURLY_FILE, None) or {
                    "schema_version": 2,
                    "day": "",
                    "hours": [],
                    "latest": {"timestamp": None, "by_gpu": {}, "by_worker": {}},
                }

                latest = doc.setdefault("latest", {"timestamp": None, "by_gpu": {}, "by_worker": {}})
                wb = latest.setdefault("by_worker", {})
                w = wb.setdefault(worker_id, {
                    "in_flight": 0,
                    "tasks_served": 0,
                    "queue_depth": 0,
                    "current_model": ""
                })

                new_in_flight = max(0, int(w.get("in_flight", 0)) - 1)
                w["in_flight"] = new_in_flight
                w["tasks_served"] = int(w.get("tasks_served", 0)) + 1
                if new_in_flight == 0:
                    w["current_model"] = ""

                if latest.get("timestamp") is None:
                    latest["timestamp"] = self._now_iso()
            
                now_iso = latest["timestamp"] 
                today = now_iso.split("T", 1)[0] if now_iso else ""
                if doc.get("day") != today:
                    doc["day"] = today
                    doc["hours"] = []

                hour_now = int(now_iso[11:13])

                hours = doc.setdefault("hours", [])
                hb = None
                for h in hours:
                    if h.get("hour") == hour_now:
                        hb = h
                        break
                if hb is None:
                    hb = {"hour": hour_now, "by_gpu": {}}
                    hours.append(hb)

                by_gpu_hour = hb.setdefault("by_gpu", {})

                candidate_keys = [k for k, s in by_gpu_hour.items() if (s or {}).get("worker_id") == worker_id]
                if not candidate_keys:
                    for k, s in (latest.get("by_gpu") or {}).items():
                        if (s or {}).get("worker_id") == worker_id:
                            if k not in by_gpu_hour:
                                parts = k.split("|")
                                uuid = parts[1] if len(parts) > 1 else ""
                                try:
                                    gpu_index = int(parts[2]) if len(parts) > 2 else 0
                                except Exception:
                                    gpu_index = 0
                                by_gpu_hour[k] = {
                                    "worker_id": worker_id,
                                    "uuid": uuid,
                                    "gpu_index": gpu_index,
                                    "gpu_name": (s or {}).get("gpu_name", ""),
                                    "samples": 0,
                                    "sum_util_gpu": 0,
                                    "sum_util_mem": 0,
                                    "sum_active_jobs": 0,
                                    "max_util_gpu": 0,
                                    "max_util_mem": 0,
                                    "max_active_jobs": 0,
                                    "over_90_samples": 0,
                                    "job_count_sum": 0,
                                    "models_hosted": {},
                                }
                            candidate_keys.append(k)

                for k in candidate_keys:
                    slot = by_gpu_hour.get(k)
                    if slot is None:
                        continue
                    slot["job_count_sum"] = int(slot.get("job_count_sum", 0)) + 1

                _save_json(HOURLY_FILE, doc)

            self.logger.info("Worker %s finished %s", worker_id, model)
        except Exception as e:
            self.logger.exception("WorkerFinishedJob failed: %s", e)