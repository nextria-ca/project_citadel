# citadel_master/stats.py
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from statistics import mean
from typing import Dict, List

STATS_DUMP_FILE = os.path.abspath("logs/stats.json")
EVENT_LOG_FILE  = os.path.abspath("logs/master.jsonl")      # append-only audit stream


# ────────────────────────────────────────────────────
#  Data structure representing one inference request
# ────────────────────────────────────────────────────
class RequestStat:
    def __init__(
        self,
        model: str,
        worker_id: str,
        queued_at: float,
        gpu_id: str | None = None,
        gpu_stats: dict | None = None,
    ):
        self.request_id = str(uuid.uuid4())             # unique RID
        self.model = model
        self.worker_id = worker_id
        self.queued_at = queued_at
        self.started_at:   float | None = None
        self.completed_at: float | None = None
        self.byte_size: int | None = None
        self.gpu_id = gpu_id
        self.gpu_stats: dict = gpu_stats or {}

    # handy dict view for logging/dumps
    def as_dict(self) -> dict:
        return {
            "rid": self.request_id,
            "model": self.model,
            "worker_id": self.worker_id,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "byte_size": self.byte_size,
            "gpu_id": self.gpu_id,
            "gpu_stats": self.gpu_stats,
        }


# ────────────────────────────────────────────────────
#  Aggregator - single instance lives in master
# ────────────────────────────────────────────────────
class StatsAggregator:
    """
    Aggregates:
      1. Historical request stats (queued / started / completed).
      2. Live replica metrics from worker_registry.
      3. Raw heartbeat + request events streamed to logs/master.jsonl.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()
        self._start_dump_thread()
        os.makedirs(os.path.dirname(EVENT_LOG_FILE), exist_ok=True)

    # ─────────────── low-level helpers ─────────────────
    @staticmethod
    def _write_jsonl(obj: dict) -> None:
        """Fire-and-forget append to EVENT_LOG_FILE (never breaks hot-path)."""
        try:
            with open(EVENT_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _percentile(vals: List[float], p: float) -> float | None:
        """Return the *p*th percentile from *vals* (0 < p ≤ 100)."""
        if not vals:
            return None
        vals = sorted(vals)
        k = (len(vals) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        d0 = vals[f] * (c - k)
        d1 = vals[c] * (k - f)
        return d0 + d1

    # ─────────────── public API ───────────────────────
    def reset(self) -> None:
        with self.lock:
            self.by_worker: Dict[str, Dict[str, List[RequestStat]]] = defaultdict(
                lambda: defaultdict(list)
            )
            self.global_stats: Dict[str, List[RequestStat]] = defaultdict(list)
            self.current_queues: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )

    # ---------------- event ingestion -----------------
    def record_heartbeat(self, worker_id: str, snapshots: List[dict]) -> None:
        self._write_jsonl(
            {"ev": "heartbeat", "ts": time.time(), "worker_id": worker_id, "stats": snapshots}
        )

    def record_queued(
        self,
        model: str,
        worker_id: str,
        byte_size: int,
        level: str | None = None,
        gpu_id: str | None = None,
        gpu_stats: dict | None = None,
    ) -> RequestStat | None:
        if level and str(level).upper() == "SYSTEM_SLOW":
            return None

        now  = time.time()
        stat = RequestStat(model, worker_id, now, gpu_id=gpu_id, gpu_stats=gpu_stats)
        stat.byte_size = byte_size

        with self.lock:
            self.by_worker[worker_id][model].append(stat)
            self.global_stats[model].append(stat)
            self.current_queues[worker_id][model] += 1

        self._write_jsonl({"ev": "queued", "ts": stat.queued_at, **stat.as_dict()})
        return stat

    def record_started(self, stat: RequestStat) -> None:
        stat.started_at = time.time()
        self._write_jsonl({"ev": "started", "ts": stat.started_at, **stat.as_dict()})

    def record_completed(
        self, stat: RequestStat | None, gpu_stats: dict | None = None
    ) -> None:
        if stat is None:
            return

        stat.completed_at = time.time()
        if gpu_stats is not None:
            stat.gpu_stats = gpu_stats
            # propagate gpu_id for easier querying
            if gpu_stats.get("gpu_id") is not None:
                stat.gpu_id = gpu_stats["gpu_id"]

        with self.lock:
            self.current_queues[stat.worker_id][stat.model] -= 1

        self._write_jsonl({"ev": "completed", "ts": stat.completed_at, **stat.as_dict()})

    # ---------------- summarisation helpers -----------------
    def _summarize(self, stats: List[RequestStat]) -> dict:
        """
        Rich summary for a collection of RequestStat objects.
        Includes every metric we can cheaply derive.
        """
        n_total = len(stats)
        completed = [s for s in stats if s.completed_at]
        n_completed = len(completed)
        n_in_flight = n_total - n_completed
        if n_total == 0:
            # empty bucket
            return {
                "count": 0,
                "completed": 0,
                "in_flight": 0,
                "time_active": 0,
                "unique_workers": 0,
                "avg_latency": None,
                "p95_latency": None,
                "min_latency": None,
                "max_latency": None,
                "avg_queue_time": None,
                "p95_queue_time": None,
                "total_bytes": 0,
                "avg_bytes": None,
                "throughput_rps": 0,
            }

        # ── basic sets ──────────────────────────────────
        latencies = [
            s.completed_at - s.started_at
            for s in completed
            if s.started_at and s.completed_at
        ]
        queues = [
            s.started_at - s.queued_at
            for s in completed
            if s.started_at and s.queued_at
        ]
        bytes_sum = sum(s.byte_size or 0 for s in completed)

        # ── time window ─────────────────────────────────
        first_seen = min(s.queued_at for s in stats)
        last_seen  = (
            max(s.completed_at for s in completed)
            if completed
            else max((s.started_at or s.queued_at) for s in stats)
        )
        time_active = last_seen - first_seen

        # ── throughput (completed only) ────────────────
        throughput = (n_completed / time_active) if (n_completed and time_active > 0) else 0

        # ── latency / queue metrics ─────────────────────
        avg_latency = mean(latencies) if latencies else None
        p95_latency = self._percentile(latencies, 95) if latencies else None
        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None

        avg_queue = mean(queues) if queues else None
        p95_queue = self._percentile(queues, 95) if queues else None

        # ── bytes metrics ───────────────────────────────
        avg_bytes = (bytes_sum / n_completed) if n_completed else None

        # ── worker spread ───────────────────────────────
        unique_workers = len({s.worker_id for s in stats})

        return {
            "count": n_total,
            "completed": n_completed,
            "in_flight": n_in_flight,
            "time_active": time_active,
            "unique_workers": unique_workers,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "avg_queue_time": avg_queue,
            "p95_queue_time": p95_queue,
            "total_bytes": bytes_sum,
            "avg_bytes": avg_bytes,
            "throughput_rps": throughput,
        }

    def _summarize_by_gpu(self, stats: List[RequestStat]) -> Dict[str, dict]:
        """Bucket *stats* by GPU-ID and summarise each bucket."""
        buckets: Dict[str, List[RequestStat]] = defaultdict(list)
        for s in stats:
            gid = str(s.gpu_id) if s.gpu_id not in (None, "", -1) else "unknown"
            buckets[gid].append(s)
        return {gid: self._summarize(lst) for gid, lst in buckets.items()}

    # ---------------- high-level report -----------------
    def get_stats(self) -> dict:
        """
        Returns a nested dict with two top-level keys:

        • history … windowed + cumulative metrics derived from RequestStat
        • live     … most recent heartbeat/replica data from worker_registry
        """
        with self.lock:
            # ---------- historical -----------------------
            hist_by_worker_gpu = {
                wid: self._summarize_by_gpu(
                    [s for model_stats in models.values() for s in model_stats]
                )
                for wid, models in self.by_worker.items()
            }

            hist_by_worker_model = {
                wid: {m: self._summarize(lst) for m, lst in models.items()}
                for wid, models in self.by_worker.items()
            }

            hist_by_model = {
                m: self._summarize(lst) for m, lst in self.global_stats.items()
            }

            hist_global = self._summarize(
                [s for lst in self.global_stats.values() for s in lst]
            )

            history = {
                "global": hist_global,
                "by_model": hist_by_model,
                "by_worker": hist_by_worker_model,
                "by_worker_gpu": hist_by_worker_gpu,
            }

        # ---------- live replica telemetry ---------------
        live = self._collect_live_metrics()
        return {"live": live, "history": history}

    # ---------------- live metrics ----------------------
    @staticmethod
    def _collect_live_metrics() -> dict:
        try:
            from master_core.worker_registry import REGISTRY
        except ImportError:
            return {}

        live_summary = {"global": {}, "by_worker": {}}
        g_totals = {
            "replicas": 0,
            "queue_depth": 0,
            "in_flight": 0,
            "latency_samples": [],
        }

        for wid, info in REGISTRY._db.items():
            reps = info.replica_stats
            if not reps:
                continue

            w_totals = {
                "replicas": len(reps),
                "queue_depth": 0,
                "in_flight": 0,
                "avg_latency_ms": None,
            }

            lat_samples = []
            for r in reps:
                w_totals["queue_depth"] += r.get("queue_depth", 0)
                w_totals["in_flight"] += r.get("in_flight", 0)
                if (lat := r.get("ewma_latency_ms")):
                    lat_samples.append(lat)

            if lat_samples:
                w_totals["avg_latency_ms"] = mean(lat_samples)
                g_totals["latency_samples"].extend(lat_samples)

            live_summary["by_worker"][wid] = w_totals
            g_totals["replicas"] += w_totals["replicas"]
            g_totals["queue_depth"] += w_totals["queue_depth"]
            g_totals["in_flight"] += w_totals["in_flight"]

        live_summary["global"] = {
            "replicas": g_totals["replicas"],
            "queue_depth": g_totals["queue_depth"],
            "in_flight": g_totals["in_flight"],
            "avg_latency_ms": (
                mean(g_totals["latency_samples"])
                if g_totals["latency_samples"]
                else None
            ),
        }
        return live_summary

    # ---------------- periodic JSON dump ----------------
    def _start_dump_thread(self) -> None:
        threading.Thread(target=self._periodic_dump, daemon=True).start()

    def _periodic_dump(self) -> None:
        while True:
            try:
                self.dump_to_file()
            except Exception:
                pass
            time.sleep(5)

    def dump_to_file(self) -> None:
        stats = self.get_stats()
        os.makedirs(os.path.dirname(STATS_DUMP_FILE), exist_ok=True)
        with open(STATS_DUMP_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


# singleton used throughout the master process
AGGREGATOR = StatsAggregator()
