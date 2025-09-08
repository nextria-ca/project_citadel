# citadel_master/master_core/autoscaler.py
"""
citadel_master.master_core.autoscaler
─────────────────────────────────────
Demand-driven autoscaler that adds or removes replicas per model
based on live queue depth and latency.

✓ Windows-only / no WSL
✓ Runs as a background thread inside the master
✓ Uses PowerShell helpers (Run-Worker.ps1) to spin up extra workers
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

from master_core.worker_registry import REGISTRY
from citadel_master.stats import AGGREGATOR as _AGG

_LOG = logging.getLogger(__name__)

# ───────────────────────── Tunables ──────────────────────────
SCALE_INTERVAL_S       = int(os.getenv("CITADEL_SCALE_INTERVAL_S", "15"))
TARGET_LATENCY_MS      = int(os.getenv("CITADEL_TARGET_LATENCY_MS", "1500"))
HYSTERESIS_QD          = int(os.getenv("CITADEL_SCALE_HYST_QDEPTH", "2"))
COOLDOWN_S             = int(os.getenv("CITADEL_SCALE_COOLDOWN_S", "60"))
MAX_REPLICAS_PER_MODEL = int(os.getenv("CITADEL_MAX_REPLICAS_PER_MODEL", "4"))
STARTUP_DEFAULT_S      = int(os.getenv("CITADEL_STARTUP_DEFAULT_S", "45"))

# New down-scale gate: we only allow scale-down when ANY model queue ≥ this.
DOWNSCALE_TRIGGER_QD   = int(os.getenv("CITADEL_DOWNSCALE_TRIGGER_QD", "100"))

RUN_WORKER_PS1 = Path(
    os.getenv("CITADEL_RUN_WORKER_PS1", r"scripts\Run-Worker.ps1")
).resolve()


class _ModelAgg:
    __slots__ = ("queue_depth", "in_flight", "lat_ms", "worker_id")

    def __init__(self, d: Dict[str, int | float], worker_id: str) -> None:
        self.queue_depth = int(d.get("queue_depth", 0))
        self.in_flight   = int(d.get("in_flight", 0))
        self.lat_ms      = float(d.get("ewma_latency_ms", 0.0))
        self.worker_id   = worker_id


class AutoScaler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="citadel-autoscaler")
        self._last_scale_ts: float = 0.0
        self._startup_hist: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=8)
        )
        self._pending_launch: dict[str, float] = {}

        self._boot_floor: Dict[str, int] = defaultdict(int)
        self._bootstrapped: bool = False

    # ───────────────────────── Main loop ─────────────────────────
    def run(self) -> None:  # noqa: D401
        _LOG.info("Autoscaler thread started — interval=%ss", SCALE_INTERVAL_S)
        while True:
            try:
                self._tick()
            except Exception:
                _LOG.exception("Autoscaler tick failed")
            time.sleep(SCALE_INTERVAL_S)

    # ───────────────────────── One tick ─────────────────────────
    def _tick(self) -> None:
        snapshot = self._collect_snapshot()
        decisions = self._decide(snapshot)
        self._apply(decisions)

    def _collect_snapshot(self) -> Dict[str, List[_ModelAgg]]:
        agg: Dict[str, List[_ModelAgg]] = defaultdict(list)
        queues = _AGG.current_queues  # priority‑filtered queues

        for wid in REGISTRY.inference_workers():
            wq = queues.get(wid, {})
            for rep in REGISTRY.replica_stats(wid):
                for model in REGISTRY._db[wid].models:
                    mstat = _ModelAgg(rep, wid)
                    mstat.queue_depth = wq.get(model, 0)
                    agg[model].append(mstat)
                    break

        for model, reps in agg.items():
            if model in self._pending_launch and len(reps) >= 2:
                self._register_startup(model, self._pending_launch.pop(model))

        return agg


    def _estimate_startup_s(self, model: str) -> float:
        hist = self._startup_hist.get(model)
        return (sum(hist) / len(hist)) if hist else STARTUP_DEFAULT_S

    def _register_startup(self, model: str, t0: float) -> None:
        self._startup_hist[model].append(time.time() - t0)

    def _decide(
        self, snap: Dict[str, List[_ModelAgg]]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        now = time.time()
        if now - self._last_scale_ts < COOLDOWN_S:
            return ([], [])

        # Boot-floor (minimum - never scale below what we saw first)
        if not self._bootstrapped:
            for m, reps in snap.items():
                self._boot_floor[m] = len(reps)
            self._bootstrapped = True

        # Gate for down-scaling: only if ANY model queue ≥ trigger
        high_backlog_models = {
            m
            for m, reps in snap.items()
            if sum(r.queue_depth for r in reps) >= DOWNSCALE_TRIGGER_QD
        }
        allow_downscale = bool(high_backlog_models)

        to_add: list[Tuple[str, str]] = []
        to_kill: list[str] = []

        for model, reps in snap.items():
            current_replica_count = len(reps)
            qd   = sum(r.queue_depth for r in reps)
            infl = sum(r.in_flight for r in reps)
            lat_s = max(r.lat_ms for r in reps) / 1000.0
            wait_s = max(qd - infl, 0) * lat_s
            boot_s = self._estimate_startup_s(model)

            # ── SCALE-UP ──────────────────────────────────────────────
            need_up = wait_s > boot_s * 2
            if need_up and current_replica_count < MAX_REPLICAS_PER_MODEL:
                to_add.append((model, reps[0].worker_id))
                continue  # do not consider down-scale for the same model

            # ── SCALE-DOWN (only if another model is backlogged) ─────
            if (
                allow_downscale
                and model not in high_backlog_models
                and qd == 0
                and infl == 0
                and current_replica_count > max(1, self._boot_floor[model])
            ):
                to_kill.append(reps[0].worker_id)

        return to_add, to_kill

    def _apply(
        self, decisions: Tuple[List[Tuple[str, str]], List[str]]
    ) -> None:
        up, down = decisions
        if not up and not down:
            return
        self._last_scale_ts = time.time()

        for model, wid in up:
            _LOG.info("Scaling UP %s (clone from %s)", model, wid)
            self._launch_replica(model)

        for wid in down:
            _LOG.info("Scaling DOWN — draining worker %s", wid)
            self._shutdown_worker(wid)

    # ───────────────────────── Worker ops ─────────────────────────
    def _launch_replica(self, model: str) -> None:
        self._pending_launch[model] = time.time()
        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUN_WORKER_PS1),
            "-Models",
            model,
        ]
        try:
            subprocess.Popen(
                cmd, creationflags=subprocess.CREATE_NEW_CONSOLE
            )  # noqa: S603,S607
            _LOG.debug("Spawned worker via %s", " ".join(cmd))
        except FileNotFoundError:
            _LOG.error("Run-Worker.ps1 not found at %s", RUN_WORKER_PS1)

    def _shutdown_worker(self, wid: str) -> None:
        REGISTRY._db.pop(wid, None)
        _LOG.info("Marked worker %s for shutdown", wid)
