from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from citadel_shared.env_loader import load_role_env

load_role_env("master")
from citadel_shared.logging_setup import get_logger
from proto.python import registry_pb2

_LOG = get_logger(__name__)


class _Info:
    __slots__ = (
        "models",
        "endpoint",
        "last_seen",
        "batch_cfg",
        "dispatch_cfg",
        "replica_stats",
        "can_infer",
        "can_train",
    )

    def __init__(
        self,
        models: Sequence[str],
        endpoint: str,
        can_infer: bool,
        can_train: bool,
        batch_cfg: Mapping[str, Tuple[int, int]] | None = None,
        dispatch_cfg: Mapping[str, str] | None = None,
        replica_stats: List[dict] | None = None,
    ) -> None:
        self.models = list(models)
        self.endpoint = endpoint
        self.last_seen = time.time()
        self.batch_cfg = dict(batch_cfg or {})
        self.dispatch_cfg = dict(dispatch_cfg or {})
        self.replica_stats = replica_stats or []
        self.can_infer = bool(can_infer)
        self.can_train = bool(can_train)


class WorkerRegistry:
    STALE_AFTER: int = int(os.getenv("CITADEL_WORKER_STALE_AFTER", "90"))
    # how long to keep last-known info for pruned workers (seconds)
    GRAVEYARD_TTL: int = int(os.getenv("CITADEL_REGISTRY_GRAVEYARD_TTL_S", "3600"))

    def __init__(self) -> None:
        self._db: Dict[str, _Info] = {}
        # wid -> (last known info, tombstone timestamp)
        self._graveyard: Dict[str, tuple[_Info, float]] = {}

    # ──────────────────────────────────────────────────────────────
    #  Pruning
    # ──────────────────────────────────────────────────────────────
    def _prune_stale(self) -> None:
        now = time.time()
        dead = [
            wid for wid, inf in self._db.items() if now - inf.last_seen > self.STALE_AFTER
        ]
        for wid in dead:
            info = self._db.pop(wid)
            self._graveyard[wid] = (info, now)
            _LOG.warning(
                "Pruned stale worker %s (%.0f s inactive) – moved to graveyard for %d s",
                wid,
                self.STALE_AFTER,
                self.GRAVEYARD_TTL,
            )

        # purge expired tombstones
        expire_before = now - self.GRAVEYARD_TTL
        for twid, (_, ts) in list(self._graveyard.items()):
            if ts < expire_before:
                try:
                    del self._graveyard[twid]
                except KeyError:
                    pass

    def _rehydrate_from_graveyard(self, wid: str) -> bool:
        """If wid was recently pruned, restore it from the graveyard."""
        tomb = self._graveyard.get(wid)
        if not tomb:
            return False

        info, ts = tomb
        if time.time() - ts > self.GRAVEYARD_TTL:
            # tombstone expired; drop it
            try:
                del self._graveyard[wid]
            except KeyError:
                pass
            return False

        # re-register using last-known info
        self._db[wid] = info
        try:
            del self._graveyard[wid]
        except KeyError:
            pass
        _LOG.warning(
            "Heartbeat from unknown worker %s – re-registered from tombstone "
            "(endpoint=%s, models=%s)",
            wid,
            info.endpoint,
            ", ".join(info.models),
        )
        return True

    # ──────────────────────────────────────────────────────────────
    #  Registration
    # ──────────────────────────────────────────────────────────────
    def register(
        self,
        wid: str,
        models: Sequence[Any],
        endpoint: str,
        can_infer: bool,
        can_train: bool,
    ) -> None:
        names: List[str] = []
        batch_cfg: Dict[str, Tuple[int, int]] = {}
        dispatch_cfg: Dict[str, str] = {}

        for m in models:
            if isinstance(m, str):
                names.append(m)
                continue

            if isinstance(m, dict):
                name = m.get("model_name") or m.get("name")
                if not name:
                    continue
                names.append(name)

                bs = int(m.get("batch_size", 0) or 0)
                to = int(m.get("batch_timeout_ms", 0) or 0)
                if bs > 0:
                    batch_cfg[name] = (bs, to)

                disp = m.get("dispatch") or {}
                strat = disp.get("strategy")
                if strat:
                    dispatch_cfg[name] = str(strat)
                continue

            if registry_pb2 and isinstance(m, registry_pb2.ModelInfo):
                name: str | None = getattr(m, "model_name", None)
                if not name:
                    continue
                names.append(name)

                bs = int(getattr(m.batch, "batch_size", 0) or 0)
                to = int(getattr(m.batch, "timeout_ms", 0) or 0)
                if bs > 0:
                    batch_cfg[name] = (bs, to)

                strat = getattr(m, "dispatch_strategy", "")
                if strat:
                    dispatch_cfg[name] = str(strat)

        self._db[wid] = _Info(
            names,
            endpoint,
            can_infer,
            can_train,
            batch_cfg=batch_cfg,
            dispatch_cfg=dispatch_cfg,
        )

        # fresh registration removes any tombstone
        try:
            del self._graveyard[wid]
        except KeyError:
            pass

        _LOG.debug(
            "Registered worker %s - models=[%s], batching=%s, dispatch=%s, endpoint=%s",
            wid,
            ", ".join(names),
            batch_cfg or "{}",
            dispatch_cfg or "{}",
            endpoint,
        )

    # ──────────────────────────────────────────────────────────────
    #  Role-aware worker lists
    # ──────────────────────────────────────────────────────────────
    def inference_workers(self) -> List[str]:
        self._prune_stale()
        now = time.time()
        return [
            wid
            for wid, inf in self._db.items()
            if inf.can_infer and now - inf.last_seen < self.STALE_AFTER
        ]

    def training_workers(self) -> List[str]:
        self._prune_stale()
        now = time.time()
        return [
            wid
            for wid, inf in self._db.items()
            if inf.can_train and now - inf.last_seen < self.STALE_AFTER
        ]

    # ──────────────────────────────────────────────────────────────
    #  Heartbeat
    # ──────────────────────────────────────────────────────────────
    def heartbeat(self, wid: str, stats: Optional[List[dict]] = None) -> None:
        # unknown? try to resurrect from the graveyard
        if wid not in self._db:
            if self._rehydrate_from_graveyard(wid):
                inf = self._db[wid]
                inf.last_seen = time.time()
                if stats is not None:
                    inf.replica_stats = stats
                return
            _LOG.warning("Heartbeat from unknown worker %s - no tombstone; cannot re-register", wid)
            return

        inf = self._db[wid]
        inf.last_seen = time.time()
        if stats is not None:
            inf.replica_stats = stats

    # ──────────────────────────────────────────────────────────────
    #  Queries used around the code-base
    # ──────────────────────────────────────────────────────────────
    def healthy_workers_for(self, model: str) -> List[str]:
        return self.workers_for(model)

    def workers_for(self, model: str) -> List[str]:
        self._prune_stale()
        now = time.time()
        return [
            wid
            for wid, inf in self._db.items()
            if model in inf.models and now - inf.last_seen < self.STALE_AFTER
        ]

    def batch_policy_for(self, model: str) -> Optional[Tuple[int, int]]:
        self._prune_stale()
        for inf in self._db.values():
            if model in inf.batch_cfg:
                return inf.batch_cfg[model]
        return None

    def dispatch_strategy_for(self, model: str) -> Optional[str]:
        self._prune_stale()
        for inf in self._db.values():
            if model in inf.dispatch_cfg:
                return inf.dispatch_cfg[model]
        return None

    def all_models(self) -> List[str]:
        self._prune_stale()
        pool: set[str] = set()
        for inf in self._db.values():
            pool.update(inf.models)
        return sorted(pool)

    def endpoint(self, wid: str) -> str:
        self._prune_stale()
        return self._db[wid].endpoint

    def replica_stats(self, wid: str) -> List[dict]:
        self._prune_stale()
        return self._db.get(wid, _Info([], "", False, False)).replica_stats


REGISTRY = WorkerRegistry()


def _start_reaper(registry: WorkerRegistry, interval: int = 30) -> None:
    def _loop() -> None:
        while True:
            time.sleep(interval)
            registry._prune_stale()

    threading.Thread(target=_loop, daemon=True, name="citadel-worker-reaper").start()


_start_reaper(REGISTRY)
