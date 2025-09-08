from citadel_master.metrics.core.metrics_general import (
    _save_json,
    _load_json_or,
    _norm_sample,
    _hour_key,
    HOURLY_FILE,
)
from citadel_master.metrics.core.metrics_day import flush_hour_into_daily
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def _new_hourly_v2(day: str, old_doc: dict | None = None) -> dict:
    latest = {}
    if old_doc and isinstance(old_doc, dict):
        latest = deepcopy(old_doc.get("latest") or {})

    latest.setdefault("timestamp", None)
    latest.setdefault("by_gpu", {})
    latest.setdefault("by_worker", {})

    return {
        "schema_version": 2,
        "day": day,
        "latest": latest,
        "hours": []
    }


def update_hourly(sample: dict, worker_qd: dict[str, int] | None = None):
    s = _norm_sample(sample)
    day, hour = _hour_key(s["timestamp"])

    doc = _load_json_or(HOURLY_FILE, None)
    if not doc or doc.get("schema_version") != 2:
        doc = _new_hourly_v2(day, doc)

    if doc["day"] != day:
        flush_hour_into_daily(doc)
        doc = _new_hourly_v2(day, doc)
        logger.info("New day %s: flushed daily, started new hourly", day)

    hours = doc["hours"]
    hslot = next((h for h in hours if h["hour"] == hour), None)
    if not hslot:
        hslot = {"hour": hour, "by_gpu": {}}
        hours.append(hslot)

    key = f'{s["worker_id"]}|{s["uuid"]}|{s["gpu_index"]}'
    slot = hslot["by_gpu"].get(key)
    if not slot:
        slot = {
            "worker_id": s["worker_id"],
            "uuid": s["uuid"],
            "gpu_index": s["gpu_index"],
            "gpu_name": s["gpu_name"],
            "samples": 0,
            "sum_util_gpu": 0,
            "sum_util_mem": 0,
            "sum_active_jobs": 0,
            "max_util_gpu": 0,
            "max_util_mem": 0,
            "max_active_jobs": 0,
            "over_90_samples": 0,
            "job_count_sum": 0,
            "models_hosted": {}
        }
        hslot["by_gpu"][key] = slot

    # Aggregate sample
    slot["samples"] += 1
    slot["sum_util_gpu"] += s["util_gpu"]
    slot["sum_util_mem"] += s["util_mem"]
    slot["sum_active_jobs"] += s["active_jobs"]
    slot["max_util_gpu"] = max(slot["max_util_gpu"], s["util_gpu"])
    slot["max_util_mem"] = max(slot["max_util_mem"], s["util_mem"])
    slot["max_active_jobs"] = max(slot["max_active_jobs"], s["active_jobs"])
    slot["over_90_samples"] += 1 if s["util_gpu"] >= 90 else 0
    slot["job_count_sum"] += s["active_jobs"]

    for m in (s["models_hosted"] or []):
        slot["models_hosted"][m] = slot["models_hosted"].get(m, 0) + 1

    latest = doc.get("latest")
    if not latest or "by_gpu" not in latest:
        doc["latest"] = {"timestamp": None, "by_gpu": {}}
        latest = doc["latest"]

    doc["latest"]["timestamp"] = s["timestamp"]
    doc["latest"]["by_gpu"][key] = {
        "worker_id": s["worker_id"],
        "gpu_id": f'{s["uuid"]}:{s["gpu_index"]}',
        "gpu_util": s["util_gpu"],
    }

    bw = doc["latest"].setdefault("by_worker", {})
    if worker_qd:
        for wid, qd in worker_qd.items():
            w = bw.setdefault(wid, {
                "in_flight": 0,
                "tasks_served": 0,
                "queue_depth": 0,
                "current_model": ""
            })
            w["queue_depth"] = int(qd if qd is not None else 0)

    _save_json(HOURLY_FILE, doc)
