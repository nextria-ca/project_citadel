from citadel_master.metrics.core.metrics_general import (
    _estimate_spph_from_hours,
    _now_iso,
    _save_json,
    _load_json_or,
    DAY_AVG_FILE,
)


RETENTION_DAYS = 7




def _ensure_daily_v2(doc: dict) -> dict:
    """Ensure daily file is v2 (map of days)."""
    if not doc:
        return {"schema_version": 2, "days": {}}
    if doc.get("schema_version") == 2 and "days" in doc:
        return doc
    return {"schema_version": 2, "days": {}}


def _aggregate_hour_slot_into_daily_slot(hour_slot: dict, daily_slot: dict):
    """Accumulate one hourly GPU slot into the daily GPU slot."""
    s = max(hour_slot.get("samples", 0), 0)
    daily_slot["samples"] += s
    daily_slot["sum_util_gpu"] += hour_slot.get("sum_util_gpu", 0.0)
    daily_slot["sum_util_mem"] += hour_slot.get("sum_util_mem", 0.0)
    daily_slot["sum_active_jobs"] += hour_slot.get("sum_active_jobs", 0.0)
    daily_slot["max_util_gpu"] = max(daily_slot["max_util_gpu"], hour_slot.get("max_util_gpu", 0.0))
    daily_slot["max_util_mem"] = max(daily_slot["max_util_mem"], hour_slot.get("max_util_mem", 0.0))
    daily_slot["max_active_jobs"] = max(daily_slot["max_active_jobs"], hour_slot.get("max_active_jobs", 0.0))
    # Note: these two are placeholders; if you later track >90% time or true GPU hours, compute here.
    daily_slot["over_90_hours"] += 0.0
    daily_slot["gpu_hours"] += 0.0
    daily_slot["job_count_sum"] += hour_slot.get("job_count_sum", 0)
    for m, cnt in (hour_slot.get("models_hosted") or {}).items():
        daily_slot["models_hosted"][m] = daily_slot["models_hosted"].get(m, 0) + cnt


def _get_or_make_daily_gpu_slot(daily_by_gpu: dict, key: str, hour_slot: dict) -> dict:
    """Fetch or initialize the daily slot for a GPU key."""
    slot = daily_by_gpu.get(key)
    if slot:
        return slot
    slot = {
        "worker_id": hour_slot.get("worker_id", ""),
        "uuid": hour_slot.get("uuid", ""),
        "gpu_index": hour_slot.get("gpu_index", 0),
        "gpu_name": hour_slot.get("gpu_name", ""),
        "samples": 0,
        "sum_util_gpu": 0.0,
        "sum_util_mem": 0.0,
        "sum_active_jobs": 0.0,
        "max_util_gpu": 0.0,
        "max_util_mem": 0.0,
        "max_active_jobs": 0.0,
        "over_90_hours": 0.0,
        "gpu_hours": 0.0,
        "job_count_sum": 0,
        "models_hosted": {}
    }
    daily_by_gpu[key] = slot
    return slot


def _prune_days(daily_v2: dict, keep: int = RETENTION_DAYS):
    """Keep only the most recent `keep` days in daily_v2."""
    days = daily_v2.get("days", {})
    if len(days) <= keep:
        return
    keys_sorted = sorted(days.keys())
    to_drop = keys_sorted[:-keep]
    for k in to_drop:
        days.pop(k, None)


def flush_hour_into_daily(hour_doc: dict):
    """
    Aggregate hourly v2 into daily v2 and persist a true gpu_hours per GPU for that day.
    """
    if not hour_doc or hour_doc.get("schema_version") != 2:
        return

    day = hour_doc.get("day")
    hours_list = hour_doc.get("hours") or []
    if not day or not hours_list:
        return

    # Estimate cadence for this day
    spph = _estimate_spph_from_hours(hours_list)

    daily_doc = _ensure_daily_v2(_load_json_or(DAY_AVG_FILE, None))
    day_bucket = daily_doc["days"].get(day, {"by_gpu": {}, "meta": {}})

    for h in hours_list:
        by_gpu = h.get("by_gpu") or {}
        for key, hour_slot in by_gpu.items():
            daily_slot = _get_or_make_daily_gpu_slot(day_bucket["by_gpu"], key, hour_slot)

            _aggregate_hour_slot_into_daily_slot(hour_slot, daily_slot)

            if spph > 0:
                s = int(hour_slot.get("samples", 0) or 0)
                daily_slot["gpu_hours"] = float(daily_slot.get("gpu_hours", 0.0)) + (s / spph)

            # Count an hour as "over_90" if this hour peaked ≥ 90%
            mu = float(hour_slot.get("max_util_gpu", 0.0) or 0.0)
            if mu >= 90.0:
                daily_slot["over_90_hours"] = float(daily_slot.get("over_90_hours", 0.0) or 0.0) + 1.0

    day_bucket["meta"]["samples_per_hour_est"] = int(spph)
    day_bucket["meta"]["computed_at"] = _now_iso()

    daily_doc["days"][day] = day_bucket
    _prune_days(daily_doc, RETENTION_DAYS)
    _save_json(DAY_AVG_FILE, daily_doc)
