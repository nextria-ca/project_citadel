from datetime import timezone, datetime
import logging
from citadel_master.metrics.core.metrics_general import (
    DASHBOARD_FILE,
    DAY_AVG_FILE,
    HOURLY_FILE,
    REFRESH_INTERVAL_SEC,
    _aggregate_today_from_hour,
    _current_active_node_count,
    _estimate_spph_from_hours,
    _get_daily_by_gpu,
    _iter_by_gpu_from_hour,
    _iter_days_from_daily,
    # _iter_days_from_daily,
    _load_json_or,
    _pick_current_day_from_hour,
    _pick_latest_hour,
    _save_json
)
import numpy as np

logger = logging.getLogger(__name__)


def _safe_div(a, b):
    return (a / b) if b else 0.0


def _build_heatmap(hour_doc: dict) -> list:
    day, hour_now, by_gpu = _pick_latest_hour(hour_doc)
    if not day or hour_now is None:
        return []

    def empty_day():
        return [
            {
                "hour": h,
                "gpu_utilization": 0.0,
                "active_nodes": 0,
                "job_count": 0,
                "samples": 0
            }
            for h in range(24)
        ]

    active_workers = set()
    for slot in (by_gpu or {}).values():
        wid = slot.get("worker_id")
        if wid:
            active_workers.add(wid)
    active_nodes_count = len(active_workers)

    host_stats = {}
    for _, slot in (by_gpu or {}).items():
        worker_id = slot.get("worker_id")
        if not worker_id:
            continue
        hs = host_stats.setdefault(worker_id, {"sum_util": 0.0, "samples": 0, "job_sum": 0})
        hs["sum_util"] += float(slot.get("sum_util_gpu", 0.0) or 0.0)
        hs["samples"] += int(slot.get("samples", 0) or 0)
        hs["job_sum"] += int(slot.get("job_count_sum", 0) or 0)

    by_host = []
    for worker_id, agg in host_stats.items():
        days = {day: empty_day()}
        val = _safe_div(agg["sum_util"], agg["samples"])
        days[day][hour_now]["gpu_utilization"] = val
        days[day][hour_now]["active_nodes"] = active_nodes_count 
        days[day][hour_now]["job_count"] = agg["job_sum"]
        days[day][hour_now]["samples"] = agg["samples"]
        by_host.append({"worker_id": worker_id, "days": days})
    return by_host


def _build_per_node(hour_doc: dict, day_doc: dict) -> dict:
    per_node = {}

    # 24h (daily) source: daily file if present; else aggregate today's hours from HOURLY
    today = (hour_doc or {}).get("day")
    day_key, by_gpu_daily = _get_daily_by_gpu(day_doc)
    if not by_gpu_daily or (today and day_key != today):

        by_gpu_daily = _aggregate_today_from_hour(hour_doc)

    # aggregate "24h" per host from by_gpu_daily
    hosts = {}
    for _, slot in (by_gpu_daily or {}).items():
        worker_id = slot.get("worker_id")
        if not worker_id:
            continue
        entry = hosts.setdefault(worker_id, {
            "gpus": {},
            "sum_util_daily": 0.0,
            "samples_daily": 0,
            "over_90_hours": 0.0,
            "job_count_sum": 0,
            "models_counter": {}
        })
        entry["gpus"][str(slot.get("gpu_index", 0))] = (
            slot.get("gpu_index", 0),
            slot.get("gpu_name", ""),
            slot.get("uuid", "")
        )
        entry["sum_util_daily"] += float(slot.get("sum_util_gpu", 0.0) or 0.0)
        entry["samples_daily"]  += int(slot.get("samples", 0) or 0)
        entry["over_90_hours"]  += float(slot.get("over_90_hours", 0.0) or 0.0)
        entry["job_count_sum"]  += int(slot.get("job_count_sum", 0) or 0)
        for m, cnt in (slot.get("models_hosted") or {}).items():
            entry["models_counter"][m] = entry["models_counter"].get(m, 0) + int(cnt or 0)

    # 1h source: latest hour only
    _, _, by_gpu_hour = _pick_latest_hour(hour_doc)
    hour_agg = {}
    for _, slot in (by_gpu_hour or {}).items():
        worker_id = slot.get("worker_id")
        if not worker_id:
            continue
        ha = hour_agg.setdefault(worker_id, {"sum": 0.0, "s": 0})
        ha["sum"] += float(slot.get("sum_util_gpu", 0.0) or 0.0)
        ha["s"]   += int(slot.get("samples", 0) or 0)

    latest_by_worker = {}
    if hour_doc and "latest" in hour_doc:
        latest_by_worker = (hour_doc["latest"].get("by_worker") or {})

    for worker_id, info in hosts.items():
        gpus_list = [
            {"gpu_index": idx, "gpu_name": name, "uuid": uuid}
            for _, (idx, name, uuid) in sorted(info["gpus"].items(), key=lambda x: int(x[0]))
        ]
        s24 = int(info["samples_daily"])
        s1h = int(hour_agg.get(worker_id, {}).get("s", 0))
        avg_24h = _safe_div(info["sum_util_daily"], s24)
        avg_1h  = _safe_div(hour_agg.get(worker_id, {}).get("sum", 0.0), s1h)

        w = latest_by_worker.get(worker_id, {})
        current_model   = w.get("current_model", "")
        in_flight       = int(w.get("in_flight", 0) or 0)
        queue_depth     = int(w.get("queue_depth", 0) or 0)
        tasks_served_ct = int(w.get("tasks_served", 0) or 0)

        per_node[worker_id] = {
            "gpus": gpus_list,
            "avg_util_current_hour": avg_1h,
            "avg_util_today": avg_24h,
            "samples_current_hour": s1h,
            "samples_today": s24,
            "time_over_90_hours": float(info["over_90_hours"]),
            "tasks_served_count": tasks_served_ct,
            "current_model": current_model,
            "in_flight": in_flight,
            "queue_depth": queue_depth,
            "flag": ""
        }

    return per_node

def _estimate_spph_today(hour_doc: dict) -> int:
    hours = (hour_doc or {}).get("hours") or []
    return _estimate_spph_from_hours(hours)


def _build_kpis(day_doc: dict | None, hour_doc: dict | None) -> dict:
    today = (hour_doc or {}).get("day")

    total_samples = 0
    total_sum_util = 0.0
    max_gpu_util = 0.0
    node_set = set()

    past_gpu_hours = 0.0
    if day_doc:
        for dkey, by_gpu, meta in _iter_days_from_daily(day_doc):
            if today and dkey == today:
                continue
            for slot in by_gpu.values():
                s  = int(slot.get("samples", 0) or 0)
                su = float(slot.get("sum_util_gpu", 0.0) or 0.0)
                mu = float(slot.get("max_util_gpu", 0.0) or 0.0)
                gh = float(slot.get("gpu_hours", 0.0) or 0.0)

                total_samples += s
                total_sum_util += su
                if mu > max_gpu_util:
                    max_gpu_util = mu
                node_set.add((slot.get("worker_id"), slot.get("uuid")))
                past_gpu_hours += gh

    today_samples = 0
    today_sum_util = 0.0
    today_max = 0.0
    if hour_doc:
        for h in (hour_doc.get("hours") or []):
            for slot in (h.get("by_gpu") or {}).values():
                s  = int(slot.get("samples", 0) or 0)
                su = float(slot.get("sum_util_gpu", 0.0) or 0.0)
                mu = float(slot.get("max_util_gpu", 0.0) or 0.0)
                today_samples += s
                today_sum_util += su
                if mu > today_max:
                    today_max = mu
                node_set.add((slot.get("worker_id"), slot.get("uuid")))
    spph_today = _estimate_spph_today(hour_doc)
    today_gpu_hours = (today_samples / spph_today) if spph_today > 0 else 0.0

    total_samples += today_samples
    total_sum_util += today_sum_util
    if today_max > max_gpu_util:
        max_gpu_util = today_max

    return {
        "avg_gpu_utilization_percent": (total_sum_util / total_samples) if total_samples else 0.0,
        "max_gpu_utilization_percent": max_gpu_util,
        "gpu_hours_per_day": past_gpu_hours + today_gpu_hours,
        "active_nodes": _current_active_node_count(hour_doc),  
        "samples_total": total_samples,
    }


def _daily_aggregate(by_gpu: dict):
    total_s, total_gpu_sum, nodes = 0, 0.0, set()
    for v in by_gpu.values():
        total_s += v.get("samples", 0)
        total_gpu_sum += v.get("sum_util_gpu", 0.0)
        nodes.add((v.get("worker_id"), v.get("uuid")))
    avg = _safe_div(total_gpu_sum, total_s)
    return avg, len(nodes), total_s


def _build_trends_from_daily(day_doc: dict) -> dict:
    daily_avg = {}
    daily_nodes = {}
    for day_key, by_gpu, _ in _iter_days_from_daily(day_doc):
        avg, node_count, _ = _daily_aggregate(by_gpu)
        daily_avg[day_key] = avg
        daily_nodes[day_key] = node_count
    return {"daily_avg": daily_avg, "daily_active_nodes": daily_nodes}


def _forecast_linear(y, total_gpus: int = -1, projection_days: int = 14, threshold: float = 90.0):
    """Tiny linear regression forecast on a 1D array of daily avgs."""
    y = np.asarray(y, dtype=float)
    if y.size < 2 or not np.isfinite(y).all():
        return "Insufficient data", "N/A", float("nan")

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)  # y ≈ slope*x + intercept

    future_x = np.arange(len(x), len(x) + projection_days, dtype=float)
    preds = slope * future_x + intercept

    hit = np.where(preds >= threshold)[0]
    weeks = f"~{round((hit[0] + 1) / 7.0, 1)} weeks" if hit.size else "N/A"

    if total_gpus <= 0:
        return weeks, "N/A", float(preds[-1])

    last_pred = float(preds[-1])
    need = max(0, int(np.ceil(total_gpus * (last_pred / threshold - 1.0))))
    action = f"+{need} GPUs or redistribute" if need > 0 else "0 (OK)"
    return weeks, action, last_pred


def _build_forecast_from_daily(day_doc: dict, projection_days: int = 14, threshold: float = 90.0) -> dict:
    """Build forecast directly from day_doc. Keeps your existing calculate_mesh call site."""
    series = []
    for day_key, by_gpu, _ in _iter_days_from_daily(day_doc):
        avg, _, _ = _daily_aggregate(by_gpu)
        series.append((day_key, float(avg)))

    if len(series) < 2:
        return {"projected_saturation": "Insufficient data", "suggested_action": "N/A"}

    series.sort(key=lambda t: t[0])
    _, vals = zip(*series)
    vals = list(vals)

    total_gpus = -1
    if day_doc:
        if day_doc.get("schema_version") == 2 and "days" in day_doc:
            last_day = max(day_doc["days"].keys())
            last_by_gpu = (day_doc["days"][last_day] or {}).get("by_gpu", {}) or {}
            total_gpus = len(last_by_gpu)
        elif day_doc.get("schema_version") == 1:
            total_gpus = len(day_doc.get("by_gpu", {}) or {})

    weeks, action, _ = _forecast_linear(vals, total_gpus=total_gpus,
                                        projection_days=projection_days,
                                        threshold=threshold)

    return {
        "projected_saturation": weeks,
        "suggested_action": action,
        "method": "linear_regression",
        "projection_days": projection_days
    }


def calculate_mesh():
    try:
        # start_ts = datetime.now(timezone.utc)
        # logger.info("Starting dashboard calculation at %s", start_ts.isoformat())

        # t0 = time.perf_counter()
        hour_doc  = _load_json_or(HOURLY_FILE, None)
        day_doc   = _load_json_or(DAY_AVG_FILE, None)
        # t1 = time.perf_counter()
        # logger.debug("Loaded source files in %.3f sec", t1 - t0)

        per_node = _build_per_node(hour_doc, day_doc)
        # t2 = time.perf_counter()
        # logger.debug("1. Built per_node in %.3f sec", t2 - t1)

        heatmap = _build_heatmap(hour_doc)
        # t3 = time.perf_counter()
        # logger.debug("2. Built heatmap in %.3f sec", t3 - t2)

        kpis = _build_kpis(day_doc, hour_doc)
        # t4 = time.perf_counter()
        # logger.debug("3. Built KPIs in %.3f sec", t4 - t3)

        trends = _build_trends_from_daily(day_doc)
        # t5 = time.perf_counter()
        # logger.debug("4. Built trends in %.3f sec", t5 - t4)

        forecast = _build_forecast_from_daily(day_doc)
        # t6 = time.perf_counter()
        # logger.debug("5. Built forecast in %.3f sec", t6 - t5)

        dashboard = {
            "title": "MESH GPU CAPACITY & FORECASTING DASHBOARD",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "key_performance_indicators": kpis,
            "per_node": per_node,
            "heatmap": heatmap,
            "trends": trends,
            "forecast": forecast,
            "refresh_interval_sec": REFRESH_INTERVAL_SEC,
        }

        _save_json(DASHBOARD_FILE, dashboard)
        # t7 = time.perf_counter()
        # logger.info("Dashboard saved to %s", DASHBOARD_FILE)
        # logger.info("Dashboard calculated in %.3f sec", t7 - t0)

        return dashboard
    except Exception as e:
        logger.exception("calculate_mesh failed: %s", e)
        return None
