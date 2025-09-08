import os, tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
import threading
import time

DAY_AVG_FILE = Path("./metrics/data/daily.json")
HOURLY_FILE = Path("./metrics/data/hourly.json")
DASHBOARD_FILE = Path("./metrics/data/gpu_dashboard.json")
DAY_AVG_FILE.parent.mkdir(parents=True, exist_ok=True)
HOURLY_FILE.parent.mkdir(parents=True, exist_ok=True)
REFRESH_INTERVAL_SEC = 5

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _norm_sample(d: dict) -> dict:
    return {
        "timestamp": d["timestamp"],
        "uuid": d["uuid"],
        "worker_id": d["worker_id"],
        "gpu_index": (
            int(d["gpu_index"])
            if "gpuIndex" in d
            else int(d.get("gpu_index", 0))
        ),
        "gpu_name": d.get("gpuName") or d.get("gpu_name", ""),
        "util_gpu": (
            int(d["util_gpu"])
            if "utilGpu" in d
            else int(d.get("util_gpu", 0))
        ),
        "util_mem": (
            int(d["utilMem"])
            if "utilMem" in d
            else int(d.get("util_mem", 0))
        ),
        "active_jobs": int(d.get("activeJobs") or d.get("active_jobs", 0)),
        "models_hosted": d.get("modelsHosted") or d.get("models_hosted", []),
    }


def _hour_key(ts: str):
    # 'YYYY-MM-DDTHH:MM:SSZ' → ('YYYY-MM-DD', HH)
    day = ts[:10]
    hour = int(ts[11:13])
    return day, hour


_SAVE_LOCK = threading.Lock()


def _load_json_or(path, default):
    """
    Robust JSON loader:
    - returns `default` if file doesn't exist
    - retries on JSONDecodeError / PermissionError
    - if file stays empty, returns `default` instead of raising
    """
    path = Path(path).resolve()
    if not path.exists():
        return default

    for attempt in range(8):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                # empty file -> transient or corrupted
                raise json.JSONDecodeError("empty", "", 0)
            return json.loads(text)
        except Exception:
            # backoff 50ms, 100ms, ... (össz ~1.8s)
            time.sleep(0.05 * (attempt + 1))
            continue
    # still bad → try to quarantine zero-length file, then return default
    try:
        if path.exists() and path.stat().st_size == 0:
            quarantine = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
            os.replace(str(path), str(quarantine))
    except Exception:
        pass
    return default


def _save_json(path, obj):
    """
    Atomically write JSON to `path` (Windows-friendly).
    - Writes to a temp file in the same directory
    - fsync() the file
    - retries os.replace() on PermissionError (Windows file locks)
    """
    path = Path(path).resolve()  # use absolute path to avoid cwd issues
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(obj, ensure_ascii=False, indent=2)

    with _SAVE_LOCK:  # serialize concurrent writers in-process
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name,  # e.g. "hourly.json"
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # retry loop for Windows 'Access is denied' when dest is momentarily locked
            # exponential-ish backoff: ~0.05, 0.1, 0.15, ... seconds
            last_err = None
            for attempt in range(10):
                try:
                    os.replace(tmp, str(path))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.05 * (attempt + 1))
            if last_err:
                raise last_err
        finally:
            # best-effort: if replace succeeded, tmp is gone; else delete
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass


def _pick_current_day_from_hour(hour_doc: dict) -> str | None:
    if not hour_doc:
        return None
    day = hour_doc.get("day")
    if day:
        return day
    ts = (hour_doc.get("latest") or {}).get("timestamp")
    if ts:
        try:
            if isinstance(ts, (int, float)):
                day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                day = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d")
            return day
        except Exception:
            return None
    return None


def _iter_by_gpu_from_hour(hour_doc: dict):
    if not hour_doc:
        return
    hours = hour_doc.get("hours") or []
    for h in hours:
        by_gpu = h.get("by_gpu") or []
        for slot in by_gpu.values():
            yield slot


def _iter_days_from_daily(day_doc: dict):
    """Yield (day_key, by_gpu, meta) from daily v2 only."""
    if not day_doc or day_doc.get("schema_version") != 2:
        return
    days = day_doc.get("days") or {}
    for day_key in sorted(days.keys()):
        d = days[day_key] or {}
        yield day_key, (d.get("by_gpu") or {}), (d.get("meta") or {})


def _aggregate_today_from_hour(hour_doc: dict) -> dict:
    """Build a daily-like by_gpu dict for *today* by summing all hour slots in HOURLY_FILE."""
    if not hour_doc or "hours" not in hour_doc or not hour_doc["hours"]:
        return {}

    # acc[key] = per-GPU accumulator
    acc = {}
    for h in hour_doc["hours"]:
        by_gpu = h.get("by_gpu") or {}
        for key, slot in by_gpu.items():
            tgt = acc.setdefault(key, {
                "worker_id": slot.get("worker_id"),
                "uuid": slot.get("uuid"),
                "gpu_index": slot.get("gpu_index"),
                "gpu_name": slot.get("gpu_name"),
                "samples": 0,
                "sum_util_gpu": 0.0,
                "max_util_gpu": 0.0,
                "sum_util_mem": 0.0,        # optional if you need it later
                "over_90_hours": 0.0,       # if you keep it hourly as count of hours ≥90%
                "job_count_sum": 0,
                "models_hosted": {}
            })
            # sum counters
            s  = int(slot.get("samples", 0) or 0)
            su = float(slot.get("sum_util_gpu", 0.0) or 0.0)
            mu = float(slot.get("max_util_gpu", 0.0) or 0.0)
            jm = int(slot.get("job_count_sum", 0) or 0)

            tgt["samples"]       += s
            tgt["sum_util_gpu"]  += su
            tgt["max_util_gpu"]   = max(tgt["max_util_gpu"], mu)
            tgt["job_count_sum"] += jm

            # merge models_hosted
            for m, cnt in (slot.get("models_hosted") or {}).items():
                tgt["models_hosted"][m] = tgt["models_hosted"].get(m, 0) + int(cnt or 0)

            # if you track "over_90_hours" per hour, increment when this hour had >=90%
            if mu >= 90.0:
                tgt["over_90_hours"] += 1.0

    return acc


def _get_daily_by_gpu(day_doc: dict):
    """Return (day_str, by_gpu) from daily v2. If missing, return (None, {})."""
    if not day_doc:
        return None, {}
    if "days" in day_doc:
        if not day_doc["days"]:
            return None, {}
        last_day = max(day_doc["days"].keys())  # YYYY-MM-DD lexicographic works
        return last_day, day_doc["days"][last_day].get("by_gpu", {}) or {}
    return None, {}


def _pick_latest_hour(hour_doc: dict):
    """Return (day_str, hour_int, by_gpu) from hourly v2 latest hour."""
    if not hour_doc:
        return None, None, {}
    if "day" in hour_doc and "hours" in hour_doc:
        day = hour_doc["day"]
        hours = hour_doc.get("hours", [])
        if not hours:
            return day, None, {}
        # choose max hour present
        hslot = max(hours, key=lambda h: h.get("hour", -1))
        return day, hslot.get("hour"), hslot.get("by_gpu", {}) or {}

    return None, None, {}


def _current_active_node_count(hour_doc: dict) -> int:
    """Count currently active nodes (workers) in the latest hour."""
    _, _, by_gpu = _pick_latest_hour(hour_doc)
    if not by_gpu:
        return 0
    workers = set()
    for slot in by_gpu.values():
        wid = slot.get("worker_id")
        if wid:
            workers.add(wid)
    return len(workers)


def _estimate_spph_from_hours(hours: list[dict]) -> int:
    """Estimate samples-per-hour (SPPH) from v2 hourly slots using the MEAN of positive 'samples'."""
    counts = []
    for h in (hours or []):
        by_gpu = h.get("by_gpu") or {}
        for slot in by_gpu.values():
            s = int(slot.get("samples", 0) or 0)
            if s > 0:
                counts.append(s)
    if not counts:
        return 0
    # mean is the right time-scale estimator for converting samples -> hours
    return int(round(sum(counts) / len(counts)))
