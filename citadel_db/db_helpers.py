"""
High-level helpers the rest of the code should call.
Keeps SQL in one place and stays tiny.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from citadel_db.db import dict_cursor

_SQL_SELECT = re.compile(r"^\s*select\b", re.I)
_SQL_WRITE = re.compile(r"^\s*(insert|update|delete)\b", re.I)  # ← allow simple writes

# ── domain models (dataclasses) ─────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Worker:
    worker_id: str
    endpoint: str
    can_infer: bool
    can_train: bool
    registered: str


# ── query helpers ───────────────────────────────────────────────────
def list_workers(
    *, only_inferable: bool | None = None, limit: int | None = None
) -> Sequence[Worker]:
    q = """
        SELECT worker_id, endpoint, can_infer, can_train,
               registered::text
          FROM workers
    """
    params: list[object] = []
    if only_inferable is not None:
        q += " WHERE can_infer = %s"
        params.append(only_inferable)
    q += " ORDER BY registered DESC"
    if limit:
        q += " LIMIT %s"
        params.append(limit)

    with dict_cursor() as (_, cur):
        cur.execute(q, params)
        return [Worker(**row) for row in cur.fetchall()]


def models_for_worker(worker_id: str) -> list[str]:
    with dict_cursor() as (_, cur):
        cur.execute(
            """
            SELECT model_name FROM models
             WHERE worker_id = %s
        """,
            (worker_id,),
        )
        return [m[0] for m in cur.fetchall()]


def register_job(job_id: str, payload_json: str) -> None:
    """Insert a row representing a new training / inference job."""
    with dict_cursor() as (_, cur):
        cur.execute(
            """
        INSERT INTO jobs(job_id, payload, status, submitted)
        VALUES (%s, %s::jsonb, 'RECEIVED', now())
        """,
            (job_id, payload_json),
        )


# ────────────────────────────────────────────────────────────────────
def update_job_status(
    job_id: str,
    *,
    status: str,
    progress: Dict[str, Any] | None = None,
    message: str | None = None,
) -> None:
    """
    Persist the latest stage / progress atoms for a job.

    Parameters
    ----------
    status
        One of the Stage enum names (e.g. 'TRAINING').
    progress
        Arbitrary JSON payload merged into the existing `progress` column.
    message
        Optional human-readable status description.
    """
    with dict_cursor() as (_, cur):
        cur.execute(
            """
        UPDATE jobs
           SET status   = %s,
               progress = COALESCE(progress, '{}'::jsonb)
                         || %s::jsonb,
               last_msg = COALESCE(%s, last_msg),
               updated  = now()
         WHERE job_id   = %s
        """,
            (
                status,
                json.dumps(progress or {}),
                message,
                job_id,
            ),
        )


# ────────────────────────────────────────────────────────────────────
def run_query(sql: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]:
    """
    Execute a **read-only** query and return a list[dict].

    • refuses anything that is not a single SELECT
    """
    if not _SQL_SELECT.match(sql):
        raise ValueError("Only simple SELECT queries are allowed")

    params = list(params or [])
    with dict_cursor() as (_, cur):
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


# ────────────────────────────────────────────────────────────────────
def exec_sql(sql: str, params: Sequence[Any] | None = None) -> int:
    """
    Execute an INSERT / UPDATE / DELETE statement.

    Returns
    -------
    int
        Number of rows affected.
    """
    if not _SQL_WRITE.match(sql):
        raise ValueError("Only INSERT / UPDATE / DELETE are allowed")

    params = list(params or [])
    with dict_cursor() as (_, cur):
        cur.execute(sql, params)
        return cur.rowcount
