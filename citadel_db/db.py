# citadel_shared/db.py  (additions only)

from __future__ import annotations
import contextlib, typing as _t
import psycopg2.extras as _ext
import psycopg2
import logging
_LOG = logging.getLogger(__name__)


@contextlib.contextmanager
def dict_cursor() -> _t.Iterator[tuple[psycopg2.extensions.connection,
                                       psycopg2.extras.DictCursor]]:
    """
    Yield (connection, DictCursor).

    • Every statement is DEBUG-logged.
    • On success → COMMIT.
    • On exception → ROLLBACK and re-raise, so callers can see the error.
    """
    con = psycopg2.connect()  # ← your DSN comes from env vars / pgpass as before
    cur = con.cursor(cursor_factory=_ext.DictCursor)
    try:
        yield con, cur          # ── control returns to caller’s `with` block
        con.commit()
    except Exception as exc:    # any DB error or other exception
        con.rollback()
        _LOG.exception("❌ DB error - rolled back: %s", exc)
        raise                  # bubble up so gRPC returns the error
    finally:
        cur.close()
        con.close()



