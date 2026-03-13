"""
Metrics Endpoints — §4.E

REST endpoints for querying inference metrics from the Persistent Store.
§2 step 7 — Parameterized queries only, DOUBLE PRECISION, TIMESTAMPTZ.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.api.db.connection import get_connection, put_connection

router = APIRouter()
logger = logging.getLogger(__name__)

# §2 step 7 — Parameterized SELECT queries for metrics
_SELECT_METRICS_SQL: str = """
    SELECT m.id, m.session_id, m.segment_id, m.timestamp_utc,
           m.au12_intensity, m.pitch_f0, m.jitter, m.shimmer, m.created_at
    FROM metrics m
    {where_clause}
    ORDER BY m.timestamp_utc DESC
    LIMIT %(limit)s
"""

_SELECT_AU12_SQL: str = """
    SELECT m.segment_id, m.timestamp_utc, m.au12_intensity
    FROM metrics m
    WHERE m.session_id = %(session_id)s AND m.au12_intensity IS NOT NULL
    ORDER BY m.timestamp_utc ASC
"""

_SELECT_ACOUSTIC_SQL: str = """
    SELECT m.segment_id, m.timestamp_utc, m.pitch_f0, m.jitter, m.shimmer
    FROM metrics m
    WHERE m.session_id = %(session_id)s
          AND (m.pitch_f0 IS NOT NULL OR m.jitter IS NOT NULL OR m.shimmer IS NOT NULL)
    ORDER BY m.timestamp_utc ASC
"""


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert cursor results to list of dicts using column names."""
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [
        {col: _serialize(val) for col, val in zip(columns, row)}
        for row in rows
    ]


def _serialize(val: Any) -> Any:
    """Serialize values for JSON response."""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


@router.get("/metrics")  # type: ignore[untyped-decorator]
async def get_metrics(
    session_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """
    Query inference metrics from Persistent Store.

    §2 step 7 — Parameterized queries, DOUBLE PRECISION, TIMESTAMPTZ.

    Args:
        session_id: Optional filter by session UUID.
        limit: Maximum rows to return (1–1000).
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            params: dict[str, Any] = {"limit": limit}

            if session_id is not None:
                where_clause = "WHERE m.session_id = %(session_id)s"
                params["session_id"] = session_id
            else:
                where_clause = ""

            # §2 step 7 — Parameterized query
            cur.execute(
                _SELECT_METRICS_SQL.format(where_clause=where_clause),
                params,
            )
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query metrics: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/metrics/{session_id}/au12")  # type: ignore[untyped-decorator]
async def get_au12_timeseries(session_id: str) -> list[dict[str, Any]]:
    """
    Retrieve AU12 intensity time-series for a session.

    §11 — AU12 Intensity Score from Variable Extraction Matrix.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # §2 step 7 — Parameterized query
            cur.execute(_SELECT_AU12_SQL, {"session_id": session_id})
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query AU12 timeseries: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/metrics/{session_id}/acoustic")  # type: ignore[untyped-decorator]
async def get_acoustic_timeseries(session_id: str) -> list[dict[str, Any]]:
    """
    Retrieve pitch, jitter, shimmer time-series for a session.

    §11 — Vocal Pitch, Jitter, Shimmer from Variable Extraction Matrix.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # §2 step 7 — Parameterized query
            cur.execute(_SELECT_ACOUSTIC_SQL, {"session_id": session_id})
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query acoustic timeseries: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
