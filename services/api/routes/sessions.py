"""
Session Endpoints

REST endpoints for managing live stream inference sessions.
§2 step 7 — Parameterized queries only.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from services.api.db.connection import get_connection, put_connection

router = APIRouter()
logger = logging.getLogger(__name__)

# §2 step 7 — Parameterized SELECT queries for sessions
_LIST_SESSIONS_SQL: str = """
    SELECT s.session_id, s.stream_url, s.started_at, s.ended_at,
           COUNT(m.id) AS metric_count
    FROM sessions s
    LEFT JOIN metrics m ON s.session_id = m.session_id
    GROUP BY s.session_id
    ORDER BY s.started_at DESC
"""

_GET_SESSION_SQL: str = """
    SELECT s.session_id, s.stream_url, s.started_at, s.ended_at
    FROM sessions s
    WHERE s.session_id = %(session_id)s
"""

# §11 — Summary metric aggregation per session
_SESSION_SUMMARY_SQL: str = """
    SELECT
        COUNT(m.id) AS total_segments,
        AVG(m.au12_intensity) AS avg_au12,
        AVG(m.pitch_f0) AS avg_pitch_f0,
        AVG(m.jitter) AS avg_jitter,
        AVG(m.shimmer) AS avg_shimmer,
        MIN(m.timestamp_utc) AS first_segment_at,
        MAX(m.timestamp_utc) AS last_segment_at
    FROM metrics m
    WHERE m.session_id = %(session_id)s
"""


def _serialize(val: Any) -> Any:
    """Serialize values for JSON response."""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def _row_to_dict(cursor: Any) -> dict[str, Any] | None:
    """Convert single cursor row to dict."""
    if cursor.description is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    row: Any = cursor.fetchone()
    if row is None:
        return None
    return {col: _serialize(val) for col, val in zip(columns, row)}


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert cursor results to list of dicts."""
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [
        {col: _serialize(val) for col, val in zip(columns, row)}
        for row in rows
    ]


@router.get("/sessions")  # type: ignore[untyped-decorator]
async def list_sessions() -> list[dict[str, Any]]:
    """
    List all inference sessions with metric counts.

    §2 step 7 — Parameterized query against Persistent Store.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(_LIST_SESSIONS_SQL)
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to list sessions: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/sessions/{session_id}")  # type: ignore[untyped-decorator]
async def get_session(session_id: str) -> dict[str, Any]:
    """
    Get session details and summary metrics.

    §11 — Aggregated metrics from Variable Extraction Matrix.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # §2 step 7 — Parameterized query for session
            cur.execute(_GET_SESSION_SQL, {"session_id": session_id})
            session = _row_to_dict(cur)

            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            # §11 — Summary metric aggregation
            cur.execute(_SESSION_SUMMARY_SQL, {"session_id": session_id})
            summary = _row_to_dict(cur)

            if summary is not None:
                session["summary"] = summary

            return session
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to get session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
