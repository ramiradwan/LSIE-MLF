"""
Co-Modulation Index Readback — §7C, §4.E.2

REST endpoint for the rolling Pearson Co-Modulation Index between
streamer and operator RMSSD, persisted to ``comodulation_log`` by the
Module E pipeline.

§2 step 7 — Parameterized queries only. No joins into raw media.
§7C       — Returns null co_modulation_index when insufficient aligned
            non-stale pairs exist for the window (preserved from Persistent Store).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.api.db.connection import get_connection, put_connection

router = APIRouter()
logger = logging.getLogger(__name__)

# §7C — Rolling co-modulation index rows for a session, newest first.
_COMOD_SERIES_SQL: str = """
    SELECT c.session_id, c.window_end_utc, c.window_minutes,
           c.co_modulation_index, c.n_paired_observations,
           c.coverage_ratio, c.streamer_rmssd_mean,
           c.operator_rmssd_mean, c.created_at
    FROM comodulation_log c
    WHERE c.session_id = %(session_id)s
    ORDER BY c.window_end_utc DESC
    LIMIT %(limit)s
"""


def _serialize(val: Any) -> Any:
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [{col: _serialize(val) for col, val in zip(columns, row, strict=True)} for row in rows]


@router.get("/comodulation/{session_id}")
async def get_comodulation(
    session_id: str,
    limit: int = Query(100, ge=1, le=2000),
) -> list[dict[str, Any]]:
    """§7C — Return the rolling Co-Modulation Index history for a session.

    Rows are ordered newest-first. A null ``co_modulation_index`` value
    is preserved verbatim — the Module E writer emits null when the
    rolling window contains too few aligned non-stale observations
    (§7C semantics).

    Args:
        session_id: Session UUID whose comodulation_log rows to return.
        limit: Maximum rows (1-2000, default 100).
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                _COMOD_SERIES_SQL,
                {"session_id": session_id, "limit": limit},
            )
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query comodulation for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
