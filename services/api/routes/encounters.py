"""
Encounter Log Endpoints — §4.E.1

REST endpoints for querying the encounter audit log — the persistent
record of every Thompson Sampling reward computation.

§2 step 7 — Parameterized queries only.
§11 — Exposes all reward pipeline variables from the Variable Extraction Matrix.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.api.db.connection import get_connection, put_connection

router = APIRouter()
logger = logging.getLogger(__name__)

# §2 step 7 — Parameterized SELECT for encounter log entries
_LIST_ENCOUNTERS_SQL: str = """
    SELECT e.id, e.session_id, e.segment_id, e.experiment_id, e.arm,
           e.timestamp_utc, e.gated_reward, e.p90_intensity,
           e.semantic_gate, e.n_frames_in_window,
           e.au12_baseline_pre, e.stimulus_time, e.created_at
    FROM encounter_log e
    {where_clause}
    ORDER BY e.created_at DESC
    LIMIT %(limit)s
"""

# §4.E.1 — Per-arm aggregation for experiment analysis
_ENCOUNTER_SUMMARY_SQL: str = """
    SELECT
        e.arm,
        COUNT(*) AS encounter_count,
        COUNT(*) FILTER (WHERE e.n_frames_in_window > 0) AS valid_count,
        AVG(e.gated_reward) AS avg_reward,
        AVG(e.gated_reward) FILTER (WHERE e.n_frames_in_window > 0) AS avg_valid_reward,
        AVG(e.p90_intensity) FILTER (WHERE e.n_frames_in_window > 0) AS avg_p90,
        AVG(e.semantic_gate::numeric) AS gate_rate,
        AVG(e.n_frames_in_window) FILTER (WHERE e.n_frames_in_window > 0) AS avg_frames,
        MIN(e.timestamp_utc) AS first_encounter,
        MAX(e.timestamp_utc) AS last_encounter
    FROM encounter_log e
    WHERE e.experiment_id = %(experiment_id)s
    GROUP BY e.arm
    ORDER BY avg_valid_reward DESC NULLS LAST
"""


def _serialize(val: Any) -> Any:
    """Serialize values for JSON response."""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert cursor results to list of dicts using column names."""
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [{col: _serialize(val) for col, val in zip(columns, row, strict=True)} for row in rows]


@router.get("/encounters")
async def list_encounters(
    experiment_id: str | None = Query(None),
    arm: str | None = Query(None),
    valid_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """
    List encounter log entries with optional filters.

    §4.E.1 — Query the reward computation audit trail.
    §2 step 7 — Parameterized queries.

    Args:
        experiment_id: Filter by experiment (e.g., "greeting_line_v1").
        arm: Filter by specific arm (e.g., "warm_welcome").
        valid_only: If True, return only encounters with measurement-window frames.
        limit: Maximum rows (1-1000, default 100).
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            conditions: list[str] = []
            params: dict[str, Any] = {"limit": limit}

            if experiment_id is not None:
                conditions.append("e.experiment_id = %(experiment_id)s")
                params["experiment_id"] = experiment_id
            if arm is not None:
                conditions.append("e.arm = %(arm)s")
                params["arm"] = arm
            if valid_only:
                conditions.append("e.n_frames_in_window > 0")

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            cur.execute(
                _LIST_ENCOUNTERS_SQL.format(where_clause=where_clause),
                params,
            )
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query encounters: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/encounters/{experiment_id}/summary")
async def get_encounter_summary(experiment_id: str) -> list[dict[str, Any]]:
    """
    Per-arm encounter summary for an experiment.

    §4.E.1 — Aggregated reward statistics for experiment analysis.
    Returns avg reward, encounter count, semantic gate rate, and avg
    frame count for each arm. This is the endpoint that answers
    "do the AU12 values have real utility" by comparing reward
    distributions across arms.

    Args:
        experiment_id: The experiment to summarize (e.g., "greeting_line_v1").
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                _ENCOUNTER_SUMMARY_SQL,
                {"experiment_id": experiment_id},
            )
            results = _rows_to_dicts(cur)

            if not results:
                raise HTTPException(
                    status_code=404,
                    detail=f"No encounters found for experiment '{experiment_id}'",
                )

            return results
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to get encounter summary for %s: %s", experiment_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
