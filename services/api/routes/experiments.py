"""
Experiment State Endpoints — §4.E.1

REST endpoints for querying Thompson Sampling experiment arm state
(alpha/beta posterior parameters). This is the read-only counterpart
to the Thompson Sampling update path in persist_metrics.

§2 step 7 — Parameterized queries only.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from services.api.db.connection import get_connection, put_connection

router = APIRouter()
logger = logging.getLogger(__name__)

# §2 step 7 — Parameterized SELECT for experiment arms
_LIST_EXPERIMENTS_SQL: str = """
    SELECT DISTINCT experiment_id
    FROM experiments
    ORDER BY experiment_id
"""

_GET_EXPERIMENT_SQL: str = """
    SELECT experiment_id, arm, alpha_param, beta_param, updated_at
    FROM experiments
    WHERE experiment_id = %(experiment_id)s
    ORDER BY arm
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


@router.get("/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    """
    List all experiment IDs.

    §4.E.1 — Discover available Thompson Sampling experiments.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(_LIST_EXPERIMENTS_SQL)
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to list experiments: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """
    Get Thompson Sampling arm state for an experiment.

    §4.E.1 — Returns alpha/beta posterior parameters for each arm,
    enabling the operator to inspect convergence and arm probabilities.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(_GET_EXPERIMENT_SQL, {"experiment_id": experiment_id})
            arms = _rows_to_dicts(cur)

            if not arms:
                raise HTTPException(
                    status_code=404,
                    detail=f"No experiment found with id '{experiment_id}'",
                )

            return {
                "experiment_id": experiment_id,
                "arms": arms,
            }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to get experiment %s: %s", experiment_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
