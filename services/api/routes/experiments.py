"""Experiment state and admin endpoints — §4.E.1.

This router preserves the legacy read shape under `/api/v1/experiments`
while adding an additive admin write surface for creating experiments,
adding arms, and patching human-owned arm metadata.

Posterior-owned numeric state (`alpha_param`, `beta_param`, selection
rollups) remains read-only and is never writable through this module.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException

from packages.schemas.experiments import (
    ExperimentArmCreateRequest,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from services.api.db.connection import get_connection, put_connection
from services.api.repos import experiments_queries as q
from services.api.services.experiment_admin_service import (
    ExperimentAdminService,
    ExperimentAlreadyExistsError,
    ExperimentArmAlreadyExistsError,
    ExperimentArmNotFoundError,
    ExperimentMutationValidationError,
    ExperimentNotFoundError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dependency provider
# ----------------------------------------------------------------------


def get_admin_service() -> ExperimentAdminService:
    return ExperimentAdminService()


_AdminDep = Depends(get_admin_service)


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


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


def _serialize_payload(payload: Any) -> Any:
    """Recursively serialize response payloads for direct-route unit tests."""
    if isinstance(payload, dict):
        return {key: _serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_serialize_payload(value) for value in payload]
    if hasattr(payload, "model_dump"):
        return _serialize_payload(payload.model_dump())
    return _serialize(payload)


# ----------------------------------------------------------------------
# Legacy read endpoints (shape preserved)
# ----------------------------------------------------------------------


@router.get("/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    """List all experiment IDs."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            payload = _serialize_payload(q.fetch_experiment_ids(cur))
            return cast(list[dict[str, Any]], payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to list experiments: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Get Thompson Sampling arm state for one experiment."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            arms = q.fetch_experiment_rows(cur, experiment_id)
            if not arms:
                raise HTTPException(
                    status_code=404,
                    detail=f"No experiment found with id '{experiment_id}'",
                )
            payload = _serialize_payload(
                {
                    "experiment_id": experiment_id,
                    "arms": arms,
                }
            )
            return cast(dict[str, Any], payload)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to get experiment %s: %s", experiment_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)


# ----------------------------------------------------------------------
# Admin write endpoints
# ----------------------------------------------------------------------


@router.post("/experiments", status_code=201)
async def create_experiment(
    request: ExperimentCreateRequest,
    service: ExperimentAdminService = _AdminDep,
) -> dict[str, Any]:
    """Create a new experiment and seed its initial arms at Beta(1,1)."""
    try:
        payload = _serialize_payload(service.create_experiment(request))
        return cast(dict[str, Any], payload)
    except ExperimentAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExperimentMutationValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to create experiment %s: %s",
            request.experiment_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/experiments/{experiment_id}/arms", status_code=201)
async def add_experiment_arm(
    experiment_id: str,
    request: ExperimentArmCreateRequest,
    service: ExperimentAdminService = _AdminDep,
) -> dict[str, Any]:
    """Add one new arm to an existing experiment at Beta(1,1)."""
    try:
        payload = _serialize_payload(service.add_arm(experiment_id, request))
        return cast(dict[str, Any], payload)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperimentArmAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExperimentMutationValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to add arm to experiment %s: %s", experiment_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.patch("/experiments/{experiment_id}/arms/{arm_id}")
async def patch_experiment_arm(
    experiment_id: str,
    arm_id: str,
    request: ExperimentArmPatchRequest,
    service: ExperimentAdminService = _AdminDep,
) -> dict[str, Any]:
    """Patch supported arm metadata without touching posterior state."""
    try:
        payload = _serialize_payload(service.patch_arm(experiment_id, arm_id, request))
        return cast(dict[str, Any], payload)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperimentArmNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperimentMutationValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to patch arm %s on experiment %s: %s",
            arm_id,
            experiment_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.delete("/experiments/{experiment_id}/arms/{arm_id}")
async def delete_experiment_arm(
    experiment_id: str,
    arm_id: str,
    service: ExperimentAdminService = _AdminDep,
) -> dict[str, Any]:
    """Delete an unused arm, or disable/end-date it to preserve posterior history."""
    try:
        payload = _serialize_payload(service.delete_arm(experiment_id, arm_id))
        return cast(dict[str, Any], payload)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperimentArmNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperimentMutationValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to delete arm %s on experiment %s: %s",
            arm_id,
            experiment_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
