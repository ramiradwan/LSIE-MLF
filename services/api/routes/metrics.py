"""
Metrics Endpoints — §4.E

REST endpoints for querying inference metrics from the Persistent Store.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    session_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Query inference metrics from Persistent Store."""
    # TODO: Implement DB query via connection pool
    raise NotImplementedError


@router.get("/metrics/{session_id}/au12")
async def get_au12_timeseries(session_id: str) -> list[dict[str, Any]]:
    """Retrieve AU12 intensity time-series for a session."""
    # TODO: Implement per §11 variable extraction matrix
    raise NotImplementedError


@router.get("/metrics/{session_id}/acoustic")
async def get_acoustic_timeseries(session_id: str) -> list[dict[str, Any]]:
    """Retrieve pitch, jitter, shimmer time-series for a session."""
    # TODO: Implement per §11 variable extraction matrix
    raise NotImplementedError
