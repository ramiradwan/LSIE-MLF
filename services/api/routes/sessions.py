"""
Session Endpoints

REST endpoints for managing live stream inference sessions.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    """List all inference sessions."""
    # TODO: Implement
    raise NotImplementedError


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get session details and summary metrics."""
    # TODO: Implement
    raise NotImplementedError
