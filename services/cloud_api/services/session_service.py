"""Service boundary for cloud session lifecycle writes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from packages.schemas.cloud import (
    CloudSessionCreateRequest,
    CloudSessionCreateResponse,
    CloudSessionEndRequest,
    CloudSessionEndResponse,
)
from services.cloud_api.repos.sessions import end_session, insert_session
from services.cloud_api.services.transactions import run_in_transaction


class SessionNotFoundError(RuntimeError):
    pass


class SessionOwnershipError(RuntimeError):
    pass


class CloudSessionService:
    def create_session(
        self,
        request: CloudSessionCreateRequest,
        *,
        client_id: str,
    ) -> CloudSessionCreateResponse:
        if request.client_id != client_id:
            raise SessionOwnershipError("session client_id does not match authenticated client")
        session_id = uuid4()

        def _write(conn: Any) -> None:
            with conn.cursor() as cur:
                insert_session(cur, session_id, request)

        run_in_transaction(_write)
        return CloudSessionCreateResponse(
            session_id=session_id,
            client_id=request.client_id,
            created_at_utc=datetime.now(UTC),
        )

    def end_session(
        self,
        session_id: UUID,
        request: CloudSessionEndRequest,
        *,
        client_id: str,
    ) -> CloudSessionEndResponse:
        def _write(conn: Any) -> int:
            with conn.cursor() as cur:
                return end_session(cur, session_id, request, client_id=client_id)

        updated = run_in_transaction(_write)
        if updated == 0:
            raise SessionNotFoundError(f"session {session_id} not found")
        return CloudSessionEndResponse(session_id=session_id, ended_at_utc=request.ended_at_utc)
