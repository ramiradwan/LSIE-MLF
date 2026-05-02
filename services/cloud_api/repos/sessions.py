"""Repository writes for cloud session lifecycle state."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from packages.schemas.cloud import CloudSessionCreateRequest, CloudSessionEndRequest

_INSERT_SESSION_SQL = """
    INSERT INTO sessions (session_id, stream_url, started_at)
    VALUES (%(session_id)s, %(stream_url)s, %(started_at)s)
    ON CONFLICT (session_id) DO NOTHING
"""

_END_SESSION_SQL = """
    UPDATE sessions
    SET ended_at = %(ended_at)s
    WHERE session_id = %(session_id)s
"""


def insert_session(cur: Any, session_id: UUID, request: CloudSessionCreateRequest) -> int:
    cur.execute(
        _INSERT_SESSION_SQL,
        {
            "session_id": str(session_id),
            "stream_url": f"cloud://{request.client_id}",
            "started_at": request.started_at_utc,
        },
    )
    return int(cur.rowcount)


def end_session(cur: Any, session_id: UUID, request: CloudSessionEndRequest) -> int:
    cur.execute(
        _END_SESSION_SQL,
        {
            "session_id": str(session_id),
            "ended_at": request.ended_at_utc,
        },
    )
    return int(cur.rowcount)
