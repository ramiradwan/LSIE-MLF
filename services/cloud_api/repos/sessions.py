"""Repository writes for cloud session lifecycle state."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from packages.schemas.cloud import CloudSessionCreateRequest, CloudSessionEndRequest
from packages.schemas.data_tiers import DataTier, mark_data_tier

_INSERT_SESSION_SQL = mark_data_tier(
    """
    INSERT INTO sessions (session_id, stream_url, started_at)
    VALUES (%(session_id)s, %(stream_url)s, %(started_at)s)
    ON CONFLICT (session_id) DO NOTHING
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)

_END_SESSION_SQL = mark_data_tier(
    """
    UPDATE sessions
    SET ended_at = %(ended_at)s
    WHERE session_id = %(session_id)s
      AND stream_url = %(stream_url)s
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)


def insert_session(cur: Any, session_id: UUID, request: CloudSessionCreateRequest) -> int:
    cur.execute(
        _INSERT_SESSION_SQL,
        mark_data_tier(
            {
                "session_id": str(session_id),
                "stream_url": f"cloud://{request.client_id}",
                "started_at": request.started_at_utc,
            },
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
        ),
    )
    return int(cur.rowcount)


def end_session(
    cur: Any,
    session_id: UUID,
    request: CloudSessionEndRequest,
    *,
    client_id: str,
) -> int:
    cur.execute(
        _END_SESSION_SQL,
        mark_data_tier(
            {
                "session_id": str(session_id),
                "ended_at": request.ended_at_utc,
                "stream_url": f"cloud://{client_id}",
            },
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
        ),
    )
    return int(cur.rowcount)
