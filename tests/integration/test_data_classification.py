from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest

from services.api.routes import physiology as physiology_route
from services.api.services.oura_hydration_service import OuraHydrationService
from services.worker.pipeline import analytics as analytics_module
from services.worker.pipeline.analytics import MetricsStore
from services.worker.pipeline.orchestrator import Orchestrator

_RAW_SENTINEL = "RAW_OURA_PAYLOAD_SHOULD_NOT_REACH_PERMANENT_ROW"
_SESSION_ID = "00000000-0000-4000-8000-000000000214"


class _FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeRedis:
    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}
        self._values: dict[str, str] = {}

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        _ = ex
        if nx and key in self._values:
            return False
        self._values[key] = value
        return True

    def delete(self, key: str) -> int:
        existed = key in self._values
        self._values.pop(key, None)
        return 1 if existed else 0

    def lpop(self, key: str) -> str | None:
        values = self._lists.setdefault(key, [])
        if not values:
            return None
        return values.pop(0)

    def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    def peek(self, key: str) -> list[str]:
        return list(self._lists.get(key, []))

    def close(self) -> None:
        return None


class _FakeOuraClient:
    def __init__(self, resource: dict[str, Any]) -> None:
        self.resource = resource

    def get_json(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        assert path == "/v2/usercollection/heartrate"
        assert query["start_datetime"]
        assert query["end_datetime"]
        return self.resource


class _RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.calls.append((sql, dict(params)))


class _RecordingConnection:
    def __init__(self) -> None:
        self.cursor_obj = _RecordingCursor()
        self.commits = 0
        self.rollbacks = 0
        self.isolation_levels: list[int] = []

    def cursor(self) -> _RecordingCursor:
        return self.cursor_obj

    def set_isolation_level(self, level: int) -> None:
        self.isolation_levels.append(level)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _run_webhook(fake_redis: _FakeRedis, payload: dict[str, Any], secret: str) -> dict[str, str]:
    body = json.dumps(payload).encode("utf-8")
    signature = _sign(body, secret)
    with (
        patch.dict("os.environ", {"OURA_WEBHOOK_SECRET": secret}, clear=False),
        patch.object(physiology_route, "_get_redis", return_value=fake_redis),
    ):
        return asyncio.run(
            physiology_route.oura_webhook(
                cast(Any, _FakeRequest(body)),
                x_oura_signature=signature,
            )
        )


@pytest.mark.audit_item("13.14")
def test_transient_oura_payload_normalizes_to_permanent_scalar_physiology_row() -> None:
    """§13.14 — raw Oura payloads stay transient; Permanent rows are scalar derivatives."""

    secret = "classification-secret"
    fake_redis = _FakeRedis()
    window_start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    window_end = window_start + timedelta(seconds=3)
    raw_notification = {
        "event_id": "provider-event-214",
        "subject_role": "streamer",
        "event_type": "updated",
        "data_type": "heart_rate",
        "start_datetime": window_start.isoformat(),
        "end_datetime": window_end.isoformat(),
        "raw_webhook_body": {"sentinel": _RAW_SENTINEL},
    }

    response = _run_webhook(fake_redis, raw_notification, secret)
    assert response["status"] == "accepted"
    hydration_items = fake_redis.peek("physio:hydrate")
    assert len(hydration_items) == 1
    assert _RAW_SENTINEL not in hydration_items[0]

    resource = {
        "data": [
            {
                "id": "ibi-window-214",
                "timestamp": window_start.isoformat(),
                "end_datetime": window_end.isoformat(),
                "ibi_ms": [800, 810, 795, 805],
                "heart_rate_bpm": [74, 75, 76, 75],
                "provider_raw_blob": {"sentinel": _RAW_SENTINEL},
            }
        ],
        "raw_provider_response": _RAW_SENTINEL,
    }
    service = OuraHydrationService(
        redis_client=fake_redis,
        oura_client=cast(Any, _FakeOuraClient(resource)),
        clock=lambda: window_end + timedelta(seconds=1),
    )
    assert service.drain_once() == 1

    event_payloads = fake_redis.peek("physio:events")
    assert len(event_payloads) == 1
    assert _RAW_SENTINEL not in event_payloads[0]

    orchestrator = Orchestrator(session_id=_SESSION_ID, experiment_id="exp-1")
    orchestrator._redis = fake_redis
    now_wall = window_end.timestamp() + 5.0
    with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
        orchestrator._drain_physio_events()
        snapshot = orchestrator._derive_physio_snapshot("streamer", now_wall=now_wall)

    assert snapshot is not None
    assert snapshot["rmssd_ms"] is not None
    assert snapshot["heart_rate_bpm"] == 75
    assert "payload" not in snapshot

    fake_conn = _RecordingConnection()
    store = MetricsStore()
    store._get_conn = lambda: fake_conn  # type: ignore[method-assign]
    store._put_conn = lambda conn: None  # type: ignore[method-assign]
    fake_psycopg2 = SimpleNamespace(extensions=SimpleNamespace(ISOLATION_LEVEL_READ_COMMITTED=1))
    with patch.object(analytics_module, "_import_psycopg2", return_value=fake_psycopg2):
        store.persist_physiology_snapshot(
            _SESSION_ID,
            "segment-214",
            "streamer",
            snapshot,
        )

    assert fake_conn.commits == 1
    assert fake_conn.rollbacks == 0
    assert len(fake_conn.cursor_obj.calls) == 1
    sql, permanent_row = fake_conn.cursor_obj.calls[0]
    assert "INSERT INTO physiology_log" in sql
    assert set(permanent_row) == {
        "session_id",
        "segment_id",
        "subject_role",
        "rmssd_ms",
        "heart_rate_bpm",
        "freshness_s",
        "is_stale",
        "provider",
        "source_kind",
        "derivation_method",
        "window_s",
        "validity_ratio",
        "is_valid",
        "source_timestamp_utc",
    }
    assert "payload" not in permanent_row
    assert _RAW_SENTINEL not in json.dumps(permanent_row, default=str)
