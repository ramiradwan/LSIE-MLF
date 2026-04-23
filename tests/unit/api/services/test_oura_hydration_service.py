from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid5

import pytest

from services.api.services.oura_hydration_service import OuraHydrationService

_UUID_NAMESPACE = uuid5(__import__("uuid").NAMESPACE_URL, "lsie-mlf:oura-hydration")


class _FakeRedis:
    def __init__(self, notifications: list[str | bytes]) -> None:
        self._notifications = list(notifications)
        self.rpush_calls: list[tuple[str, str]] = []

    def lpop(self, queue_name: str) -> str | bytes | None:
        assert queue_name == "physio:hydrate"
        if not self._notifications:
            return None
        return self._notifications.pop(0)

    def rpush(self, queue_name: str, payload: str) -> None:
        self.rpush_calls.append((queue_name, payload))


def _notification(*, data_type: str = "heartrate", subject_role: str = "streamer") -> str:
    return json.dumps(
        {
            "unique_id": "11111111-1111-1111-1111-111111111111",
            "subject_role": subject_role,
            "event_type": "daily_update",
            "data_type": data_type,
            "start_datetime": "2026-04-20T12:00:00Z",
            "end_datetime": "2026-04-20T12:05:00Z",
        }
    )


def _ibi_resource(*, resource_id: str = "hr-1") -> dict[str, Any]:
    return {
        "data": [
            {
                "id": resource_id,
                "timestamp": "2026-04-20T12:00:00Z",
                "end_datetime": "2026-04-20T12:05:00Z",
                "ibi_ms": [800.0, 810.0, 790.0],
                "heart_rate_bpm": [75, 74, 76],
                "sample_interval_s": 1,
            }
        ]
    }


def _session_resource(*, resource_id: str = "sess-1") -> dict[str, Any]:
    return {
        "data": [
            {
                "id": resource_id,
                "start_datetime": "2026-04-20T12:00:00Z",
                "end_datetime": "2026-04-20T12:30:00Z",
                "rmssd_ms": [22.5],
                "heart_rate_bpm": [61],
                "motion_items": [0.0],
                "sample_interval_s": 300,
            }
        ]
    }


def _service(
    redis_client: _FakeRedis,
    *,
    resource: dict[str, Any] | None = None,
    clock: Any = lambda: datetime(2026, 4, 20, 12, 10, tzinfo=UTC),
) -> tuple[OuraHydrationService, MagicMock]:
    oura_client = MagicMock()
    oura_client.get_json.return_value = resource if resource is not None else _ibi_resource()
    service = OuraHydrationService(
        redis_client=redis_client,
        oura_client=oura_client,
        clock=clock,
    )
    return service, oura_client


def test_drain_once_drains_all_notifications_and_enqueues_schema_complete_events() -> None:
    redis_client = _FakeRedis(
        [_notification(), _notification(data_type="session", subject_role="operator")]
    )
    service, oura_client = _service(redis_client, resource=_ibi_resource())
    oura_client.get_json.side_effect = [_ibi_resource(), _session_resource()]

    drained = service.drain_once()

    assert drained == 2
    assert len(redis_client.rpush_calls) == 2
    assert [call[0] for call in redis_client.rpush_calls] == ["physio:events", "physio:events"]

    event_1 = json.loads(redis_client.rpush_calls[0][1])
    event_2 = json.loads(redis_client.rpush_calls[1][1])

    assert event_1 == {
        "unique_id": event_1["unique_id"],
        "event_type": "physiological_chunk",
        "provider": "oura",
        "subject_role": "streamer",
        "source_kind": "ibi",
        "window_start_utc": "2026-04-20T12:00:00Z",
        "window_end_utc": "2026-04-20T12:05:00Z",
        "ingest_timestamp_utc": "2026-04-20T12:10:00Z",
        "payload": {
            "sample_interval_s": 1,
            "valid_sample_count": 3,
            "expected_sample_count": 300,
            "derivation_method": "provider",
            "ibi_ms_items": [800.0, 810.0, 790.0],
            "heart_rate_items_bpm": [75, 74, 76],
            "rmssd_items_ms": None,
            "motion_items": None,
        },
    }
    assert event_2 == {
        "unique_id": event_2["unique_id"],
        "event_type": "physiological_chunk",
        "provider": "oura",
        "subject_role": "operator",
        "source_kind": "session",
        "window_start_utc": "2026-04-20T12:00:00Z",
        "window_end_utc": "2026-04-20T12:30:00Z",
        "ingest_timestamp_utc": "2026-04-20T12:10:00Z",
        "payload": {
            "sample_interval_s": 300,
            "valid_sample_count": 1,
            "expected_sample_count": 6,
            "derivation_method": "provider",
            "ibi_ms_items": None,
            "heart_rate_items_bpm": [61],
            "rmssd_items_ms": [22.5],
            "motion_items": [0.0],
        },
    }


def test_source_kind_is_restricted_to_ibi_or_session() -> None:
    redis_client = _FakeRedis([_notification(data_type="spo2")])
    service, oura_client = _service(redis_client)

    drained = service.drain_once()

    assert drained == 1
    oura_client.get_json.assert_not_called()
    assert redis_client.rpush_calls == []


def test_deterministic_unique_id_is_stable_for_same_resource_window() -> None:
    notification = _notification()
    resource = _ibi_resource(resource_id="stable-resource")

    first_redis = _FakeRedis([notification])
    second_redis = _FakeRedis([notification])
    first_service, _ = _service(first_redis, resource=resource)
    second_service, _ = _service(second_redis, resource=resource)

    first_service.drain_once()
    second_service.drain_once()

    first_event = json.loads(first_redis.rpush_calls[0][1])
    second_event = json.loads(second_redis.rpush_calls[0][1])
    expected_unique_id = str(
        uuid5(
            _UUID_NAMESPACE,
            "streamer|ibi|stable-resource|2026-04-20T12:00:00+00:00|2026-04-20T12:05:00+00:00",
        )
    )

    assert first_event["event_type"] == "physiological_chunk"
    assert first_event["unique_id"] == second_event["unique_id"] == expected_unique_id
    UUID(first_event["unique_id"])


def test_malformed_notification_is_warn_and_skip() -> None:
    redis_client = _FakeRedis(["not-json", json.dumps({"subject_role": "streamer"})])
    service, oura_client = _service(redis_client)

    drained = service.drain_once()

    assert drained == 2
    oura_client.get_json.assert_not_called()
    assert redis_client.rpush_calls == []


def test_malformed_provider_records_are_skipped_before_enqueue() -> None:
    redis_client = _FakeRedis([_notification()])
    service, _ = _service(
        redis_client,
        resource={
            "data": [
                {"id": "bad-1", "timestamp": "2026-04-20T12:00:00Z", "ibi_ms": [-5]},
                "not-a-dict",
            ]
        },
    )

    drained = service.drain_once()

    assert drained == 1
    assert redis_client.rpush_calls == []


def test_missing_config_noops_without_draining_or_fetching() -> None:
    redis_client = _FakeRedis([_notification()])
    service = OuraHydrationService(
        redis_client=redis_client,
        oura_client=None,
        oura_client_factory=lambda: None,
    )

    drained = service.drain_once()

    assert drained == 0
    assert redis_client.rpush_calls == []


def test_per_item_failures_do_not_escape_run_forever() -> None:
    redis_client = _FakeRedis([_notification()])
    oura_client = MagicMock()
    oura_client.get_json.side_effect = RuntimeError("boom")
    sleep_calls: list[float] = []

    class _StopLoopError(Exception):
        pass

    def stop_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        raise _StopLoopError

    service = OuraHydrationService(
        redis_client=redis_client,
        oura_client=oura_client,
        sleep=stop_sleep,
    )

    with pytest.raises(_StopLoopError):
        service.run_forever(idle_sleep_s=0.25)

    assert sleep_calls == [0.25]
    assert redis_client.rpush_calls == []
