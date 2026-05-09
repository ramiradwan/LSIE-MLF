from __future__ import annotations

import json
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from packages.schemas.operator_console import HealthSnapshot, HealthState, OverviewSnapshot
from services.operator_console.api_client import ApiError
from services.operator_console.event_client import (
    OperatorEventClient,
    SseFrame,
    parse_sse_frames,
    validate_event_frame,
)


def _event_payload(event_id: str = "overview:1") -> dict[str, Any]:
    generated_at = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    health = HealthSnapshot(generated_at_utc=generated_at, overall_state=HealthState.OK)
    overview = OverviewSnapshot(generated_at_utc=generated_at, health=health)
    return {
        "event_id": event_id,
        "event_type": "overview",
        "cursor": event_id,
        "generated_at_utc": generated_at.isoformat(),
        "payload": overview.model_dump(mode="json"),
    }


def test_parse_sse_frames_collects_multiline_data_and_metadata() -> None:
    payload = json.dumps(_event_payload())
    frames = list(
        parse_sse_frames(
            [
                ": keepalive\n",
                "id: overview:1\n",
                "event: overview\n",
                "retry: 1000\n",
                f"data: {payload[:20]}\n",
                f"data: {payload[20:]}\n",
                "\n",
            ]
        )
    )

    assert len(frames) == 1
    assert frames[0].event_id == "overview:1"
    assert frames[0].event_type == "overview"
    assert frames[0].retry_ms == 1000
    assert frames[0].data == payload[:20] + "\n" + payload[20:]


def test_validate_event_frame_returns_operator_event_envelope() -> None:
    frame = SseFrame(
        event_id="overview:1",
        event_type="overview",
        data=json.dumps(_event_payload()),
    )

    envelope = validate_event_frame(frame)

    assert envelope.event_id == "overview:1"
    assert isinstance(envelope.payload, OverviewSnapshot)


def test_parse_sse_frames_emits_eof_without_blank_line_and_ignores_invalid_retry() -> None:
    frames = list(
        parse_sse_frames(
            [
                "id: overview:1\n",
                "event: overview\n",
                "retry: nope\n",
                f"data: {json.dumps(_event_payload())}\n",
            ]
        )
    )

    assert len(frames) == 1
    assert frames[0].retry_ms is None
    assert frames[0].event_id == "overview:1"


def test_validate_event_frame_rejects_bad_envelope() -> None:
    frame = SseFrame(event_id="bad", event_type="overview", data=json.dumps({"event_id": "bad"}))

    with pytest.raises(ApiError) as exc_info:
        validate_event_frame(frame)

    assert exc_info.value.retryable is True
    assert "validation" in exc_info.value.message


def test_validate_event_frame_rejects_id_mismatch() -> None:
    frame = SseFrame(
        event_id="overview:wrong",
        event_type="overview",
        data=json.dumps(_event_payload()),
    )

    with pytest.raises(ApiError, match="id does not match"):
        validate_event_frame(frame)


def test_validate_event_frame_rejects_event_mismatch() -> None:
    frame = SseFrame(
        event_id="overview:1",
        event_type="health",
        data=json.dumps(_event_payload()),
    )

    with pytest.raises(ApiError, match="event does not match"):
        validate_event_frame(frame)


def test_validate_event_frame_wraps_malformed_json() -> None:
    with pytest.raises(ApiError) as exc_info:
        validate_event_frame(SseFrame(event_id=None, event_type=None, data="{"))

    assert exc_info.value.retryable is True
    assert "malformed SSE JSON" in exc_info.value.message


def test_client_sends_last_event_id_header_and_yields_envelope() -> None:
    payload = json.dumps(_event_payload("overview:2")).encode("utf-8")
    captured_headers: dict[str, str] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(
                [
                    b"id: overview:2\n",
                    b"event: overview\n",
                    b"data: " + payload + b"\n",
                    b"\n",
                ]
            )

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        captured_headers.update(dict(request.header_items()))
        return _Response()

    client = OperatorEventClient("http://api.test", timeout_seconds=1.0)
    with patch("services.operator_console.event_client.urlopen", fake_urlopen):
        events = list(client.stream_events(last_event_id="overview:1"))

    assert captured_headers["Last-event-id"] == "overview:1"
    assert captured_headers["Accept"] == "text/event-stream"
    assert events[0].event_id == "overview:2"


def test_client_on_open_runs_only_after_response_established() -> None:
    opened: list[object] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def __iter__(self) -> Any:
            return iter([f"data: {json.dumps(_event_payload())}\n".encode(), b"\n"])

    def fake_urlopen(_request: Any, *, timeout: float) -> _Response:
        del timeout
        assert opened == []
        return _Response()

    client = OperatorEventClient("http://api.test", timeout_seconds=1.0)
    with patch("services.operator_console.event_client.urlopen", fake_urlopen):
        events = list(client.stream_events(on_open=lambda response: opened.append(response)))

    assert len(events) == 1
    assert len(opened) == 1


def test_client_wraps_http_error_detail() -> None:
    class _HttpError(HTTPError):
        def read(self, n: int = -1, /) -> bytes:
            del n
            return b'{"detail":"nope"}'

    def fake_urlopen(_request: Any, *, timeout: float) -> object:
        del timeout
        raise _HttpError("http://api.test", 503, "unavailable", EmailMessage(), None)

    client = OperatorEventClient("http://api.test", timeout_seconds=1.0)
    with (
        patch("services.operator_console.event_client.urlopen", fake_urlopen),
        pytest.raises(ApiError) as exc_info,
    ):
        list(client.stream_events())

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True
    assert exc_info.value.message == "nope"


def test_client_wraps_connection_failure_as_retryable_api_error() -> None:
    def fake_urlopen(_request: Any, *, timeout: float) -> object:
        del timeout
        raise URLError("offline")

    client = OperatorEventClient("http://api.test", timeout_seconds=1.0)
    with (
        patch("services.operator_console.event_client.urlopen", fake_urlopen),
        pytest.raises(ApiError) as exc_info,
    ):
        list(client.stream_events())

    assert exc_info.value.retryable is True
