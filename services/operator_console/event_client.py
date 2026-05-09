from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from packages.schemas.operator_console import OperatorEventEnvelope
from services.operator_console.api_client import ApiError

EVENTS_ENDPOINT = "/api/v1/operator/state/events"


@dataclass(frozen=True)
class SseFrame:
    event_id: str | None
    event_type: str | None
    data: str
    retry_ms: int | None = None


class EventStreamResponse(Protocol):
    def __iter__(self) -> Iterator[bytes | str]: ...

    def close(self) -> None: ...


class OperatorEventClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def stream_events(
        self,
        *,
        last_event_id: str | None = None,
        on_open: Callable[[EventStreamResponse], None] | None = None,
    ) -> Iterator[OperatorEventEnvelope]:
        request = Request(f"{self._base_url}{EVENTS_ENDPOINT}", method="GET")
        request.add_header("Accept", "text/event-stream")
        request.add_header("Cache-Control", "no-cache")
        if last_event_id is not None:
            request.add_header("Last-Event-ID", last_event_id)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                stream_response = response
                if on_open is not None:
                    on_open(stream_response)
                for frame in parse_sse_frames(_iter_response_lines(stream_response)):
                    yield validate_event_frame(frame)
        except HTTPError as exc:
            status = exc.code
            raise ApiError(
                message=_parse_http_error(exc),
                endpoint=EVENTS_ENDPOINT,
                status_code=status,
                retryable=status >= 500,
            ) from exc
        except URLError as exc:
            raise ApiError(
                message=f"cannot reach event stream: {exc.reason}",
                endpoint=EVENTS_ENDPOINT,
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise ApiError(
                message="event stream timed out",
                endpoint=EVENTS_ENDPOINT,
                retryable=True,
            ) from exc
        except UnicodeDecodeError as exc:
            raise ApiError(
                message=f"malformed event stream: {exc}",
                endpoint=EVENTS_ENDPOINT,
                retryable=True,
            ) from exc


def parse_sse_frames(lines: Iterable[str]) -> Iterator[SseFrame]:
    event_id: str | None = None
    event_type: str | None = None
    data_lines: list[str] = []
    retry_ms: int | None = None
    for line in lines:
        stripped = line.rstrip("\r\n")
        if stripped == "":
            if data_lines:
                yield SseFrame(
                    event_id=event_id,
                    event_type=event_type,
                    data="\n".join(data_lines),
                    retry_ms=retry_ms,
                )
            event_id = None
            event_type = None
            data_lines = []
            retry_ms = None
            continue
        if stripped.startswith(":"):
            continue
        field, value = _split_sse_field(stripped)
        if field == "id":
            event_id = value
        elif field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)
        elif field == "retry":
            retry_ms = _parse_retry(value)
    if data_lines:
        yield SseFrame(
            event_id=event_id,
            event_type=event_type,
            data="\n".join(data_lines),
            retry_ms=retry_ms,
        )


def validate_event_frame(frame: SseFrame) -> OperatorEventEnvelope:
    try:
        payload = json.loads(frame.data)
    except json.JSONDecodeError as exc:
        raise ApiError(
            message=f"malformed SSE JSON: {exc}",
            endpoint=EVENTS_ENDPOINT,
            retryable=True,
            payload_excerpt=frame.data[:512],
        ) from exc
    try:
        envelope = OperatorEventEnvelope.model_validate(payload)
    except ValidationError as exc:
        raise ApiError(
            message=f"SSE event failed validation: {exc}",
            endpoint=EVENTS_ENDPOINT,
            retryable=True,
            payload_excerpt=frame.data[:512],
        ) from exc
    if frame.event_id is not None and frame.event_id != envelope.event_id:
        raise ApiError(
            message="SSE frame id does not match envelope event_id",
            endpoint=EVENTS_ENDPOINT,
            retryable=True,
            payload_excerpt=frame.data[:512],
        )
    if frame.event_type is not None and frame.event_type != envelope.event_type:
        raise ApiError(
            message="SSE frame event does not match envelope event_type",
            endpoint=EVENTS_ENDPOINT,
            retryable=True,
            payload_excerpt=frame.data[:512],
        )
    return envelope


def _iter_response_lines(response: EventStreamResponse) -> Iterator[str]:
    for raw_line in response:
        if isinstance(raw_line, bytes):
            yield raw_line.decode("utf-8")
        else:
            yield str(raw_line)


def _split_sse_field(line: str) -> tuple[str, str]:
    if ":" not in line:
        return line, ""
    field, value = line.split(":", 1)
    if value.startswith(" "):
        value = value[1:]
    return field, value


def _parse_retry(value: str) -> int | None:
    try:
        retry_ms = int(value)
    except ValueError:
        return None
    return retry_ms if retry_ms >= 0 else None


def _parse_http_error(exc: HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
    except Exception:
        return str(exc)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        return str(detail) if not isinstance(detail, str) else detail
    return raw
