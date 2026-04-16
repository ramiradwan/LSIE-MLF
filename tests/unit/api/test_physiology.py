"""Tests for services/api/routes/physiology.py."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import json
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch


@dataclass
class _InvocationResult:
    status_code: int
    body: dict[str, Any] | None = None
    detail: str | None = None


class _FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _get_physiology_module() -> Any:
    mod_name = "services.api.routes.physiology"
    sys.modules.pop(mod_name, None)
    return importlib.import_module(mod_name)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _valid_payload(*, event_id: str = "11111111-1111-1111-1111-111111111111") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "subject_role": "streamer",
        "data": {
            "timestamp": "2026-03-13T12:00:00Z",
            "rmssd": 42.5,
            "heart_rate": 61,
        },
    }


def _invoke_webhook(
    payload_bytes: bytes,
    *,
    secret: str = "test-secret",
    signature: str | None = None,
    redis_client: MagicMock | None = None,
) -> tuple[_InvocationResult, MagicMock]:
    redis_module = MagicMock()
    mock_client = redis_client or MagicMock()
    redis_module.Redis.from_url.return_value = mock_client

    with (
        patch.dict(
            "os.environ",
            {
                "OURA_WEBHOOK_SECRET": secret,
                "REDIS_URL": "redis://localhost:6379/0",
            },
            clear=False,
        ),
        patch.dict("sys.modules", {"redis": redis_module}),
    ):
        mod = _get_physiology_module()
        request = _FakeRequest(payload_bytes)
        effective_signature = signature if signature is not None else _sign(payload_bytes, secret)

        try:
            body = _run_async(mod.oura_webhook(request, x_oura_signature=effective_signature))
            return _InvocationResult(status_code=200, body=body), mock_client
        except Exception as exc:
            return (
                _InvocationResult(
                    status_code=getattr(exc, "status_code", 500),
                    detail=getattr(exc, "detail", str(exc)),
                ),
                mock_client,
            )


def test_valid_webhook_accepted_returns_200() -> None:
    payload = _valid_payload()
    payload_bytes = json.dumps(payload).encode("utf-8")
    mock_client = MagicMock()
    mock_client.set.return_value = True

    result, mock_client = _invoke_webhook(payload_bytes, redis_client=mock_client)

    assert result.status_code == 200
    assert result.body == {
        "status": "accepted",
        "event_id": payload["event_id"],
    }
    mock_client.set.assert_called_once_with(
        f"physio:seen:{payload['event_id']}",
        "1",
        nx=True,
        ex=3600,
    )
    mock_client.rpush.assert_called_once()

    queue_name, event_json = mock_client.rpush.call_args.args
    enqueued_event = json.loads(event_json)
    assert queue_name == "physio:events"
    assert enqueued_event["provider"] == "oura"
    assert enqueued_event["subject_role"] == "streamer"
    assert enqueued_event["event_type"] == "physiological_sample"
    assert enqueued_event["payload"]["rmssd_ms"] == 42.5
    assert enqueued_event["payload"]["heart_rate_bpm"] == 61
    assert enqueued_event["payload"]["sample_window_s"] == 300


def test_invalid_hmac_signature_rejected_returns_401() -> None:
    payload_bytes = json.dumps(_valid_payload()).encode("utf-8")

    result, mock_client = _invoke_webhook(payload_bytes, signature="bad-signature")

    assert result.status_code == 401
    assert result.detail == "Invalid signature"
    mock_client.set.assert_not_called()
    mock_client.rpush.assert_not_called()


def test_malformed_json_rejected_returns_422() -> None:
    payload_bytes = b'{"event_id": "11111111-1111-1111-1111-111111111111",'

    result, mock_client = _invoke_webhook(payload_bytes)

    assert result.status_code == 422
    assert result.detail == "Malformed JSON"
    mock_client.set.assert_not_called()
    mock_client.rpush.assert_not_called()


def test_missing_subject_role_rejected_returns_422() -> None:
    payload = _valid_payload()
    del payload["subject_role"]
    payload_bytes = json.dumps(payload).encode("utf-8")

    result, mock_client = _invoke_webhook(payload_bytes)

    assert result.status_code == 422
    assert result.detail == "Invalid or missing subject_role: None"
    mock_client.set.assert_not_called()
    mock_client.rpush.assert_not_called()


def test_duplicate_event_returns_200_with_duplicate_status() -> None:
    payload = _valid_payload()
    payload_bytes = json.dumps(payload).encode("utf-8")
    mock_client = MagicMock()
    mock_client.set.return_value = False

    result, mock_client = _invoke_webhook(payload_bytes, redis_client=mock_client)

    assert result.status_code == 200
    assert result.body == {
        "status": "duplicate",
        "event_id": payload["event_id"],
    }
    mock_client.set.assert_called_once_with(
        f"physio:seen:{payload['event_id']}",
        "1",
        nx=True,
        ex=3600,
    )
    mock_client.rpush.assert_not_called()


def test_redis_failure_returns_503() -> None:
    payload_bytes = json.dumps(_valid_payload()).encode("utf-8")
    mock_client = MagicMock()
    mock_client.set.return_value = True
    mock_client.rpush.side_effect = RuntimeError("redis unavailable")

    result, mock_client = _invoke_webhook(payload_bytes, redis_client=mock_client)

    assert result.status_code == 503
    assert result.detail == "Service temporarily unavailable"
    mock_client.delete.assert_called_once_with("physio:seen:11111111-1111-1111-1111-111111111111")
