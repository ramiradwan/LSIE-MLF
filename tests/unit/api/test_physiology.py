"""Tests for services/api/routes/physiology.py notification-only ingress."""

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
    sys.modules.pop("packages.schemas.physiology", None)
    return importlib.import_module(mod_name)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _valid_payload(*, event_id: str = "11111111-1111-1111-1111-111111111111") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "subject_role": "streamer",
        "event_type": "daily_update",
        "data_type": "heartrate",
        "start_datetime": "2026-03-13T12:00:00Z",
        "end_datetime": "2026-03-13T12:05:00Z",
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


def test_valid_webhook_accepted_enqueues_minimal_hydration_metadata() -> None:
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
    assert queue_name == "physio:hydrate"
    assert enqueued_event == {
        "unique_id": payload["event_id"],
        "subject_role": payload["subject_role"],
        "event_type": payload["event_type"],
        "data_type": payload["data_type"],
        "start_datetime": payload["start_datetime"],
        "end_datetime": payload["end_datetime"],
        "notification_received_utc": enqueued_event["notification_received_utc"],
    }
    assert isinstance(enqueued_event["notification_received_utc"], str)
    assert "T" in enqueued_event["notification_received_utc"]


def test_duplicate_event_returns_200_with_duplicate_status_and_no_second_enqueue() -> None:
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


def test_redis_setnx_failure_returns_503() -> None:
    payload_bytes = json.dumps(_valid_payload()).encode("utf-8")
    mock_client = MagicMock()
    mock_client.set.side_effect = RuntimeError("redis unavailable")

    result, mock_client = _invoke_webhook(payload_bytes, redis_client=mock_client)

    assert result.status_code == 503
    assert result.detail == "Service temporarily unavailable"
    mock_client.rpush.assert_not_called()
    mock_client.delete.assert_not_called()


def test_redis_rpush_failure_returns_503_and_releases_idempotency_key() -> None:
    payload = _valid_payload()
    payload_bytes = json.dumps(payload).encode("utf-8")
    mock_client = MagicMock()
    mock_client.set.return_value = True
    mock_client.rpush.side_effect = RuntimeError("redis unavailable")

    result, mock_client = _invoke_webhook(payload_bytes, redis_client=mock_client)

    assert result.status_code == 503
    assert result.detail == "Service temporarily unavailable"
    mock_client.delete.assert_called_once_with(f"physio:seen:{payload['event_id']}")


def test_route_module_exports_hydration_queue_and_not_legacy_event_queue() -> None:
    payload_bytes = json.dumps(_valid_payload()).encode("utf-8")

    with (
        patch.dict(
            "os.environ",
            {
                "OURA_WEBHOOK_SECRET": "test-secret",
                "REDIS_URL": "redis://localhost:6379/0",
            },
            clear=False,
        ),
        patch.dict("sys.modules", {"redis": MagicMock()}),
    ):
        mod = _get_physiology_module()

    assert mod._PHYSIO_HYDRATE_QUEUE == "physio:hydrate"
    assert not hasattr(mod, "_PHYSIO_QUEUE_KEY")
    assert "PhysiologicalChunkEvent" not in mod.__dict__
    assert "physio:events" not in payload_bytes.decode("utf-8")
