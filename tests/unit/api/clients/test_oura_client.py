from __future__ import annotations

import json
import os
import stat
import tempfile
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from services.api.clients.oura_client import (
    FileOuraTokenStore,
    OuraAPIClient,
    OuraClientConfigError,
    OuraTokenSet,
    OuraTokenStoreError,
    create_oura_client_from_env,
    default_token_store_from_env,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _MemoryTokenStore:
    def __init__(self, tokens: OuraTokenSet | None) -> None:
        self.tokens = tokens
        self.saved: list[OuraTokenSet] = []

    def load_tokens(self) -> OuraTokenSet | None:
        return self.tokens

    def save_tokens(self, tokens: OuraTokenSet) -> None:
        self.tokens = tokens
        self.saved.append(tokens)


def _token_set(*, expires_at: datetime, access_token: str = "access-1") -> OuraTokenSet:
    return OuraTokenSet(
        access_token=access_token,
        refresh_token="refresh-1",
        expires_at_utc=expires_at,
        token_type="Bearer",
        scope="daily",
    )


def _http_error(code: int, msg: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.ouraring.com/v2/usercollection/heartrate",
        code=code,
        msg=msg,
        hdrs=Message(),
        fp=None,
    )


def test_default_token_store_uses_repo_scoped_file_backing() -> None:
    old_env = os.environ.copy()
    try:
        os.environ.pop("OURA_TOKEN_STORE", None)
        os.environ.pop("OURA_TOKEN_FILE", None)
        os.environ.pop("OURA_STATE_DIR", None)
        os.environ.pop("XDG_STATE_HOME", None)

        store = default_token_store_from_env()
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    assert isinstance(store, FileOuraTokenStore)
    assert store.path == Path(__file__).resolve().parents[4] / ".state" / "oura_tokens.json"


def test_default_token_store_prefers_explicit_state_dir_when_set() -> None:
    old_env = os.environ.copy()
    try:
        os.environ.pop("OURA_TOKEN_STORE", None)
        os.environ.pop("OURA_TOKEN_FILE", None)
        os.environ["OURA_STATE_DIR"] = "/tmp/oura-state"

        store = default_token_store_from_env()
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    assert isinstance(store, FileOuraTokenStore)
    assert store.path == Path("/tmp/oura-state") / "oura_tokens.json"


def test_default_token_store_falls_back_to_file_when_postgres_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    old_env = os.environ.copy()
    try:
        os.environ["OURA_TOKEN_STORE"] = "postgres"
        os.environ["OURA_STATE_DIR"] = "/tmp/oura-state"

        store = default_token_store_from_env()
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    assert isinstance(store, FileOuraTokenStore)
    assert store.path == Path("/tmp/oura-state") / "oura_tokens.json"
    assert "Unsupported Oura token store 'postgres' configured" in caplog.text


def test_get_json_refreshes_when_token_expires_within_60_seconds() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    store = _MemoryTokenStore(_token_set(expires_at=now + timedelta(seconds=30)))
    requests: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        requests.append(request)
        if request.full_url.endswith("/oauth/token"):
            return _FakeResponse(
                {
                    "access_token": "access-2",
                    "refresh_token": "refresh-2",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "scope": "daily",
                }
            )
        return _FakeResponse({"data": [], "used_token": request.headers["Authorization"]})

    client = OuraAPIClient(
        client_id="client-id",
        client_secret="client-secret",
        token_store=store,
        now=lambda: now,
        urlopen=fake_urlopen,
    )

    result = client.get_json("/v2/usercollection/heartrate")

    assert result["used_token"] == "Bearer access-2"
    assert len(store.saved) == 1
    assert store.saved[0].refresh_token == "refresh-2"
    assert requests[0].full_url.endswith("/oauth/token")
    assert requests[1].full_url.endswith("/v2/usercollection/heartrate")


def test_get_json_uses_existing_token_when_not_near_expiry() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    store = _MemoryTokenStore(_token_set(expires_at=now + timedelta(minutes=10)))
    requests: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del timeout
        requests.append(request)
        return _FakeResponse({"ok": True, "auth": request.headers["Authorization"]})

    client = OuraAPIClient(
        client_id="client-id",
        client_secret="client-secret",
        token_store=store,
        now=lambda: now,
        urlopen=fake_urlopen,
    )

    result = client.get_json("/v2/usercollection/heartrate")

    assert result == {"ok": True, "auth": "Bearer access-1"}
    assert store.saved == []
    assert len(requests) == 1


def test_retries_provider_429_and_then_succeeds() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    store = _MemoryTokenStore(_token_set(expires_at=now + timedelta(minutes=10)))
    sleep_calls: list[float] = []
    attempts = {"count": 0}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del request, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _http_error(429, "rate limited")
        return _FakeResponse({"ok": True})

    client = OuraAPIClient(
        client_id="client-id",
        client_secret="client-secret",
        token_store=store,
        now=lambda: now,
        urlopen=fake_urlopen,
        sleep=sleep_calls.append,
        max_retries=3,
        backoff_seconds=0.25,
    )

    result = client.get_json("/v2/usercollection/heartrate")

    assert result == {"ok": True}
    assert attempts["count"] == 2
    assert sleep_calls == [0.25]


def test_gives_up_after_bounded_retries_on_5xx() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    store = _MemoryTokenStore(_token_set(expires_at=now + timedelta(minutes=10)))
    sleep_calls: list[float] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        del request, timeout
        raise _http_error(503, "unavailable")

    client = OuraAPIClient(
        client_id="client-id",
        client_secret="client-secret",
        token_store=store,
        now=lambda: now,
        urlopen=fake_urlopen,
        sleep=sleep_calls.append,
        max_retries=3,
        backoff_seconds=0.5,
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        client.get_json("/v2/usercollection/heartrate")

    assert exc_info.value.code == 503
    assert sleep_calls == [0.5, 1.0]


def test_file_token_store_enforces_0600_permissions() -> None:
    now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "oura_tokens.json"
        store = FileOuraTokenStore(path)
        store.save_tokens(_token_set(expires_at=now + timedelta(hours=1)))

        file_mode = stat.S_IMODE(path.stat().st_mode)
        assert file_mode == 0o600

        os.chmod(path, 0o644)
        with pytest.raises(OuraTokenStoreError, match="Unsafe Oura token file permissions"):
            store.load_tokens()


def test_missing_config_returns_none_factory_for_hydration_layer() -> None:
    old_env = os.environ.copy()
    try:
        os.environ.pop("OURA_CLIENT_ID", None)
        os.environ.pop("OURA_CLIENT_SECRET", None)
        os.environ["OURA_TOKEN_STORE"] = "file"
        os.environ["OURA_STATE_DIR"] = "/tmp/oura-state"
        os.environ.pop("OURA_TOKEN_FILE", None)
        client = create_oura_client_from_env()
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    assert client is None


def test_unconfigured_client_raises_safe_config_error_when_used() -> None:
    client = OuraAPIClient(client_id=None, client_secret=None, token_store=None)

    with pytest.raises(OuraClientConfigError, match="not configured"):
        client.get_json("/v2/usercollection/heartrate")
