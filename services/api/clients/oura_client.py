from __future__ import annotations

import json
import logging
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_OURA_API_BASE_URL = "https://api.ouraring.com"
_OURA_TOKEN_URL = "https://api.ouraring.com/oauth/token"
_REFRESH_SKEW_SECONDS = 60
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 0.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ALLOWED_TOKEN_FILE_MODE = 0o600
_DEFAULT_TOKEN_FILENAME = "oura_tokens.json"


class OuraClientConfigError(RuntimeError):
    """Raised when required Oura OAuth configuration is unavailable."""


class OuraTokenStoreError(RuntimeError):
    """Raised when Oura token persistence fails or is unsafe."""


class SupportsNow(Protocol):
    def __call__(self) -> datetime: ...


@dataclass(frozen=True)
class OuraTokenSet:
    access_token: str
    refresh_token: str
    expires_at_utc: datetime
    token_type: str = "Bearer"
    scope: str | None = None

    def expires_within(self, seconds: int, *, now: datetime) -> bool:
        return self.expires_at_utc <= now + timedelta(seconds=seconds)


class OuraTokenStore(Protocol):
    def load_tokens(self) -> OuraTokenSet | None: ...

    def save_tokens(self, tokens: OuraTokenSet) -> None: ...


class FileOuraTokenStore:
    """File-backed token store with strict unix permissions."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_tokens(self) -> OuraTokenSet | None:
        if not self._path.exists():
            return None
        self._validate_permissions()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return _tokens_from_mapping(payload)

    def save_tokens(self, tokens: OuraTokenSet) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        payload = json.dumps(_tokens_to_mapping(tokens), separators=(",", ":"))
        temp_path.write_text(payload, encoding="utf-8")
        os.chmod(temp_path, _ALLOWED_TOKEN_FILE_MODE)
        temp_path.replace(self._path)
        os.chmod(self._path, _ALLOWED_TOKEN_FILE_MODE)
        self._validate_permissions()

    def _validate_permissions(self) -> None:
        file_mode = stat.S_IMODE(self._path.stat().st_mode)
        if file_mode != _ALLOWED_TOKEN_FILE_MODE:
            raise OuraTokenStoreError(
                f"Unsafe Oura token file permissions: expected 0o600, got {oct(file_mode)}"
            )


class OuraAPIClient:
    """Injectable OAuth2 client for Oura API hydration work."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_store: OuraTokenStore | None = None,
        api_base_url: str = _OURA_API_BASE_URL,
        token_url: str = _OURA_TOKEN_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        now: SupportsNow = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._client_id = client_id if client_id is not None else os.environ.get("OURA_CLIENT_ID")
        self._client_secret = (
            client_secret if client_secret is not None else os.environ.get("OURA_CLIENT_SECRET")
        )
        self._token_store = (
            token_store if token_store is not None else default_token_store_from_env()
        )
        self._api_base_url = api_base_url.rstrip("/")
        self._token_url = token_url
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)
        self._now = now
        self._sleep = sleep
        self._urlopen = urlopen

    @property
    def is_configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._token_store is not None)

    def get_json(self, path: str, *, query: dict[str, Any] | None = None) -> dict[str, Any]:
        tokens = self._ensure_access_token()
        url = self._build_url(path, query=query)
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {tokens.access_token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        return self._send_json_request(request)

    def _build_url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self._api_base_url}{normalized_path}"
        if not query:
            return url
        encoded_query = urllib.parse.urlencode(
            {key: value for key, value in query.items() if value is not None}
        )
        return f"{url}?{encoded_query}" if encoded_query else url

    def _ensure_access_token(self) -> OuraTokenSet:
        if not self.is_configured:
            raise OuraClientConfigError("Oura OAuth client is not configured")
        assert self._token_store is not None
        tokens = self._token_store.load_tokens()
        if tokens is None:
            raise OuraClientConfigError("Oura OAuth tokens are not available")
        if tokens.expires_within(_REFRESH_SKEW_SECONDS, now=self._now()):
            tokens = self._refresh_tokens(tokens)
        return tokens

    def _refresh_tokens(self, tokens: OuraTokenSet) -> OuraTokenSet:
        if not self._client_id or not self._client_secret:
            raise OuraClientConfigError("Oura OAuth client is not configured")
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": tokens.refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._token_url,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        payload = self._send_json_request(request)
        refreshed = _tokens_from_oauth_payload(
            payload,
            previous_refresh_token=tokens.refresh_token,
            now=self._now(),
        )
        assert self._token_store is not None
        self._token_store.save_tokens(refreshed)
        logger.info("Refreshed Oura OAuth access token")
        return refreshed

    def _send_json_request(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._urlopen(request, timeout=self._timeout_seconds) as response:
                    raw_body = response.read()
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("Oura API response was not a JSON object")
                return payload
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS_CODES or attempt >= self._max_retries:
                    raise
                self._backoff(attempt, status_code=exc.code)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise
                self._backoff(attempt, status_code=None)
            except TimeoutError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise
                self._backoff(attempt, status_code=None)
        assert last_error is not None
        raise last_error

    def _backoff(self, attempt: int, *, status_code: int | None) -> None:
        delay = self._backoff_seconds * (2 ** (attempt - 1))
        logger.warning(
            "Retrying Oura API request after transient failure: attempt=%s status=%s delay_s=%.2f",
            attempt,
            status_code,
            delay,
        )
        self._sleep(delay)


def default_token_store_from_env() -> OuraTokenStore | None:
    token_store_kind = os.environ.get("OURA_TOKEN_STORE", "file").strip().lower()
    if token_store_kind and token_store_kind != "file":
        logger.warning(
            "Unsupported Oura token store %r configured; falling back to secure file store",
            token_store_kind,
        )
    token_path = os.environ.get("OURA_TOKEN_FILE")
    if token_path:
        return FileOuraTokenStore(token_path)
    repo_root = Path(__file__).resolve().parents[3]
    default_state_dir = repo_root / ".state"
    state_dir = os.environ.get("OURA_STATE_DIR") or os.environ.get("XDG_STATE_HOME")
    if state_dir:
        return FileOuraTokenStore(Path(state_dir) / _DEFAULT_TOKEN_FILENAME)
    return FileOuraTokenStore(default_state_dir / _DEFAULT_TOKEN_FILENAME)


def create_oura_client_from_env(**kwargs: Any) -> OuraAPIClient | None:
    client = OuraAPIClient(**kwargs)
    return client if client.is_configured else None


def _tokens_from_mapping(payload: dict[str, Any]) -> OuraTokenSet:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_at_utc = payload.get("expires_at_utc")
    token_type = payload.get("token_type") or "Bearer"
    scope = payload.get("scope")

    if not isinstance(access_token, str) or not access_token:
        raise OuraTokenStoreError("Stored Oura access token is missing")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OuraTokenStoreError("Stored Oura refresh token is missing")
    if isinstance(expires_at_utc, str):
        expires_at = datetime.fromisoformat(expires_at_utc.replace("Z", "+00:00"))
    elif isinstance(expires_at_utc, datetime):
        expires_at = expires_at_utc
    else:
        raise OuraTokenStoreError("Stored Oura token expiry is missing")
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    else:
        expires_at = expires_at.astimezone(UTC)
    return OuraTokenSet(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_utc=expires_at,
        token_type=str(token_type),
        scope=str(scope) if scope is not None else None,
    )


def _tokens_from_oauth_payload(
    payload: dict[str, Any],
    *,
    previous_refresh_token: str,
    now: datetime,
) -> OuraTokenSet:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token") or previous_refresh_token
    expires_in = payload.get("expires_in")
    token_type = payload.get("token_type") or "Bearer"
    scope = payload.get("scope")

    if not isinstance(access_token, str) or not access_token:
        raise OuraTokenStoreError("Oura OAuth response did not include access_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OuraTokenStoreError("Oura OAuth response did not include refresh_token")
    if expires_in is None:
        raise OuraTokenStoreError("Oura OAuth response did not include expires_in")

    expires_at = now + timedelta(seconds=int(expires_in))
    return OuraTokenSet(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_utc=expires_at.astimezone(UTC),
        token_type=str(token_type),
        scope=str(scope) if scope is not None else None,
    )


def _tokens_to_mapping(tokens: OuraTokenSet) -> dict[str, Any]:
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at_utc": tokens.expires_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "token_type": tokens.token_type,
        "scope": tokens.scope,
    }
