"""Cloud sync worker process.

Drains the desktop cloud outbox into the cloud API without importing ML
runtime libraries. The process heartbeat starts before network work and
stops during cooperative shutdown so operator health freshness remains
visible even when the cloud path is unavailable.

ML import discipline: this module must not import torch, mediapipe,
faster_whisper, or ctranslate2.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing.synchronize as mpsync
import os
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import ValidationError

from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import (
    OAuthTokenRequest,
    OAuthTokenResponse,
    PosteriorDelta,
    TelemetryPosteriorDeltaBatch,
    TelemetrySegmentBatch,
)
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import (
    CloudOutbox,
    PendingUpload,
    RedactedPayloadError,
    UploadEndpoint,
)
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.privacy.secrets import (
    SECRET_KEY_CLOUD_REFRESH_TOKEN,
    SecretStoreUnavailableError,
    get_secret,
    set_secret,
)

logger = logging.getLogger(__name__)

SQLITE_FILENAME = "desktop.sqlite"
DEFAULT_CLOUD_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SYNC_INTERVAL_S = 5.0
DEFAULT_BATCH_SIZE = 50
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_CLIENT_ID = "desktop-app"

_ENDPOINT_PATHS: dict[UploadEndpoint, str] = {
    "telemetry_segments": "/v4/telemetry/segments",
    "telemetry_posterior_deltas": "/v4/telemetry/posterior_deltas",
}


class ShutdownEvent(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True)
class CloudSyncConfig:
    base_url: str = DEFAULT_CLOUD_BASE_URL
    interval_s: float = DEFAULT_SYNC_INTERVAL_S
    batch_size: int = DEFAULT_BATCH_SIZE
    timeout_s: float = DEFAULT_TIMEOUT_S
    client_id: str = DEFAULT_CLIENT_ID

    @classmethod
    def from_env(cls) -> CloudSyncConfig:
        return cls(
            base_url=os.environ.get("LSIE_CLOUD_BASE_URL", DEFAULT_CLOUD_BASE_URL).rstrip("/"),
            interval_s=_env_float("LSIE_CLOUD_SYNC_INTERVAL_S", DEFAULT_SYNC_INTERVAL_S),
            batch_size=_env_int("LSIE_CLOUD_SYNC_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            timeout_s=_env_float("LSIE_CLOUD_SYNC_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            client_id=os.environ.get("LSIE_CLOUD_CLIENT_ID", DEFAULT_CLIENT_ID),
        )


class CloudSyncWorker:
    def __init__(self, outbox: CloudOutbox, config: CloudSyncConfig) -> None:
        self._outbox = outbox
        self._config = config
        self._access_token: str | None = None

    async def run_until_shutdown(self, shutdown_event: ShutdownEvent) -> None:
        async with httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        ) as client:
            while not shutdown_event.is_set():
                await self.sync_once(client)
                await asyncio.to_thread(shutdown_event.wait, self._config.interval_s)

    async def sync_once(self, client: httpx.AsyncClient) -> None:
        self._outbox.reset_stale_locks()
        self._outbox.apply_retention_policy()
        for endpoint in _ENDPOINT_PATHS:
            while True:
                uploads = self._outbox.fetch_ready_batch(endpoint, limit=self._config.batch_size)
                if not uploads:
                    break
                await self._sync_batch(client, endpoint, uploads)

    async def _sync_batch(
        self,
        client: httpx.AsyncClient,
        endpoint: UploadEndpoint,
        uploads: list[PendingUpload],
    ) -> None:
        upload_ids = [upload.upload_id for upload in uploads]
        try:
            batch = self._build_batch(endpoint, uploads)
        except ValidationError:
            return
        except ValueError as exc:
            self._outbox.mark_dead_letter(upload_ids, error=str(exc))
            return

        if batch is None:
            return

        payload, valid_upload_ids = batch
        try:
            response = await self._post_with_auth(client, _ENDPOINT_PATHS[endpoint], payload)
        except httpx.TimeoutException:
            self._outbox.mark_retry(valid_upload_ids, error="cloud sync timeout")
            return
        except httpx.NetworkError as exc:
            self._outbox.mark_retry(valid_upload_ids, error=type(exc).__name__)
            return
        except CloudAuthRefreshError as exc:
            self._outbox.mark_retry(valid_upload_ids, error=str(exc))
            return

        status_code = response.status_code
        if 200 <= status_code < 300:
            self._outbox.delete_uploads(valid_upload_ids)
        elif status_code == 429 or status_code >= 500:
            self._outbox.mark_retry(valid_upload_ids, error=f"HTTP {status_code}")
        elif status_code == 401:
            self._outbox.mark_retry(valid_upload_ids, error="HTTP 401 after token refresh")
        elif 400 <= status_code < 500:
            self._outbox.mark_dead_letter(valid_upload_ids, error=f"HTTP {status_code}")
        else:
            self._outbox.mark_retry(valid_upload_ids, error=f"HTTP {status_code}")

    def _build_batch(
        self,
        endpoint: UploadEndpoint,
        uploads: list[PendingUpload],
    ) -> tuple[dict[str, object], list[str]] | None:
        if endpoint == "telemetry_segments":
            segments: list[InferenceHandoffPayload] = []
            attribution_events: list[AttributionEvent] = []
            valid_upload_ids: list[str] = []
            for upload in uploads:
                try:
                    model = self._outbox.validate_upload(upload)
                except (RedactedPayloadError, ValidationError):
                    continue
                if isinstance(model, InferenceHandoffPayload):
                    segments.append(model)
                elif isinstance(model, AttributionEvent):
                    attribution_events.append(model)
                else:
                    raise ValueError(f"unsupported payload_type {upload.payload_type!r}")
                valid_upload_ids.append(upload.upload_id)
            if not valid_upload_ids:
                return None
            segment_batch = TelemetrySegmentBatch(
                segments=segments,
                attribution_events=attribution_events,
            )
            return segment_batch.model_dump(mode="json", by_alias=True), valid_upload_ids

        deltas: list[PosteriorDelta] = []
        valid_upload_ids = []
        for upload in uploads:
            try:
                model = self._outbox.validate_upload(upload)
            except (RedactedPayloadError, ValidationError):
                continue
            if not isinstance(model, PosteriorDelta):
                raise ValueError(f"unsupported payload_type {upload.payload_type!r}")
            deltas.append(model)
            valid_upload_ids.append(upload.upload_id)
        if not deltas:
            return None
        delta_batch = TelemetryPosteriorDeltaBatch(deltas=deltas)
        return delta_batch.model_dump(mode="json", by_alias=True), valid_upload_ids

    async def _post_with_auth(
        self,
        client: httpx.AsyncClient,
        path: str,
        payload: dict[str, object],
    ) -> httpx.Response:
        access_token = await self._access_token_or_refresh(client)
        response = await client.post(path, json=payload, headers=_auth_headers(access_token))
        if response.status_code != 401:
            return response
        self._access_token = None
        access_token = await self._refresh_access_token(client)
        return await client.post(path, json=payload, headers=_auth_headers(access_token))

    async def _access_token_or_refresh(self, client: httpx.AsyncClient) -> str:
        if self._access_token is None:
            return await self._refresh_access_token(client)
        return self._access_token

    async def _refresh_access_token(self, client: httpx.AsyncClient) -> str:
        try:
            refresh_token = get_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN)
        except SecretStoreUnavailableError as exc:
            raise CloudAuthRefreshError("cloud secret store unavailable") from exc
        if refresh_token is None:
            raise CloudAuthRefreshError("cloud refresh token unavailable")
        request = OAuthTokenRequest(
            grant_type="refresh_token",
            client_id=self._config.client_id,
            refresh_token=refresh_token,
        )
        try:
            response = await client.post(
                "/v4/auth/oauth/token",
                json=request.model_dump(mode="json", exclude_none=True),
            )
            response.raise_for_status()
            token_response = OAuthTokenResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise CloudAuthRefreshError("cloud access token refresh failed") from exc
        if token_response.refresh_token is not None:
            try:
                set_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN, token_response.refresh_token)
            except SecretStoreUnavailableError as exc:
                raise CloudAuthRefreshError("cloud refresh token storage failed") from exc
        self._access_token = token_response.access_token
        return token_response.access_token


class CloudAuthRefreshError(RuntimeError):
    pass


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels
    logger.info("cloud_sync_worker started")

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder

    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    heartbeat = HeartbeatRecorder(db_path, "cloud_sync_worker")
    heartbeat.start()
    outbox = CloudOutbox(db_path)

    try:
        asyncio.run(
            CloudSyncWorker(outbox, CloudSyncConfig.from_env()).run_until_shutdown(shutdown_event)
        )
    finally:
        outbox.close()
        heartbeat.stop()
        logger.info("cloud_sync_worker stopped")


def _auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
