from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from packages.schemas.operator_console import (
    CloudActionStatus,
    CloudAuthState,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOperatorErrorCode,
    CloudOutboxSummary,
    CloudSignInResult,
    ExperimentBundleRefreshResult,
)
from services.desktop_app.cloud.auth_flow import (
    AuthFlowConfig,
    DesktopAuthFlow,
    OAuthCallbackError,
    OAuthTokenExchangeError,
)
from services.desktop_app.cloud.experiment_bundle import (
    ExperimentBundleClient,
    ExperimentBundleFetchError,
    ExperimentBundleStore,
    ExperimentBundleVerificationError,
)
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.privacy.secrets import (
    SECRET_KEY_CLOUD_REFRESH_TOKEN,
    SecretStoreUnavailableError,
    get_secret,
)
from services.desktop_app.processes.cloud_sync_worker import CloudSyncConfig


class SqliteCloudOperatorService:
    def __init__(self, db_path: Path, *, config: CloudSyncConfig | None = None) -> None:
        self._db_path = db_path
        self._config = config or CloudSyncConfig.from_env()

    def get_auth_status(self) -> CloudAuthStatus:
        now = datetime.now(UTC)
        try:
            refresh_token = get_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN)
        except SecretStoreUnavailableError:
            return CloudAuthStatus(
                state=CloudAuthState.SECRET_STORE_UNAVAILABLE,
                checked_at_utc=now,
                message="Cloud sign-in is temporarily unavailable.",
                retryable=True,
            )
        if refresh_token is None:
            return CloudAuthStatus(
                state=CloudAuthState.SIGNED_OUT,
                checked_at_utc=now,
                message="Cloud sign-in is required.",
            )
        if not refresh_token.strip():
            return CloudAuthStatus(
                state=CloudAuthState.REFRESH_TOKEN_UNAVAILABLE,
                checked_at_utc=now,
                message="Cloud refresh token is unavailable.",
            )
        flow = DesktopAuthFlow(_auth_config(self._config))
        try:
            flow.refresh_access_token(refresh_token=refresh_token)
        except SecretStoreUnavailableError:
            return CloudAuthStatus(
                state=CloudAuthState.SECRET_STORE_UNAVAILABLE,
                checked_at_utc=now,
                message="Cloud sign-in is temporarily unavailable.",
                retryable=True,
            )
        except (OAuthTokenExchangeError, ValidationError):
            return CloudAuthStatus(
                state=CloudAuthState.REFRESH_FAILED,
                checked_at_utc=now,
                message="Cloud sign-in could not be refreshed.",
                retryable=True,
            )
        return CloudAuthStatus(
            state=CloudAuthState.SIGNED_IN,
            checked_at_utc=now,
            message="Cloud sign-in is active.",
        )

    def sign_in(self) -> CloudSignInResult:
        now = datetime.now(UTC)
        try:
            DesktopAuthFlow(_auth_config(self._config)).run_loopback_authorization()
        except SecretStoreUnavailableError:
            return _sign_in_failure(
                CloudAuthState.SECRET_STORE_UNAVAILABLE,
                CloudOperatorErrorCode.SECRET_STORE_UNAVAILABLE,
                "Cloud sign-in is temporarily unavailable.",
                now,
                retryable=True,
            )
        except (OAuthCallbackError, OAuthTokenExchangeError, ValidationError):
            return _sign_in_failure(
                CloudAuthState.REFRESH_FAILED,
                CloudOperatorErrorCode.AUTHORIZATION_FAILED,
                "Cloud authorization failed.",
                now,
                retryable=True,
            )
        return CloudSignInResult(
            status=CloudActionStatus.SUCCEEDED,
            auth_state=CloudAuthState.SIGNED_IN,
            completed_at_utc=now,
            message="Cloud sign-in completed.",
        )

    def get_outbox_summary(self) -> CloudOutboxSummary:
        outbox = CloudOutbox(self._db_path)
        try:
            return outbox.summarize()
        finally:
            outbox.close()

    def get_latest_experiment_refresh(self) -> ExperimentBundleRefreshResult | None:
        outbox = CloudOutbox(self._db_path)
        try:
            return outbox.latest_experiment_refresh()
        finally:
            outbox.close()

    def refresh_experiment_bundle(self) -> ExperimentBundleRefreshResult:
        now = datetime.now(UTC)
        try:
            refresh_token = get_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN)
        except SecretStoreUnavailableError:
            return _refresh_failure(
                CloudOperatorErrorCode.SECRET_STORE_UNAVAILABLE,
                "Cloud sign-in is temporarily unavailable.",
                now,
                retryable=True,
            )
        if refresh_token is None:
            return _refresh_failure(
                CloudOperatorErrorCode.REFRESH_TOKEN_UNAVAILABLE,
                "Cloud sign-in is required before refreshing experiments.",
                now,
            )
        if not refresh_token.strip():
            return _refresh_failure(
                CloudOperatorErrorCode.REFRESH_TOKEN_UNAVAILABLE,
                "Cloud refresh token is unavailable.",
                now,
            )
        try:
            token = DesktopAuthFlow(_auth_config(self._config)).refresh_access_token(
                refresh_token=refresh_token
            )
        except SecretStoreUnavailableError:
            return _refresh_failure(
                CloudOperatorErrorCode.SECRET_STORE_UNAVAILABLE,
                "Cloud sign-in is temporarily unavailable.",
                now,
                retryable=True,
            )
        except (OAuthTokenExchangeError, ValidationError):
            return _refresh_failure(
                CloudOperatorErrorCode.REFRESH_FAILED,
                "Cloud sign-in could not be refreshed.",
                now,
                retryable=True,
            )
        try:
            bundle = ExperimentBundleClient(
                self._config.base_url,
                timeout_s=self._config.timeout_s,
            ).fetch_bundle(access_token=token.access_token)
        except ExperimentBundleFetchError as exc:
            error_code, retryable = _fetch_error(exc)
            return _refresh_failure(
                error_code,
                _refresh_error_message(error_code),
                now,
                retryable=retryable,
            )
        store = ExperimentBundleStore(self._db_path)
        try:
            store.cache_verified_bundle(bundle)
        except ExperimentBundleVerificationError as exc:
            error_code = _verification_error(exc)
            return _refresh_failure(error_code, _refresh_error_message(error_code), now)
        finally:
            store.close()
        return ExperimentBundleRefreshResult(
            status=CloudExperimentRefreshStatus.APPLIED,
            completed_at_utc=now,
            message="Experiment bundle refreshed.",
            bundle_id=bundle.bundle_id,
            experiment_count=len(bundle.experiments),
        )


def _auth_config(config: CloudSyncConfig) -> AuthFlowConfig:
    return AuthFlowConfig(
        authorization_endpoint=f"{config.base_url}/v4/auth/oauth/authorize",
        token_endpoint=f"{config.base_url}/v4/auth/oauth/token",
        client_id=config.client_id,
    )


def _sign_in_failure(
    state: CloudAuthState,
    error_code: CloudOperatorErrorCode,
    message: str,
    completed_at_utc: datetime,
    *,
    retryable: bool = False,
) -> CloudSignInResult:
    return CloudSignInResult(
        status=CloudActionStatus.FAILED,
        auth_state=state,
        completed_at_utc=completed_at_utc,
        message=message,
        error_code=error_code,
        retryable=retryable,
    )


def _refresh_failure(
    error_code: CloudOperatorErrorCode,
    message: str,
    completed_at_utc: datetime,
    *,
    retryable: bool = False,
) -> ExperimentBundleRefreshResult:
    return ExperimentBundleRefreshResult(
        status=CloudExperimentRefreshStatus.FAILED,
        completed_at_utc=completed_at_utc,
        message=message,
        error_code=error_code,
        retryable=retryable,
    )


def _fetch_error(exc: ExperimentBundleFetchError) -> tuple[CloudOperatorErrorCode, bool]:
    status_code = exc.status_code
    if status_code == 401:
        return CloudOperatorErrorCode.UNAUTHORIZED, False
    if status_code == 429:
        return CloudOperatorErrorCode.RATE_LIMITED, True
    if status_code is not None and status_code >= 500:
        return CloudOperatorErrorCode.CLOUD_UNAVAILABLE, True
    if isinstance(exc.__cause__, httpx.NetworkError | httpx.TimeoutException):
        return CloudOperatorErrorCode.OFFLINE, True
    if status_code is None:
        return CloudOperatorErrorCode.INVALID_RESPONSE, False
    return CloudOperatorErrorCode.CLOUD_UNAVAILABLE, True


def _verification_error(exc: ExperimentBundleVerificationError) -> CloudOperatorErrorCode:
    message = str(exc).lower()
    if "validity" in message:
        return CloudOperatorErrorCode.BUNDLE_EXPIRED
    return CloudOperatorErrorCode.SIGNATURE_FAILED


def _refresh_error_message(error_code: CloudOperatorErrorCode) -> str:
    messages = {
        CloudOperatorErrorCode.BUNDLE_EXPIRED: "Experiment bundle is outside its validity window.",
        CloudOperatorErrorCode.CLOUD_UNAVAILABLE: "Cloud experiment service is unavailable.",
        CloudOperatorErrorCode.INVALID_RESPONSE: "Cloud experiment bundle response was invalid.",
        CloudOperatorErrorCode.OFFLINE: "Cloud experiment service is offline.",
        CloudOperatorErrorCode.RATE_LIMITED: "Cloud experiment refresh is rate-limited.",
        CloudOperatorErrorCode.SIGNATURE_FAILED: "Experiment bundle signature verification failed.",
        CloudOperatorErrorCode.UNAUTHORIZED: "Cloud authorization was rejected.",
    }
    return messages.get(error_code, "Experiment bundle refresh failed.")


__all__ = ["SqliteCloudOperatorService"]
