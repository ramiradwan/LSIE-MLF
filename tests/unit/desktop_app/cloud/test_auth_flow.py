from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    OAuthTokenRequest,
    OAuthTokenResponse,
)
from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.operator_console import (
    CloudActionStatus,
    CloudAuthState,
    CloudExperimentRefreshStatus,
    CloudOperatorErrorCode,
    ExperimentBundleRefreshRequest,
    ExperimentBundleRefreshResult,
)
from services.desktop_app.cloud.auth_flow import (
    AuthFlowConfig,
    DesktopAuthFlow,
    build_code_challenge,
)
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.privacy.secrets import SECRET_KEY_CLOUD_REFRESH_TOKEN
from services.desktop_app.processes.cloud_sync_worker import CloudSyncConfig
from services.desktop_app.state.sqlite_cloud_operator_service import SqliteCloudOperatorService
from services.desktop_app.state.sqlite_schema import bootstrap_schema


def _config() -> AuthFlowConfig:
    return AuthFlowConfig(
        authorization_endpoint="https://cloud.example.test/oauth/authorize",
        token_endpoint="https://cloud.example.test/oauth/token",
        client_id="desktop-a",
        redirect_uri="lsie://oauth/callback",
    )


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text=text,
        ),
        expected_stimulus_rule="Deliver the spoken greeting to the creator",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


def _bundle() -> ExperimentBundle:
    return ExperimentBundle(
        bundle_id="bundle-a",
        issued_at_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        expires_at_utc=datetime(2036, 5, 3, 12, 0, tzinfo=UTC),
        policy_version="v4.0",
        experiments=[
            ExperimentBundleExperiment(
                experiment_id="experiment-a",
                label="Experiment A",
                arms=[
                    ExperimentBundleArm(
                        arm_id="arm-a",
                        stimulus_definition=_stimulus_definition("Hello A"),
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                        selection_count=5,
                    )
                ],
            )
        ],
        signature="signature-a",
    )


def _latest_refresh(db_path: Path) -> ExperimentBundleRefreshResult | None:
    outbox = CloudOutbox(db_path)
    try:
        return outbox.latest_experiment_refresh()
    finally:
        outbox.close()


def test_pkce_challenge_matches_s256_reference_vector() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    assert build_code_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_open_authorization_uses_injected_browser_opener() -> None:
    opened_urls: list[str] = []

    def open_url(url: str) -> bool:
        opened_urls.append(url)
        return True

    flow = DesktopAuthFlow(_config(), browser_opener=open_url)

    request = flow.open_authorization(state="state-a")

    assert opened_urls == [request.authorization_url]
    assert "code_challenge_method=S256" in request.authorization_url
    assert "state=state-a" in request.authorization_url
    assert request.code_verifier


def test_loopback_authorization_validates_state_and_exchanges_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}
    opened_urls: list[str] = []
    seen_requests: list[OAuthTokenRequest] = []

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        seen_requests.append(request)
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
        )

    def open_callback(url: str) -> bool:
        opened_urls.append(url)
        redirect_uri = str(httpx.URL(url).params["redirect_uri"])
        state = str(httpx.URL(url).params["state"])

        def send_callback() -> None:
            response = httpx.get(redirect_uri, params={"code": "code-a", "state": state})
            response.raise_for_status()

        threading.Thread(target=send_callback).start()
        return True

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(
        _config(),
        browser_opener=open_callback,
        token_exchange=exchange,
        state_generator=lambda: "state-a",
    )

    response = flow.run_loopback_authorization()

    assert response.access_token == "access-a"
    assert opened_urls
    assert seen_requests == [
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code="code-a",
            code_verifier=seen_requests[0].code_verifier,
            redirect_uri=seen_requests[0].redirect_uri,
        )
    ]
    assert seen_requests[0].code_verifier
    redirect_uri = cast(str, seen_requests[0].redirect_uri)
    assert redirect_uri.startswith("http://127.0.0.1:")
    assert stored == {SECRET_KEY_CLOUD_REFRESH_TOKEN: "refresh-a"}


def test_authorization_code_exchange_uses_schema_and_stores_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}
    seen_requests: list[OAuthTokenRequest] = []

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        seen_requests.append(request)
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
        )

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(_config(), token_exchange=exchange)

    response = flow.exchange_authorization_code(code="code-a", code_verifier="verifier-a")

    assert response.access_token == "access-a"
    assert seen_requests == [
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code="code-a",
            code_verifier="verifier-a",
            redirect_uri="lsie://oauth/callback",
        )
    ]
    assert stored == {SECRET_KEY_CLOUD_REFRESH_TOKEN: "refresh-a"}


def test_refresh_exchange_does_not_overwrite_secret_when_response_omits_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        assert request.grant_type == "refresh_token"
        assert request.refresh_token == "refresh-a"
        return OAuthTokenResponse(access_token="access-b", expires_in=3600)

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(_config(), token_exchange=exchange)

    response = flow.refresh_access_token(refresh_token="refresh-a")

    assert response.access_token == "access-b"
    assert stored == {}


def test_sqlite_cloud_operator_auth_status_maps_secret_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from services.desktop_app.privacy.secrets import SecretStoreUnavailableError

    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: (_ for _ in ()).throw(SecretStoreUnavailableError("backend missing")),
    )
    service = SqliteCloudOperatorService(tmp_path / "desktop.sqlite")

    status = service.get_auth_status()

    assert status.state is CloudAuthState.SECRET_STORE_UNAVAILABLE
    assert status.retryable is True
    assert "backend missing" not in (status.message or "")


def test_sqlite_cloud_operator_sign_in_returns_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from services.desktop_app.cloud.auth_flow import OAuthCallbackError

    class _Flow:
        def __init__(self, _config: object) -> None:
            return None

        def run_loopback_authorization(self) -> object:
            raise OAuthCallbackError("raw oauth failure with secret details")

    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.DesktopAuthFlow",
        _Flow,
    )
    service = SqliteCloudOperatorService(tmp_path / "desktop.sqlite")

    result = service.sign_in()

    assert result.error_code is CloudOperatorErrorCode.AUTHORIZATION_FAILED
    assert result.message == "Cloud authorization failed."
    assert "secret details" not in result.message


def test_sqlite_cloud_operator_reports_signed_out_without_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: None,
    )
    service = SqliteCloudOperatorService(tmp_path / "desktop.sqlite")

    status = service.get_auth_status()

    assert status.state is CloudAuthState.SIGNED_OUT
    assert status.message == "Cloud sign-in is required."


def test_sqlite_cloud_operator_outbox_summary_resets_stale_inflight_uploads(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "desktop.sqlite"
    outbox = CloudOutbox(db_path)
    try:
        outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json='{"posterior_delta_id":"not-a-real-payload"}',
        )
        outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)
    finally:
        outbox.close()
    service = SqliteCloudOperatorService(db_path)

    summary = service.get_outbox_summary()

    assert summary.in_flight_count == 0
    assert summary.pending_count == 1


def test_sqlite_cloud_operator_reports_refresh_token_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: "   ",
    )
    db_path = tmp_path / "desktop.sqlite"
    service = SqliteCloudOperatorService(db_path)

    status = service.get_auth_status()
    refresh = service.refresh_experiment_bundle(
        ExperimentBundleRefreshRequest(preview_token="preview-token-a")
    )
    latest = _latest_refresh(db_path)

    assert status.state is CloudAuthState.REFRESH_TOKEN_UNAVAILABLE
    assert refresh.error_code is CloudOperatorErrorCode.REFRESH_TOKEN_UNAVAILABLE
    assert latest is not None
    assert latest.error_code is CloudOperatorErrorCode.REFRESH_TOKEN_UNAVAILABLE


def test_sqlite_cloud_operator_preview_fetches_without_recording_or_mutating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Flow:
        def __init__(self, _config: object) -> None:
            return None

        def refresh_access_token(self, *, refresh_token: str) -> OAuthTokenResponse:
            assert refresh_token == "refresh-a"
            return OAuthTokenResponse(access_token="access-a", expires_in=3600)

    class _Client:
        def __init__(self, base_url: str, *, timeout_s: float) -> None:
            assert base_url == "https://cloud.example.test"
            assert timeout_s == 15.0

        def fetch_bundle(self, *, access_token: str) -> ExperimentBundle:
            assert access_token == "access-a"
            return _bundle()

    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: "refresh-a",
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.DesktopAuthFlow",
        _Flow,
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.ExperimentBundleClient",
        _Client,
    )
    monkeypatch.setattr(
        "services.desktop_app.cloud.experiment_bundle.verify_bundle",
        lambda _bundle, *, config=None: None,
    )
    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    bootstrap_schema(conn, seed_experiments=False)
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "experiment-a",
            "Experiment A",
            "arm-a",
            _stimulus_definition("Old A").model_dump_json(),
            11.0,
            12.0,
            1,
        ),
    )
    conn.close()
    service = SqliteCloudOperatorService(
        db_path,
        config=CloudSyncConfig(base_url="https://cloud.example.test"),
    )

    preview = service.preview_experiment_bundle_refresh()
    latest = _latest_refresh(db_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    row = conn.execute(
        """
        SELECT stimulus_definition, alpha_param, beta_param, enabled
        FROM experiments
        WHERE experiment_id = 'experiment-a' AND arm = 'arm-a'
        """
    ).fetchone()
    conn.close()

    assert preview.status is CloudActionStatus.SUCCEEDED
    assert preview.preview_token is not None
    assert preview.updated_count == 1
    assert preview.existing_preserved_count == 1
    assert latest is None
    assert row is not None
    assert json.loads(row[0]) == _stimulus_definition("Old A").model_dump(mode="json")
    assert row[1:] == (11.0, 12.0, 1)


def test_sqlite_cloud_operator_rejects_apply_when_bundle_changes_after_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Flow:
        def __init__(self, _config: object) -> None:
            return None

        def refresh_access_token(self, *, refresh_token: str) -> OAuthTokenResponse:
            assert refresh_token == "refresh-a"
            return OAuthTokenResponse(access_token="access-a", expires_in=3600)

    stale_bundle = _bundle()
    changed_bundle = _bundle().model_copy(
        update={
            "experiments": [
                ExperimentBundleExperiment(
                    experiment_id="experiment-a",
                    label="Experiment A",
                    arms=[
                        ExperimentBundleArm(
                            arm_id="arm-a",
                            stimulus_definition=_stimulus_definition("Changed A"),
                            posterior_alpha=2.0,
                            posterior_beta=3.0,
                            selection_count=5,
                        )
                    ],
                )
            ],
        }
    )
    bundles = [stale_bundle, changed_bundle]

    class _Client:
        def __init__(self, base_url: str, *, timeout_s: float) -> None:
            assert base_url == "https://cloud.example.test"
            assert timeout_s == 15.0

        def fetch_bundle(self, *, access_token: str) -> ExperimentBundle:
            assert access_token == "access-a"
            return bundles.pop(0)

    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: "refresh-a",
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.DesktopAuthFlow",
        _Flow,
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.ExperimentBundleClient",
        _Client,
    )
    monkeypatch.setattr(
        "services.desktop_app.cloud.experiment_bundle.verify_bundle",
        lambda _bundle, *, config=None: None,
    )
    db_path = tmp_path / "desktop.sqlite"
    service = SqliteCloudOperatorService(
        db_path,
        config=CloudSyncConfig(base_url="https://cloud.example.test"),
    )

    preview = service.preview_experiment_bundle_refresh()
    assert preview.preview_token is not None
    result = service.refresh_experiment_bundle(
        ExperimentBundleRefreshRequest(preview_token=preview.preview_token)
    )
    latest = _latest_refresh(db_path)

    assert result.status is CloudExperimentRefreshStatus.FAILED
    assert result.error_code is CloudOperatorErrorCode.BUNDLE_CHANGED
    assert latest is not None
    assert latest.error_code is CloudOperatorErrorCode.BUNDLE_CHANGED


def test_sqlite_cloud_operator_records_successful_manual_experiment_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Flow:
        def __init__(self, _config: object) -> None:
            return None

        def refresh_access_token(self, *, refresh_token: str) -> OAuthTokenResponse:
            assert refresh_token == "refresh-a"
            return OAuthTokenResponse(access_token="access-a", expires_in=3600)

    class _Client:
        def __init__(self, base_url: str, *, timeout_s: float) -> None:
            assert base_url == "https://cloud.example.test"
            assert timeout_s == 15.0

        def fetch_bundle(self, *, access_token: str) -> ExperimentBundle:
            assert access_token == "access-a"
            return _bundle()

    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.get_secret",
        lambda _key: "refresh-a",
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.DesktopAuthFlow",
        _Flow,
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_cloud_operator_service.ExperimentBundleClient",
        _Client,
    )
    monkeypatch.setattr(
        "services.desktop_app.cloud.experiment_bundle.verify_bundle",
        lambda _bundle, *, config=None: None,
    )
    db_path = tmp_path / "desktop.sqlite"
    service = SqliteCloudOperatorService(
        db_path,
        config=CloudSyncConfig(base_url="https://cloud.example.test"),
    )

    preview = service.preview_experiment_bundle_refresh()
    assert preview.preview_token is not None

    result = service.refresh_experiment_bundle(
        ExperimentBundleRefreshRequest(preview_token=preview.preview_token)
    )
    latest = _latest_refresh(db_path)

    assert result.status is CloudExperimentRefreshStatus.APPLIED
    assert result.bundle_id == "bundle-a"
    assert latest is not None
    assert latest.status is CloudExperimentRefreshStatus.APPLIED
    assert latest.bundle_id == "bundle-a"
    assert latest.experiment_count == 1
