"""Desktop SQLite FastAPI override wiring tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.experiments import ExperimentArmSeedRequest, ExperimentCreateRequest
from packages.schemas.operator_console import (
    CloudExperimentRefreshStatus,
    CloudOperatorErrorCode,
    ExperimentBundleRefreshResult,
)
from services.api.main import app as api_app
from services.api.routes.experiments import get_admin_service
from services.api.routes.operator import get_action_service, get_cloud_service, get_read_service
from services.api.routes.sessions import get_session_lifecycle_service
from services.api.services import operator_action_service as action_module
from services.api.services import session_lifecycle_service as lifecycle_module
from services.desktop_app.state.sqlite_api_overrides import (
    bootstrap_sqlite_api_store,
    configure_sqlite_api_overrides,
)
from services.desktop_app.state.sqlite_cloud_operator_service import SqliteCloudOperatorService
from services.desktop_app.state.sqlite_experiment_admin_service import (
    SqliteExperimentAdminService,
)
from services.desktop_app.state.sqlite_operator_action_service import (
    SqliteOperatorActionService,
)
from services.desktop_app.state.sqlite_operator_read_service import SqliteOperatorReadService
from services.desktop_app.state.sqlite_session_lifecycle_service import (
    SqliteSessionLifecycleService,
)

CLIENT_ACTION_ID = UUID("00000000-0000-4000-8000-0000000000a1")
STIMULUS_ACTION_ID = UUID("00000000-0000-4000-8000-0000000000b2")


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


def _now() -> datetime:
    return datetime(2026, 5, 12, 12, 0, tzinfo=UTC)


def test_configure_sqlite_api_overrides_installs_desktop_services(tmp_path: Path) -> None:
    app = FastAPI()

    services = configure_sqlite_api_overrides(app, tmp_path / "desktop.sqlite")

    assert isinstance(app.dependency_overrides[get_read_service](), SqliteOperatorReadService)
    assert isinstance(app.dependency_overrides[get_action_service](), SqliteOperatorActionService)
    assert isinstance(
        app.dependency_overrides[get_session_lifecycle_service](),
        SqliteSessionLifecycleService,
    )
    assert isinstance(app.dependency_overrides[get_admin_service](), SqliteExperimentAdminService)
    assert isinstance(app.dependency_overrides[get_cloud_service](), SqliteCloudOperatorService)
    assert services.read_service is app.dependency_overrides[get_read_service]()


@pytest.mark.parametrize("module", [lifecycle_module, action_module])
def test_configured_desktop_app_write_paths_do_not_use_server_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
) -> None:
    def fail_default_dependency(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("default server dependency reached")

    monkeypatch.setattr(module, "get_connection", fail_default_dependency)
    api_app.dependency_overrides.clear()
    original_lifespan = api_app.router.lifespan_context
    try:
        configure_sqlite_api_overrides(api_app, tmp_path / "desktop.sqlite")
        with TestClient(api_app) as client:
            session_response = client.post(
                "/api/v1/sessions",
                json={
                    "stream_url": "test://stream",
                    "experiment_id": "greeting_line_v1",
                    "client_action_id": str(CLIENT_ACTION_ID),
                },
            )
            assert session_response.status_code == 200
            session_id = session_response.json()["session_id"]

            stimulus_response = client.post(
                f"/api/v1/operator/sessions/{session_id}/stimulus",
                json={"client_action_id": str(STIMULUS_ACTION_ID)},
            )
            assert stimulus_response.status_code == 200
            message = stimulus_response.json()["message"]
            assert "response measurement is starting" in message.lower()
            assert "before sending another test message" in message.lower()

            experiment_response = client.post(
                "/api/v1/experiments",
                json={
                    "experiment_id": "desktop_exp",
                    "label": "Desktop experiment",
                    "arms": [
                        {
                            "arm": "warm",
                            "stimulus_definition": _stimulus_definition("Hei").model_dump(
                                mode="json"
                            ),
                        }
                    ],
                },
            )
            assert experiment_response.status_code == 201
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan


def test_configured_cloud_outbox_route_exposes_background_refresh_state(tmp_path: Path) -> None:
    from services.desktop_app.cloud.outbox import CloudOutbox

    db_path = tmp_path / "desktop.sqlite"
    outbox = CloudOutbox(db_path)
    try:
        outbox.record_experiment_refresh_result(
            ExperimentBundleRefreshResult(
                status=CloudExperimentRefreshStatus.FAILED,
                completed_at_utc=_now(),
                message="Cloud authorization was rejected.",
                error_code=CloudOperatorErrorCode.UNAUTHORIZED,
                retryable=False,
            )
        )
    finally:
        outbox.close()

    api_app.dependency_overrides.clear()
    original_lifespan = api_app.router.lifespan_context
    try:
        configure_sqlite_api_overrides(api_app, db_path)
        with TestClient(api_app) as client:
            response = client.get("/api/v1/operator/cloud/outbox")
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan

    assert response.status_code == 200
    payload = response.json()["latest_experiment_refresh"]
    assert payload["status"] == "failed"
    assert payload["error_code"] == "unauthorized"
    assert payload["retryable"] is False
    assert payload["message"] == "Cloud authorization was rejected."


def test_bootstrap_sqlite_api_store_migrates_upgraded_db_for_operator_overview(
    tmp_path: Path,
) -> None:
    import sqlite3

    db_path = tmp_path / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            stream_url TEXT NOT NULL,
            experiment_id TEXT,
            active_arm TEXT,
            stimulus_definition TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_session_state (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
            active_arm TEXT,
            is_calibrating INTEGER NOT NULL,
            calibration_frames_accumulated INTEGER NOT NULL,
            calibration_frames_required INTEGER NOT NULL,
            face_present INTEGER NOT NULL,
            latest_au12_intensity REAL,
            latest_au12_timestamp_s REAL,
            status TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, stream_url, experiment_id, active_arm,
            stimulus_definition, started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "00000000-0000-4000-8000-000000000001",
            "android://device",
            "greeting_line_v1",
            "warm_welcome",
            _stimulus_definition("Hello from upgraded db").model_dump_json(),
            "2026-05-12 12:00:00",
            None,
        ),
    )
    conn.close()

    bootstrap_sqlite_api_store(db_path)

    upgraded = sqlite3.connect(str(db_path), isolation_level=None)
    upgraded.execute(
        """
        INSERT INTO live_session_state (
            session_id, active_arm, stimulus_definition, is_calibrating,
            calibration_frames_accumulated, calibration_frames_required,
            face_present, status, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "00000000-0000-4000-8000-000000000001",
            "warm_welcome",
            _stimulus_definition("Hello from upgraded db").model_dump_json(),
            0,
            10,
            10,
            1,
            "ready",
            "2026-05-12 12:00:05",
        ),
    )
    upgraded.close()

    api_app.dependency_overrides.clear()
    original_lifespan = api_app.router.lifespan_context
    try:
        configure_sqlite_api_overrides(api_app, db_path)
        with TestClient(api_app) as client:
            response = client.get("/api/v1/operator/overview")
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_session"]["session_id"] == "00000000-0000-4000-8000-000000000001"
    assert (
        payload["active_session"]["expected_response_text"]
        == "The live streamer acknowledges the greeting"
    )


def test_configured_lifespan_skips_server_pool_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from services.api.db import connection

    def fail_init_pool(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("init_pool should not run in desktop lifespan")

    monkeypatch.setattr(connection, "init_pool", fail_init_pool)
    api_app.dependency_overrides.clear()
    original_lifespan = api_app.router.lifespan_context
    try:
        configure_sqlite_api_overrides(api_app, tmp_path / "desktop.sqlite")
        with TestClient(api_app) as client:
            response = client.post(
                "/api/v1/experiments",
                json=ExperimentCreateRequest(
                    experiment_id="desktop_exp",
                    label="Desktop experiment",
                    arms=[
                        ExperimentArmSeedRequest(
                            arm="warm",
                            stimulus_definition=_stimulus_definition("Hei"),
                        )
                    ],
                ).model_dump(mode="json"),
            )
            assert response.status_code == 201
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan
