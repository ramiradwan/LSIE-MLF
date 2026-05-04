"""Desktop SQLite FastAPI override wiring tests."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.schemas.experiments import ExperimentArmSeedRequest, ExperimentCreateRequest
from services.api.main import app as api_app
from services.api.routes.experiments import get_admin_service
from services.api.routes.operator import get_action_service, get_read_service
from services.api.routes.sessions import get_session_lifecycle_service
from services.api.services import operator_action_service as action_module
from services.api.services import session_lifecycle_service as lifecycle_module
from services.desktop_app.state.sqlite_api_overrides import configure_sqlite_api_overrides
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
                    "arms": [{"arm": "warm", "greeting_text": "Hei"}],
                },
            )
            assert experiment_response.status_code == 201
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan


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
                    arms=[ExperimentArmSeedRequest(arm="warm", greeting_text="Hei")],
                ).model_dump(mode="json"),
            )
            assert response.status_code == 201
    finally:
        api_app.dependency_overrides.clear()
        api_app.router.lifespan_context = original_lifespan
