"""FastAPI dependency overrides for the SQLite-backed desktop API shell.

``ui_api_shell`` reuses retained route definitions but not their default
PostgreSQL/Redis dependencies. This module is the handoff point: read
routes receive ``SqliteReader``-backed services, operator/session write
routes receive SQLite service methods that publish IPC live-session control
messages, and the server lifespan is replaced so the desktop loopback API
does not open a server Persistent Store pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI

from services.api.routes.experiments import get_admin_service
from services.api.routes.operator import get_action_service, get_event_service, get_read_service
from services.api.routes.sessions import get_session_lifecycle_service
from services.api.services.operator_event_service import OperatorEventService
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_experiment_admin_service import (
    SqliteExperimentAdminService,
)
from services.desktop_app.state.sqlite_operator_action_service import (
    SqliteOperatorActionService,
)
from services.desktop_app.state.sqlite_operator_read_service import SqliteOperatorReadService
from services.desktop_app.state.sqlite_reader import SqliteReader
from services.desktop_app.state.sqlite_schema import bootstrap_schema
from services.desktop_app.state.sqlite_session_lifecycle_service import (
    SqliteSessionLifecycleService,
)


class LiveSessionControlPublisher(Protocol):
    def publish(self, message: LiveSessionControlMessage) -> None: ...


class DesktopApiServices:
    """Holds desktop API service singletons for dependency override tests."""

    def __init__(
        self,
        db_path: Path,
        *,
        control_publisher: LiveSessionControlPublisher | None = None,
    ) -> None:
        self.reader = SqliteReader(db_path)
        self.read_service = SqliteOperatorReadService(self.reader)
        self.event_service = OperatorEventService(
            read_service=self.read_service,
            marker_provider=self.read_service.operator_event_markers,
        )
        self.action_service = SqliteOperatorActionService(
            db_path,
            control_publisher=control_publisher,
        )
        self.session_lifecycle_service = SqliteSessionLifecycleService(
            db_path,
            control_publisher=control_publisher,
        )
        self.experiment_admin_service = SqliteExperimentAdminService(db_path)


def configure_sqlite_api_overrides(
    api_app: FastAPI,
    db_path: Path,
    *,
    control_publisher: LiveSessionControlPublisher | None = None,
) -> DesktopApiServices:
    """Install desktop SQLite services and skip the server Postgres lifespan."""
    services = DesktopApiServices(db_path, control_publisher=control_publisher)

    def _read_service_dependency() -> SqliteOperatorReadService:
        return services.read_service

    def _action_service_dependency() -> SqliteOperatorActionService:
        return services.action_service

    def _event_service_dependency() -> OperatorEventService:
        return services.event_service

    def _session_lifecycle_service_dependency() -> SqliteSessionLifecycleService:
        return services.session_lifecycle_service

    def _experiment_admin_service_dependency() -> SqliteExperimentAdminService:
        return services.experiment_admin_service

    api_app.dependency_overrides[get_read_service] = _read_service_dependency
    api_app.dependency_overrides[get_action_service] = _action_service_dependency
    api_app.dependency_overrides[get_event_service] = _event_service_dependency
    api_app.dependency_overrides[get_session_lifecycle_service] = (
        _session_lifecycle_service_dependency
    )
    api_app.dependency_overrides[get_admin_service] = _experiment_admin_service_dependency

    @asynccontextmanager
    async def _desktop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    api_app.router.lifespan_context = _desktop_lifespan
    return services


def bootstrap_sqlite_api_store(db_path: Path) -> None:
    """Create or migrate the desktop SQLite schema for the API shell."""
    import sqlite3

    bootstrap_conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(bootstrap_conn)
    finally:
        bootstrap_conn.close()


__all__ = ["DesktopApiServices", "bootstrap_sqlite_api_store", "configure_sqlite_api_overrides"]
