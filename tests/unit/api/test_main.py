"""Tests for services/api/main.py router registration."""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _reset_api_modules() -> None:
    """Force a clean import of API modules using the test FastAPI shim."""
    for module_name in [
        "services.api.main",
        "services.api.routes",
        "services.api.routes.comodulation",
        "services.api.routes.encounters",
        "services.api.routes.experiments",
        "services.api.routes.health",
        "services.api.routes.metrics",
        "services.api.routes.operator",
        "services.api.routes.physiology",
        "services.api.routes.sessions",
        "services.api.routes.stimulus",
    ]:
        sys.modules.pop(module_name, None)


@pytest.fixture()
def routes_package() -> Any:
    """Import the routes package and expose its exported router modules."""
    _reset_api_modules()
    return importlib.import_module("services.api.routes")


@pytest.fixture()
def main_module() -> Any:
    """Import the API main module with a fresh FastAPI shim application."""
    _reset_api_modules()
    return importlib.import_module("services.api.main")


def test_routes_package_exports_experiments(routes_package: Any) -> None:
    """services.api.routes exposes the experiments router module."""
    assert "experiments" in dir(routes_package)


def test_routes_package_exports_physiology(routes_package: Any) -> None:
    """services.api.routes exposes the physiology router module."""
    assert "physiology" in dir(routes_package)


def test_main_app_registers_physiology_webhook(main_module: Any) -> None:
    """The main app mounts the Oura webhook under /api/v1."""
    api_paths = [route.path for route in main_module.app.routes if route.path.startswith("/api/v1")]
    assert "/api/v1/ingest/oura/webhook" in api_paths


def test_main_app_registers_experiments_after_encounters(main_module: Any) -> None:
    """The main app mounts experiments under /api/v1 after encounters."""
    api_paths = [route.path for route in main_module.app.routes if route.path.startswith("/api/v1")]

    assert "/api/v1/experiments" in api_paths
    assert "/api/v1/experiments/{experiment_id}" in api_paths
    assert "/api/v1/experiments/{experiment_id}/arms" in api_paths
    assert "/api/v1/experiments/{experiment_id}/arms/{arm_id}" in api_paths
    assert api_paths.index("/api/v1/encounters") < api_paths.index("/api/v1/experiments")


def test_start_oura_hydration_worker_returns_none_when_client_id_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_api_modules()
    monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
    main_module = importlib.import_module("services.api.main")
    create_redis = MagicMock()

    monkeypatch.setattr(main_module, "_create_redis_client", create_redis)

    assert main_module._start_oura_hydration_worker() is None
    create_redis.assert_not_called()


def test_start_oura_hydration_worker_starts_daemon_thread_when_client_id_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_api_modules()
    monkeypatch.setenv("OURA_CLIENT_ID", "client-id")
    main_module = importlib.import_module("services.api.main")
    redis_client = MagicMock()
    thread = MagicMock()
    thread_cls = MagicMock(return_value=thread)

    monkeypatch.setattr(main_module, "_create_redis_client", MagicMock(return_value=redis_client))
    monkeypatch.setattr(main_module.threading, "Thread", thread_cls)

    result = main_module._start_oura_hydration_worker()

    assert result == (thread, redis_client)
    thread_cls.assert_called_once()
    assert thread_cls.call_args.kwargs["daemon"] is True
    assert thread_cls.call_args.kwargs["name"] == "oura-hydration-worker"
    thread.start.assert_called_once_with()


def test_lifespan_starts_hydration_worker_when_oura_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_api_modules()
    monkeypatch.setenv("OURA_CLIENT_ID", "client-id")
    main_module = importlib.import_module("services.api.main")
    redis_client = MagicMock()
    start_worker = MagicMock(return_value=(MagicMock(), redis_client))
    init_pool = AsyncMock()
    close_pool = AsyncMock()

    monkeypatch.setattr(main_module, "_start_oura_hydration_worker", start_worker)
    monkeypatch.setattr(main_module, "init_pool", init_pool)
    monkeypatch.setattr(main_module, "close_pool", close_pool)

    async def _run() -> None:
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(_run())

    init_pool.assert_awaited_once()
    start_worker.assert_called_once_with()
    redis_client.close.assert_called_once_with()
    close_pool.assert_awaited_once()


def test_lifespan_skips_hydration_worker_without_oura_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_api_modules()
    monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
    main_module = importlib.import_module("services.api.main")
    init_pool = AsyncMock()
    close_pool = AsyncMock()
    start_worker = MagicMock(return_value=None)

    monkeypatch.setattr(main_module, "_start_oura_hydration_worker", start_worker)
    monkeypatch.setattr(main_module, "init_pool", init_pool)
    monkeypatch.setattr(main_module, "close_pool", close_pool)

    async def _run() -> None:
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(_run())

    init_pool.assert_awaited_once()
    start_worker.assert_called_once_with()
    close_pool.assert_awaited_once()
