"""Tests for services/api/main.py router registration."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest


def _reset_api_modules() -> None:
    """Force a clean import of API modules using the test FastAPI shim."""
    for module_name in [
        "services.api.main",
        "services.api.routes",
        "services.api.routes.encounters",
        "services.api.routes.experiments",
        "services.api.routes.health",
        "services.api.routes.metrics",
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
    assert api_paths.index("/api/v1/encounters") < api_paths.index("/api/v1/experiments")
