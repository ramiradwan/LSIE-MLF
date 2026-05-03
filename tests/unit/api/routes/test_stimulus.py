"""Regression coverage for the retired direct stimulus route."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def _reset_api_modules() -> None:
    for module_name in [
        "services.api.main",
        "services.api.routes",
        "services.api.routes.operator",
    ]:
        sys.modules.pop(module_name, None)


def test_direct_stimulus_route_is_not_mounted() -> None:
    _reset_api_modules()
    main_module: Any = importlib.import_module("services.api.main")

    api_paths = [route.path for route in main_module.app.routes]

    assert "/api/v1/stimulus" not in api_paths
    assert "/api/v1/operator/sessions/{session_id}/stimulus" in api_paths
