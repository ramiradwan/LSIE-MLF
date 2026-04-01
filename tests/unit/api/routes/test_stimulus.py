"""
Tests for services/api/routes/stimulus.py — Phase 3.6 gap-fix coverage.

Verifies the POST /stimulus endpoint:
  §4.E.1 — Operator intervention bridge via Redis pub/sub
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch


def _get_stimulus_module() -> Any:
    """Import stimulus route module (conftest provides FastAPI mocks)."""
    mod_name = "services.api.routes.stimulus"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _run_async(coro: Any) -> Any:
    """Run an async function synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTriggerStimulus:
    """POST /stimulus — trigger injection via Redis pub/sub."""

    def test_publishes_to_redis_and_returns_triggered(self) -> None:
        """Returns status='triggered' with receiver count when subscribers exist."""
        mod = _get_stimulus_module()
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.publish.return_value = 2  # 2 subscribers received
        mock_redis.from_url.return_value = mock_client

        with (
            patch.dict("sys.modules", {"redis": mock_redis}),
            patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379/0"}),
        ):
            # Re-import to pick up the mocked redis
            mod = _get_stimulus_module()
            result = _run_async(mod.trigger_stimulus())

        assert result["status"] == "triggered"
        assert result["receivers"] == 2
        mock_client.publish.assert_called_once_with("stimulus:inject", "inject")
        mock_client.close.assert_called_once()

    def test_returns_warning_when_no_subscribers(self) -> None:
        """Returns status='published' with warning when no orchestrator is listening."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.publish.return_value = 0  # No subscribers
        mock_redis.from_url.return_value = mock_client

        with (
            patch.dict("sys.modules", {"redis": mock_redis}),
            patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379/0"}),
        ):
            mod = _get_stimulus_module()
            result = _run_async(mod.trigger_stimulus())

        assert result["status"] == "published"
        assert result["receivers"] == 0
        assert "warning" in result

    def test_raises_503_when_redis_unavailable(self) -> None:
        """Raises HTTPException with status 503 when Redis connection fails."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ConnectionError("Connection refused")

        with (
            patch.dict("sys.modules", {"redis": mock_redis}),
            patch.dict("os.environ", {"REDIS_URL": "redis://bad:6379/0"}),
        ):
            mod = _get_stimulus_module()
            try:
                _run_async(mod.trigger_stimulus())
                raised = False
            except Exception as exc:
                raised = True
                assert hasattr(exc, "status_code")
                assert exc.status_code == 503  # type: ignore[attr-defined]

        assert raised, "Expected HTTPException 503 to be raised"

    def test_response_schema(self) -> None:
        """Response dict contains required keys."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.publish.return_value = 1
        mock_redis.from_url.return_value = mock_client

        with (
            patch.dict("sys.modules", {"redis": mock_redis}),
            patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379/0"}),
        ):
            mod = _get_stimulus_module()
            result = _run_async(mod.trigger_stimulus())

        assert "status" in result
        assert "receivers" in result
