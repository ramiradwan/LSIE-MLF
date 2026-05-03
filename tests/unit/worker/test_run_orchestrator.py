"""
Tests for the retained server-side services/worker/run_orchestrator.py entrypoint.

Verifies the orchestrator entrypoint:
  §4.C — Module C standalone process startup
  §4.E.1 — retained server/API stimulus trigger setup (auto-trigger + Redis listener)
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch


class TestMain:
    """run_orchestrator.main(): orchestrator process startup."""

    def _run_main(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        """Run main() with fully mocked infrastructure. Returns mock references."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_cls = MagicMock(return_value=mock_orchestrator)

        # Make run() return a completed coroutine
        async def _noop() -> None:
            pass

        mock_orchestrator.run = MagicMock(return_value=_noop())

        mock_auto_timer = MagicMock()
        mock_redis_thread = MagicMock()

        env_vars = {
            "STREAM_URL": "",
            "EXPERIMENT_ID": "greeting_line_v1",
        }
        if env:
            env_vars.update(env)

        with (
            patch(
                "services.worker.run_orchestrator.Orchestrator",
                mock_orchestrator_cls,
            )
            if False
            else patch(
                "services.worker.pipeline.orchestrator.Orchestrator",
                mock_orchestrator_cls,
            ),
            patch.dict("os.environ", env_vars, clear=False),
            patch(
                "services.worker.pipeline.stimulus.setup_auto_trigger",
                return_value=mock_auto_timer,
            ) as mock_setup_auto,
            patch(
                "services.worker.pipeline.stimulus.start_redis_listener",
                return_value=mock_redis_thread,
            ) as mock_start_redis,
        ):
            from services.worker.run_orchestrator import main

            main()

        return {
            "orchestrator_cls": mock_orchestrator_cls,
            "orchestrator": mock_orchestrator,
            "auto_timer": mock_auto_timer,
            "redis_thread": mock_redis_thread,
            "setup_auto": mock_setup_auto,
            "start_redis": mock_start_redis,
        }

    def test_instantiates_orchestrator_with_env_vars(self) -> None:
        """Orchestrator is created with STREAM_URL and EXPERIMENT_ID from env."""
        refs = self._run_main(
            env={
                "STREAM_URL": "https://tiktok.com/@test",
                "EXPERIMENT_ID": "test_exp_v1",
            }
        )
        refs["orchestrator_cls"].assert_called_once_with(
            stream_url="https://tiktok.com/@test",
            experiment_id="test_exp_v1",
        )

    def test_default_env_vars(self) -> None:
        """Uses default values when env vars are not set."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove env vars if they exist
            import os

            os.environ.pop("STREAM_URL", None)
            os.environ.pop("EXPERIMENT_ID", None)

            mock_orch = MagicMock()
            mock_orch_cls = MagicMock(return_value=mock_orch)

            async def _noop() -> None:
                pass

            mock_orch.run = MagicMock(return_value=_noop())

            with (
                patch(
                    "services.worker.pipeline.orchestrator.Orchestrator",
                    mock_orch_cls,
                ),
                patch("services.worker.pipeline.stimulus.setup_auto_trigger", return_value=None),
                patch("services.worker.pipeline.stimulus.start_redis_listener", return_value=None),
            ):
                from services.worker.run_orchestrator import main

                main()

            mock_orch_cls.assert_called_once_with(
                stream_url="",
                experiment_id="greeting_line_v1",
            )

    def test_runs_orchestrator_loop(self) -> None:
        """Calls orchestrator.run() via asyncio event loop."""
        refs = self._run_main()
        refs["orchestrator"].run.assert_called_once()

    def test_stimulus_triggers_setup(self) -> None:
        """Retained server entrypoint sets up both auto-trigger and Redis listener."""
        refs = self._run_main()
        refs["setup_auto"].assert_called_once_with(refs["orchestrator"])
        refs["start_redis"].assert_called_once_with(refs["orchestrator"])

    def test_auto_timer_cancelled_on_shutdown(self) -> None:
        """Auto-trigger timer is cancelled during shutdown."""
        refs = self._run_main()
        refs["auto_timer"].cancel.assert_called_once()

    def test_orchestrator_stop_called_on_shutdown(self) -> None:
        """orchestrator.stop() is called during shutdown."""
        refs = self._run_main()
        refs["orchestrator"].stop.assert_called()

    def test_handles_stimulus_setup_failure(self) -> None:
        """Continues even if stimulus trigger setup fails."""
        mock_orch = MagicMock()
        mock_orch_cls = MagicMock(return_value=mock_orch)

        async def _noop() -> None:
            pass

        mock_orch.run = MagicMock(return_value=_noop())

        with (
            patch(
                "services.worker.pipeline.orchestrator.Orchestrator",
                mock_orch_cls,
            ),
            patch(
                "services.worker.pipeline.stimulus.setup_auto_trigger",
                side_effect=ImportError("no redis"),
            ),
            patch(
                "services.worker.pipeline.stimulus.start_redis_listener",
                side_effect=ImportError("no redis"),
            ),
        ):
            from services.worker.run_orchestrator import main

            # Should not raise
            main()

        mock_orch.run.assert_called_once()

    def test_entrypoint_leaves_handoff_enqueue_to_orchestrator(self) -> None:
        mock_task = MagicMock()
        inference_module = ModuleType("services.worker.tasks.inference")
        inference_module.process_segment = mock_task
        sys.modules.pop("services.worker.tasks.inference", None)

        with patch.dict(sys.modules, {"services.worker.tasks.inference": inference_module}):
            importlib.invalidate_caches()
            refs = self._run_main()

        refs["orchestrator"].run.assert_called_once()
        mock_task.delay.assert_not_called()
