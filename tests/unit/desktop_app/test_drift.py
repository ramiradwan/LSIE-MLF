"""DriftCorrector unit tests.

Moved from ``tests/unit/worker/pipeline/test_orchestrator.py`` when the
class itself moved from ``services.worker.pipeline.orchestrator`` to
``services.desktop_app.drift`` (the poll runs in capture_supervisor;
the orchestrator only calls ``correct_timestamp``). The freeze/reset
behaviour locked by the §4.C.1 invariant table is asserted verbatim.
"""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

from services.desktop_app.drift import (
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_RESET_TIMEOUT,
    DriftCorrector,
)


class TestDriftCorrector:
    """§4.C.1 — Temporal drift correction."""

    def test_initial_offset_zero(self) -> None:
        dc = DriftCorrector()
        assert dc.drift_offset == 0.0

    def test_poll_success_computes_offset(self) -> None:
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.123456\n"

        with (
            patch("services.desktop_app.drift.subprocess.run", return_value=mock_result),
            patch("services.desktop_app.drift.time.time", return_value=1710000000.5),
        ):
            offset = dc.poll()

        expected = 1710000000.5 - 1710000000.123456
        assert abs(offset - expected) < 1e-6
        assert dc._consecutive_failures == 0
        assert not dc._frozen

    def test_poll_failure_increments_counter(self) -> None:
        dc = DriftCorrector()
        with patch(
            "services.desktop_app.drift.subprocess.run",
            side_effect=subprocess.TimeoutExpired("adb", 5),
        ):
            dc.poll()
        assert dc._consecutive_failures == 1
        assert not dc._frozen

    def test_poll_freezes_after_3_failures(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 0.5

        with patch(
            "services.desktop_app.drift.subprocess.run",
            side_effect=RuntimeError("ADB down"),
        ):
            for _ in range(DRIFT_FREEZE_AFTER_FAILURES):
                dc.poll()

        assert dc._frozen
        assert dc.drift_offset == 0.5

    def test_frozen_returns_cached_offset(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 1.5
        dc._frozen = True
        dc._frozen_at = time.monotonic()

        with patch("services.desktop_app.drift.subprocess.run") as mock_run:
            offset = dc.poll()
            mock_run.assert_not_called()

        assert offset == 1.5

    def test_frozen_resets_after_5_minutes(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 2.0
        dc._frozen = True
        dc._frozen_at = time.monotonic() - DRIFT_RESET_TIMEOUT - 1

        offset = dc.poll()
        assert offset == 0.0
        assert not dc._frozen
        assert dc._consecutive_failures == 0

    def test_correct_timestamp(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 0.5
        assert dc.correct_timestamp(100.0) == 100.5

    def test_success_resets_failure_count(self) -> None:
        dc = DriftCorrector()
        dc._consecutive_failures = 2

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.0\n"

        with (
            patch("services.desktop_app.drift.subprocess.run", return_value=mock_result),
            patch("services.desktop_app.drift.time.time", return_value=1710000000.0),
        ):
            dc.poll()

        assert dc._consecutive_failures == 0

    def test_nonzero_returncode_is_failure(self) -> None:
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("services.desktop_app.drift.subprocess.run", return_value=mock_result):
            dc.poll()

        assert dc._consecutive_failures == 1
