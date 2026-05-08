"""DriftCorrector unit tests.

Moved from ``tests/unit/worker/pipeline/test_orchestrator.py`` when the
class itself moved from ``services.worker.pipeline.orchestrator`` to
``services.desktop_app.drift`` (the poll runs in capture_supervisor;
the orchestrator only calls ``correct_timestamp``). The freeze/reset
behaviour locked by the §4.C.1 invariant table is asserted verbatim.
"""

from __future__ import annotations

import logging
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from services.desktop_app.drift import (
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_RESET_TIMEOUT,
    MAX_DRIFT_CHANGE_MS,
    MAX_TOLERATED_DRIFT_MS,
    DriftCorrector,
)


def _mock_time(*, before: float, after: float | None = None) -> MagicMock:
    """Return a ``time.time`` substitute matching the (t_before, t_after) sequence in ``poll``.

    Returns ``before`` on the first call and ``after`` on every
    subsequent call. ``after`` defaults to ``before``. The fall-through
    matters because Python ``logging`` itself calls ``time.time`` to
    stamp record creation; without it the test sees ``StopIteration``
    once a logger call fires after the two ``poll``-internal samples.
    """
    after_value = after if after is not None else before
    state = {"calls": 0}

    def fake_time() -> float:
        state["calls"] += 1
        return before if state["calls"] == 1 else after_value

    mock = MagicMock(side_effect=fake_time)
    return mock


class TestDriftCorrector:
    """§4.C.1 — Temporal drift correction."""

    def test_initial_offset_zero(self) -> None:
        dc = DriftCorrector()
        assert dc.drift_offset == 0.0

    def test_poll_success_uses_midpoint_for_offset(self) -> None:
        """Host instant compared against the device's epoch is the (t_before, t_after) midpoint.

        Removes ``RTT/2`` bias from the reported drift.
        """
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.0\n"

        with (
            patch("services.desktop_app.drift.subprocess.run", return_value=mock_result),
            patch(
                "services.desktop_app.drift.time.time",
                _mock_time(before=1710000000.5, after=1710000000.7),
            ),
        ):
            offset = dc.poll()

        # midpoint is 1710000000.6; android_epoch is 1710000000.0
        assert abs(offset - 0.6) < 1e-6
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
            patch(
                "services.desktop_app.drift.time.time",
                _mock_time(before=1710000000.0, after=1710000000.05),
            ),
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


class TestDriftLogging:
    """§4.C.1 advisory log discipline.

    ``MAX_TOLERATED_DRIFT_MS`` is an advisory parameter — exceedance
    logs at INFO level because correction still applies cleanly at
    any stable magnitude. ``MAX_DRIFT_CHANGE_MS`` is the real
    reliability signal — exceedance logs at WARNING.
    """

    def _run_poll(
        self, dc: DriftCorrector, *, host_before: float, host_after: float, android_epoch: float
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"{android_epoch}\n"
        with (
            patch("services.desktop_app.drift.subprocess.run", return_value=mock_result),
            patch(
                "services.desktop_app.drift.time.time",
                _mock_time(before=host_before, after=host_after),
            ),
        ):
            dc.poll()

    def test_stable_large_drift_logs_at_info_not_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stable 1.4 s skew across two polls produces no warnings — only an info advisory."""
        dc = DriftCorrector()
        # First poll: drift_offset = 1.400 s, no previous offset → no Δ check.
        with caplog.at_level(logging.INFO, logger="services.desktop_app.drift"):
            self._run_poll(dc, host_before=1000.0, host_after=1000.1, android_epoch=998.65)
            self._run_poll(dc, host_before=1030.0, host_after=1030.1, android_epoch=1028.65)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert warning_records == [], "stable drift must not emit warnings"
        assert any("advisory" in r.getMessage() for r in info_records)

    def test_unstable_drift_change_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A jump >MAX_DRIFT_CHANGE_MS between two polls is the real reliability signal."""
        dc = DriftCorrector()
        with caplog.at_level(logging.INFO, logger="services.desktop_app.drift"):
            # Poll 1: drift_offset = 0.020 s
            self._run_poll(dc, host_before=1000.0, host_after=1000.1, android_epoch=999.93)
            # Poll 2: drift_offset = 0.300 s → Δ = 280 ms, well above MAX_DRIFT_CHANGE_MS
            self._run_poll(dc, host_before=1030.0, host_after=1030.1, android_epoch=1029.75)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("unstable" in r.getMessage().lower() for r in warning_records), (
            f"expected an instability warning, got {[r.getMessage() for r in caplog.records]}"
        )

    def test_first_poll_skips_delta_check(self, caplog: pytest.LogCaptureFixture) -> None:
        """No previous offset on the first poll — Δdrift is undefined, must not warn."""
        dc = DriftCorrector()
        with caplog.at_level(logging.WARNING, logger="services.desktop_app.drift"):
            self._run_poll(dc, host_before=1000.0, host_after=1000.1, android_epoch=998.65)
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == []

    def test_advisory_constants_match_spec(self) -> None:
        """§4.C.1 advisory parameters are unchanged by this refactor."""
        assert MAX_TOLERATED_DRIFT_MS == 150
        # Δdrift threshold is small relative to MAX_TOLERATED_DRIFT_MS
        # because over a 30 s poll interval, even fast oscillator drift
        # (~20 ppm = ~0.6 ms / 30 s) should not approach this.
        assert 10 <= MAX_DRIFT_CHANGE_MS <= MAX_TOLERATED_DRIFT_MS
