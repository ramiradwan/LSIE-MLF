"""ADB drift correction (§4.C.1, WS3 P3).

Moved here from ``services.worker.pipeline.orchestrator`` as part of
the v4.0 process-graph rewrite: the poll runs in
``services.desktop_app.processes.capture_supervisor`` (where it shares
a process with the ADB-connected device control loop) and the resulting
``drift_offset`` is shipped to ``module_c_orchestrator`` over the IPC
``drift_updates`` channel. ``Orchestrator`` retains an instance for the
``correct_timestamp`` apply-side call but no longer calls ``poll``.

The freeze/reset thresholds and the
``corrected_ts = original_ts + drift_offset`` invariant survive
verbatim from §4.C.1.
"""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

# §4.C.1 Drift polling specification
DRIFT_POLL_INTERVAL: int = 30  # seconds
MAX_TOLERATED_DRIFT_MS: int = 150  # milliseconds
# argv-style invocation so the shell-quoting differences between bash
# and Windows cmd.exe don't mangle the embedded ``$EPOCHREALTIME``
# expansion (which runs on the device's shell, not the host's). The
# v3.4 string-form ``adb shell 'echo $EPOCHREALTIME'`` only worked in
# the bash-inside-docker capture container.
ADB_ARGV: list[str] = ["adb", "shell", "echo $EPOCHREALTIME"]
# Backwards-compat alias for any caller that imported the v3.4 name.
ADB_COMMAND: str = " ".join(ADB_ARGV)
DRIFT_FREEZE_AFTER_FAILURES: int = 3
DRIFT_RESET_TIMEOUT: int = 300  # 5 minutes in seconds


class DriftCorrector:
    """Maintain drift-corrected UTC timestamps for Module C.

    Accepts host time and Android epoch readings from the configured
    ADB poll, produces the current seconds offset, and applies it to
    media/event timestamps. It freezes the last offset after repeated
    poll failures and resets after the configured timeout (§13.5); it
    does not block segment assembly on ADB loss or persist drift
    history.
    """

    def __init__(self) -> None:
        self.drift_offset: float = 0.0
        self._consecutive_failures: int = 0
        self._frozen: bool = False
        self._frozen_at: float = 0.0

    def poll(self) -> float:
        """Execute the ADB epoch poll and update ``drift_offset``.

        §4.C.1: ``drift_offset = host_utc - android_epoch``.
        §12 Hardware loss C: freeze drift after 3 failures; reset to
        zero after 5 minutes of frozen state. Returns the current
        ``drift_offset`` regardless of poll success.
        """
        if self._frozen:
            elapsed = time.monotonic() - self._frozen_at
            if elapsed >= DRIFT_RESET_TIMEOUT:
                logger.warning("Drift frozen for %ds, resetting to zero", int(elapsed))
                self.drift_offset = 0.0
                self._frozen = False
                self._consecutive_failures = 0
            return self.drift_offset

        try:
            host_utc = time.time()
            result = subprocess.run(
                ADB_ARGV,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ADB returned code {result.returncode}: stderr={result.stderr!r}"
                )

            android_epoch = float(result.stdout.strip())

            self.drift_offset = host_utc - android_epoch
            self._consecutive_failures = 0

            drift_ms = abs(self.drift_offset * 1000)
            if drift_ms > MAX_TOLERATED_DRIFT_MS:
                logger.warning(
                    "Drift %.1fms exceeds %dms tolerance",
                    drift_ms,
                    MAX_TOLERATED_DRIFT_MS,
                )

        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            logger.error(
                "ADB poll failed (%d/%d): %s",
                self._consecutive_failures,
                DRIFT_FREEZE_AFTER_FAILURES,
                exc,
            )

            if self._consecutive_failures >= DRIFT_FREEZE_AFTER_FAILURES and not self._frozen:
                logger.warning("Freezing drift at %.6f", self.drift_offset)
                self._frozen = True
                self._frozen_at = time.monotonic()

        return self.drift_offset

    def correct_timestamp(self, original_ts: float) -> float:
        """Apply drift correction: ``corrected_ts = original_ts + drift_offset``."""
        return original_ts + self.drift_offset
