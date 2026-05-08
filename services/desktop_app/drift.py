"""ADB drift correction (§4.C.1).

Moved here from ``services.worker.pipeline.orchestrator`` as part of
the desktop process-graph rewrite: the poll runs in
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
MAX_TOLERATED_DRIFT_MS: int = 150  # milliseconds — §4.C.1 advisory threshold
# §4.C.1 makes ``MAX_TOLERATED_DRIFT_MS`` an advisory parameter; what
# matters for segment alignment is that ``drift_offset`` is *stable*
# across polls. A 1.4 s skew that doesn't change between polls applies
# the same correction to every event in a segment and the math works.
# A 50 ms skew that swings ±200 ms between polls breaks alignment
# within a single segment because events get corrected against
# different offsets. The Δdrift threshold below is what we actually
# warn on; absolute exceedance becomes informational ("clocks need
# NTP resync — correction still applied").
MAX_DRIFT_CHANGE_MS: int = 50
# argv-style invocation so the shell-quoting differences between bash
# and Windows cmd.exe don't mangle the embedded ``$EPOCHREALTIME``
# expansion (which runs on the device's shell, not the host's). A
# string-form shell command would be host-shell-sensitive here.
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
    poll failures and resets after the configured timeout, matching
    §4.C.1, §12, and §13.5. It does not block segment assembly on ADB
    loss or persist drift history.
    """

    def __init__(self) -> None:
        self.drift_offset: float = 0.0
        # Set to the prior successful poll's drift on every successful
        # poll; left as ``None`` until a first successful poll exists,
        # so the Δdrift check skips on the very first observation
        # rather than comparing against the initial 0.0 sentinel.
        self._previous_drift_offset: float | None = None
        self._consecutive_failures: int = 0
        self._frozen: bool = False
        self._frozen_at: float = 0.0

    def poll(self) -> float:
        """Execute the ADB epoch poll and update ``drift_offset``.

        §4.C.1: ``drift_offset = host_utc - android_epoch``. The host
        clock is sampled twice (before and after the ``adb shell``
        round-trip) and the midpoint is used as the reference instant.
        This removes the ~RTT/2 measurement bias from anchoring at
        ``time.time()`` before the subprocess started, which over a
        ~130 ms RTT inflates the reported drift by ~65 ms even when
        the underlying clock skew is zero.

        §12 / §13.5: freeze drift after 3 failures; reset to zero after
        5 minutes of frozen state. Returns the current ``drift_offset``
        regardless of poll success.
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
            t_before = time.time()
            result = subprocess.run(
                ADB_ARGV,
                capture_output=True,
                text=True,
                timeout=5,
            )
            t_after = time.time()
            if result.returncode != 0:
                raise RuntimeError(
                    f"ADB returned code {result.returncode}: stderr={result.stderr!r}"
                )

            android_epoch = float(result.stdout.strip())
            host_utc_midpoint = (t_before + t_after) / 2.0
            rtt_s = t_after - t_before

            new_drift = host_utc_midpoint - android_epoch
            self._log_drift_observation(
                new_drift=new_drift,
                previous_drift=self._previous_drift_offset,
                rtt_s=rtt_s,
            )
            self.drift_offset = new_drift
            self._previous_drift_offset = new_drift
            self._consecutive_failures = 0

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

    def _log_drift_observation(
        self,
        *,
        new_drift: float,
        previous_drift: float | None,
        rtt_s: float,
    ) -> None:
        """Emit the per-poll drift / Δdrift / RTT diagnostic line.

        Two distinct concerns get distinct log levels:

        * **Absolute drift > MAX_TOLERATED_DRIFT_MS** is a hygiene
          observation — the host or device clock has not NTP-synced
          recently. Drift correction still applies cleanly at any
          magnitude as long as the offset is stable, so this is
          ``info`` not ``warning``.
        * **|Δdrift| > MAX_DRIFT_CHANGE_MS between consecutive polls**
          is a real reliability signal — within-segment alignment
          breaks if the offset is shifting faster than the poll
          interval. This is ``warning``.
        """
        drift_ms = new_drift * 1000.0
        rtt_ms = rtt_s * 1000.0

        delta_ms: float | None = None
        if previous_drift is not None:
            delta_ms = (new_drift - previous_drift) * 1000.0

        if delta_ms is not None and abs(delta_ms) > MAX_DRIFT_CHANGE_MS:
            logger.warning(
                "Drift unstable: Δ%+.1fms between polls (drift %+.1fms, adb RTT %.1fms)",
                delta_ms,
                drift_ms,
                rtt_ms,
            )
        elif abs(drift_ms) > MAX_TOLERATED_DRIFT_MS:
            logger.info(
                "Drift %+.1fms exceeds %dms advisory (Δ%s, adb RTT %.1fms); "
                "correction applied — consider NTP resync if drift keeps growing",
                drift_ms,
                MAX_TOLERATED_DRIFT_MS,
                f"{delta_ms:+.1f}ms" if delta_ms is not None else "n/a",
                rtt_ms,
            )

    def correct_timestamp(self, original_ts: float) -> float:
        """Apply drift correction: ``corrected_ts = original_ts + drift_offset``."""
        return original_ts + self.drift_offset
