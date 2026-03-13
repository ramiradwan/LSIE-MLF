"""
Tests for services/worker/vault_cron.py — Phase 4.4 validation.

Verifies vault cron calls EphemeralVault.secure_delete() correctly
and handles exceptions per §5.1.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, call, patch

from services.worker.vault_cron import INTERVAL_HOURS, SHRED_TARGETS


class TestVaultCron:
    """§5.1 — Secure deletion scheduler."""

    def test_shred_targets(self) -> None:
        """§5.1 — Targets /data/raw/ and /data/interim/."""
        assert "/data/raw/" in SHRED_TARGETS
        assert "/data/interim/" in SHRED_TARGETS

    def test_interval_24_hours(self) -> None:
        """§5.1 — 24-hour deletion cycle."""
        assert INTERVAL_HOURS == 24

    @patch("services.worker.vault_cron.time.sleep", side_effect=StopIteration)
    @patch("services.worker.vault_cron.EphemeralVault")
    def test_calls_secure_delete_for_each_target(
        self, mock_vault: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """§5.1 — secure_delete() called for each target directory."""
        from services.worker.vault_cron import run_vault_cron

        with contextlib.suppress(StopIteration):
            run_vault_cron()

        expected_calls = [call.secure_delete(t) for t in SHRED_TARGETS]
        mock_vault.assert_has_calls(expected_calls, any_order=False)

    @patch("services.worker.vault_cron.time.sleep", side_effect=StopIteration)
    @patch("services.worker.vault_cron.EphemeralVault")
    def test_handles_delete_exception(self, mock_vault: MagicMock, mock_sleep: MagicMock) -> None:
        """§5.1 — Exception in secure_delete logged, loop continues."""
        mock_vault.secure_delete.side_effect = [
            OSError("permission denied"),
            None,  # Second target succeeds
        ]

        from services.worker.vault_cron import run_vault_cron

        with contextlib.suppress(StopIteration):
            run_vault_cron()

        # Both targets should have been attempted
        assert mock_vault.secure_delete.call_count == 2

    @patch("services.worker.vault_cron.time.sleep", side_effect=StopIteration)
    @patch("services.worker.vault_cron.EphemeralVault")
    def test_sleeps_24_hours(self, mock_vault: MagicMock, mock_sleep: MagicMock) -> None:
        """§5.1 — Sleeps for 24 hours between cycles."""
        from services.worker.vault_cron import run_vault_cron

        with contextlib.suppress(StopIteration):
            run_vault_cron()

        mock_sleep.assert_called_with(INTERVAL_HOURS * 3600)
