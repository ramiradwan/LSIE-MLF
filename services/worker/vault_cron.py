"""
Vault Cron — §5.1 Secure Deletion Scheduler

Executes shred -vfz -n 3 every 24 hours on /data/raw/ and /data/interim/.
"""

from __future__ import annotations

import logging
import time

from packages.ml_core.encryption import EphemeralVault

logger = logging.getLogger(__name__)

SHRED_TARGETS: list[str] = ["/data/raw/", "/data/interim/"]
INTERVAL_HOURS: int = 24


def run_vault_cron() -> None:
    """
    §5.1 — Periodic secure deletion loop.
    Runs indefinitely, executing shred on target directories every 24 hours.
    """
    while True:
        for target in SHRED_TARGETS:
            try:
                EphemeralVault.secure_delete(target)
                logger.info("Secure deletion completed: %s", target)
            except Exception:
                logger.exception("Secure deletion failed: %s", target)
        time.sleep(INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_vault_cron()
