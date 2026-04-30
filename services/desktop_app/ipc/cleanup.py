"""Dirty State Recovery sweep for orphan IPC blocks (WS3 P2 / WS4 P2).

Run at ``ui_api_shell`` startup, before any process is spawned. Walks
the platform's shared-memory namespace looking for blocks named
``lsie_ipc_*`` and unlinks them — these are residue from a previous
ungraceful exit (parent crash, force-kill, BSOD). The platform-specific
mechanics live behind ``services.desktop_app.os_adapter`` per the
project's Platform Abstraction Rule.

This is the IPC half of the broader Dirty State Recovery sweep; the
SQLite half lives in ``services.desktop_app.state.recovery`` (WS4 P2).
"""

from __future__ import annotations

import logging

from services.desktop_app.ipc.shared_buffers import SHM_PREFIX
from services.desktop_app.os_adapter import cleanup_orphan_ipc_blocks

logger = logging.getLogger(__name__)


def recover_orphan_ipc_blocks() -> int:
    """Unlink every leftover ``lsie_ipc_*`` SharedMemory block.

    Non-fatal: an unlink failure on an individual block is logged by
    the OS adapter and skipped. Returns the total count unlinked so
    the caller can surface it to telemetry.
    """
    unlinked = cleanup_orphan_ipc_blocks(SHM_PREFIX)
    if unlinked:
        logger.info("dirty-state recovery unlinked %d orphan ipc blocks", unlinked)
    return unlinked
