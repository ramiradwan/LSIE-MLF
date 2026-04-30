"""Cross-platform OS adapter (Platform Abstraction Rule).

This is the **only** module in the desktop app that branches on
``sys.platform``. Every Win32 integration (Job Objects, Credential
Manager, ``SetErrorMode``, Windows-specific path resolution) and every
POSIX fallback (``os.killpg``, ``setrlimit``, ``/dev/shm`` enumeration,
XDG paths) lives behind this interface. Consumer modules
(``capture_supervisor``, ``ui_api_shell``, ``secrets``, ``cleanup``,
etc.) MUST call the public symbols below and stay OS-agnostic.

Phase boundaries: WS3 P2 introduces the IPC-block recovery sweep here.
WS3 P3 adds supervised subprocess + Win32 Job Object support. WS4 P3
adds crash-dump suppression and memory zeroisation. WS4 P4 adds the
secret-store wrapper. WS1 P2/P4 add path resolution.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def cleanup_orphan_ipc_blocks(prefix: str) -> int:
    """Unlink leftover SharedMemory blocks whose name starts with ``prefix``.

    POSIX: enumerate ``/dev/shm/`` and ``shm_unlink`` each matching name
    so a crashed parent does not leak rebooted-tmpfs entries.

    Windows: no-op. Anonymous-named ``CreateFileMapping`` mappings are
    reference-counted by the kernel and reclaimed on last-handle close;
    a crashed process leaves no orphaned shared region. Tracking-file
    schemes (writing block names to disk) are deferred until WS4 P1's
    SQLite manifest gives us a durable place to record them.

    Returns the number of blocks unlinked. A failure to remove an
    individual block is logged and skipped — the caller's start path
    must remain non-fatal.
    """
    if sys.platform == "win32":
        return 0

    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return 0

    unlinked = 0
    for entry in os.listdir(shm_dir):
        if not entry.startswith(prefix):
            continue
        path = os.path.join(shm_dir, entry)
        try:
            os.unlink(path)
            unlinked += 1
        except OSError as exc:
            logger.warning("orphan ipc block unlink failed: %s (%s)", path, exc)
    return unlinked
