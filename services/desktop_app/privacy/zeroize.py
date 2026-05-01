"""Zeroize SharedMemory PCM blocks before unlink (WS4 P3 / §5.2).

Per §5.2 the Ephemeral Vault contract requires raw biometric media to
be securely overwritten the moment it is no longer needed; SharedMemory
PCM blocks are the most ephemeral leg of that pipeline (they live
30 seconds at most). Plain unlink leaves the bytes in physical memory
until the OS reclaims the page; on POSIX the ``/dev/shm`` tmpfs entry
is reusable in-place by another process; on Windows the
``CreateFileMapping`` backing pages may be paged out to ``pagefile.sys``
before reclamation. Calling :func:`ctypes.memset` against the mapping
address closes both windows.

The implementation lives behind
:func:`services.desktop_app.os_adapter.zeroize_shared_memory` so this
module stays free of platform branches and the call sites
(``shared_buffers.PcmBlock.close_and_unlink``) keep their narrow
public surface.
"""

from __future__ import annotations

import logging
from typing import Any

from services.desktop_app.os_adapter import zeroize_shared_memory

logger = logging.getLogger(__name__)


def zeroize_pcm_block(shm: Any) -> int:
    """Wipe ``shm.buf`` to all zeroes; return bytes wiped.

    Wraps the OS adapter primitive with a logger so the privacy path
    leaves a verifiable trail. Idempotent and tolerant of an already-
    closed buffer (returns ``0``).
    """
    nbytes = zeroize_shared_memory(shm)
    if nbytes:
        logger.debug("zeroized %d-byte SharedMemory PCM block", nbytes)
    return nbytes
