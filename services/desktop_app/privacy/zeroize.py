"""Zeroize SharedMemory PCM blocks before unlink.

The Ephemeral Vault contract requires raw biometric media to be
overwritten as soon as it is no longer needed. SharedMemory PCM blocks
are the most ephemeral part of that pipeline, but plain unlink leaves
bytes resident until the OS reclaims the page. Calling
:func:`ctypes.memset` against the mapping address closes that window.

The implementation lives behind
:func:`services.desktop_app.os_adapter.zeroize_shared_memory` so this
module stays free of platform branches and call sites keep a narrow
public surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.desktop_app.os_adapter import secure_delete_file, zeroize_shared_memory

logger = logging.getLogger(__name__)

_RAW_CAPTURE_FILENAMES: tuple[str, str] = ("audio_stream.wav", "video_stream.mkv")


def zeroize_pcm_block(shm: Any) -> int:
    """Wipe ``shm.buf`` to all zeroes; return bytes wiped."""
    nbytes = zeroize_shared_memory(shm)
    if nbytes:
        logger.debug("zeroized %d-byte SharedMemory PCM block", nbytes)
    return nbytes


def cleanup_capture_files(capture_dir: Path) -> tuple[list[Path], list[Path]]:
    """Delete raw capture artifacts from ``capture_dir`` and report what happened."""
    deleted: list[Path] = []
    retained: list[Path] = []
    for filename in _RAW_CAPTURE_FILENAMES:
        path = capture_dir / filename
        if not path.exists():
            continue
        if secure_delete_file(path):
            deleted.append(path)
            logger.info("deleted transient capture artifact %s", path)
        else:
            retained.append(path)
            logger.warning("retained transient capture artifact %s", path)
    return deleted, retained
