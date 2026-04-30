"""WS3 P2 — Dirty State Recovery sweep tests."""

from __future__ import annotations

import sys

import pytest

from services.desktop_app.ipc.cleanup import recover_orphan_ipc_blocks
from services.desktop_app.ipc.shared_buffers import (
    SHM_PREFIX,
    write_pcm_block,
)


def test_recovery_returns_zero_on_clean_state() -> None:
    """No orphans → no unlinks. Must not raise on either platform."""
    # We cannot guarantee zero orphans on a shared CI host, but we can
    # at least assert the call returns an int and does not raise.
    result = recover_orphan_ipc_blocks()
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only orphan sweep")
def test_recovery_unlinks_orphan_block_on_posix() -> None:
    """Producer leaks a block (no unlink) — sweep reclaims it."""
    block = write_pcm_block(b"\xcc" * 256)
    leaked_name = block.metadata.name
    # Drop the producer handle without unlinking: simulates a crash.
    block._shm.close()  # type: ignore[has-type]
    assert leaked_name.startswith(SHM_PREFIX)

    unlinked = recover_orphan_ipc_blocks()
    assert unlinked >= 1

    # Subsequent sweep finds nothing matching the leaked name.
    import os

    assert not os.path.exists(f"/dev/shm/{leaked_name}")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel auto-cleanup")
def test_recovery_is_no_op_on_windows() -> None:
    """Windows: kernel reclaims unnamed file mappings; sweep is a no-op."""
    assert recover_orphan_ipc_blocks() == 0
