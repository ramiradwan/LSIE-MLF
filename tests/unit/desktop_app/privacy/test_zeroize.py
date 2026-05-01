"""WS4 P3 — zeroize SharedMemory PCM blocks before unlink."""

from __future__ import annotations

from multiprocessing import shared_memory

import pytest

from services.desktop_app.ipc.shared_buffers import write_pcm_block
from services.desktop_app.os_adapter import zeroize_shared_memory
from services.desktop_app.privacy.zeroize import zeroize_pcm_block


def test_zeroize_shared_memory_overwrites_buffer() -> None:
    shm = shared_memory.SharedMemory(create=True, size=16)
    try:
        buf = shm.buf
        assert buf is not None
        buf[:] = b"\xab" * 16
        assert bytes(buf[:]) == b"\xab" * 16

        wiped = zeroize_shared_memory(shm)
        assert wiped == 16
        assert bytes(buf[:]) == b"\x00" * 16
    finally:
        shm.close()
        shm.unlink()


def test_zeroize_shared_memory_idempotent_on_closed() -> None:
    shm = shared_memory.SharedMemory(create=True, size=8)
    shm.close()
    shm.unlink()
    # The Python SharedMemory wrapper still exposes a stale .buf-less
    # shape after close; the wipe helper must accept that.
    assert zeroize_shared_memory(shm) == 0


def test_zeroize_pcm_block_wipes_audio_before_unlink() -> None:
    """The privacy wrapper must wipe the bytes while the producer owns the mapping."""
    audio = b"\xde\xad\xbe\xef" * 64  # 256 bytes
    block = write_pcm_block(audio)
    try:
        wiped = zeroize_pcm_block(block._shm)
        assert wiped == len(audio)
        # After wipe but before close: the buffer is all zero.
        buf = block._shm.buf
        assert buf is not None
        assert bytes(buf[: len(audio)]) == b"\x00" * len(audio)
    finally:
        block.close_and_unlink()


def test_pcm_block_close_zeroizes_buffer() -> None:
    """``close_and_unlink`` invokes zeroize before unlinking the OS name."""
    audio = b"\xff" * 128
    block = write_pcm_block(audio)
    name = block.metadata.name
    block.close_and_unlink()

    # Re-attach should fail because the producer side unlinked. On
    # Windows the kernel reclaims the mapping when the last handle
    # closes; on POSIX the shm_unlink removes the /dev/shm entry.
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_close_and_unlink_remains_idempotent() -> None:
    audio = b"\x55" * 32
    block = write_pcm_block(audio)
    block.close_and_unlink()
    # Second call must not raise.
    block.close_and_unlink()
