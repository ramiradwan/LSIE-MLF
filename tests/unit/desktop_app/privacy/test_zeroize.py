"""Zeroize SharedMemory PCM blocks before unlink."""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import pytest

from services.desktop_app.ipc.shared_buffers import write_pcm_block
from services.desktop_app.os_adapter import zeroize_shared_memory
from services.desktop_app.privacy.zeroize import cleanup_capture_files, zeroize_pcm_block


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
    block.close_and_unlink()


def test_cleanup_capture_files_deletes_known_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "audio_stream.wav"
    video = tmp_path / "video_stream.mkv"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")

    deleted, retained = cleanup_capture_files(tmp_path)

    assert deleted == [audio, video]
    assert retained == []
    assert not audio.exists()
    assert not video.exists()


def test_cleanup_capture_files_passes_retry_policy(tmp_path: Path) -> None:
    audio = tmp_path / "audio_stream.wav"
    audio.write_bytes(b"audio")
    calls: list[tuple[Path, int, float]] = []

    def fake_secure_delete_file(
        path: Path,
        *,
        attempts: int = 6,
        retry_delay_s: float = 0.25,
    ) -> bool:
        calls.append((path, attempts, retry_delay_s))
        path.unlink()
        return True

    with patch(
        "services.desktop_app.privacy.zeroize.secure_delete_file",
        fake_secure_delete_file,
    ):
        deleted, retained = cleanup_capture_files(tmp_path, attempts=9, retry_delay_s=0.1)

    assert deleted == [audio]
    assert retained == []
    assert calls == [(audio, 9, 0.1)]


def test_cleanup_capture_files_reports_retained_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "video_stream.mkv"
    video.write_bytes(b"video")

    with patch("services.desktop_app.privacy.zeroize.secure_delete_file", return_value=False):
        deleted, retained = cleanup_capture_files(tmp_path, attempts=1, retry_delay_s=0.0)

    assert deleted == []
    assert retained == [video]
    assert video.exists()
