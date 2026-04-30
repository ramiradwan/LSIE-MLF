"""WS3 P2 — SharedMemory PCM transport unit tests."""

from __future__ import annotations

import hashlib

import pytest

from services.desktop_app.ipc.shared_buffers import (
    SHM_PREFIX,
    PcmBlockMetadata,
    read_pcm_block,
    write_pcm_block,
)


def _bytes_30s_at_16khz_s16le_mono() -> bytes:
    # 16 kHz × 30 s × 2 bytes = 960 KB. Fixed-pattern bytes so SHA-256 is deterministic.
    return (b"\x12\x34" * 480_000)[:960_000]


def test_write_then_read_roundtrip() -> None:
    audio = _bytes_30s_at_16khz_s16le_mono()
    block = write_pcm_block(audio)
    try:
        assert block.metadata.name.startswith(SHM_PREFIX)
        assert block.metadata.byte_length == len(audio)
        assert block.metadata.sha256 == hashlib.sha256(audio).hexdigest()

        recovered = read_pcm_block(block.metadata)
        assert recovered == audio
    finally:
        block.close_and_unlink()


def test_unique_names_across_writes() -> None:
    audio = b"\x00" * 1024
    block_1 = write_pcm_block(audio)
    block_2 = write_pcm_block(audio)
    try:
        assert block_1.metadata.name != block_2.metadata.name
    finally:
        block_1.close_and_unlink()
        block_2.close_and_unlink()


def test_empty_audio_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        write_pcm_block(b"")


def test_checksum_mismatch_raises() -> None:
    audio = b"\x42" * 256
    block = write_pcm_block(audio)
    try:
        tampered = PcmBlockMetadata(
            name=block.metadata.name,
            byte_length=block.metadata.byte_length,
            sha256="0" * 64,  # valid hex shape but wrong digest
        )
        with pytest.raises(ValueError, match="checksum mismatch"):
            read_pcm_block(tampered)
    finally:
        block.close_and_unlink()


def test_close_and_unlink_is_idempotent() -> None:
    block = write_pcm_block(b"\x99" * 64)
    block.close_and_unlink()
    block.close_and_unlink()  # second call: must not raise


def test_read_after_close_raises_filenotfound() -> None:
    block = write_pcm_block(b"\xaa" * 32)
    metadata = block.metadata
    block.close_and_unlink()
    with pytest.raises(FileNotFoundError):
        read_pcm_block(metadata)


def test_multiple_consumers_can_attach_concurrently() -> None:
    """A producer block must support multiple sequential reads."""
    audio = b"\x77" * 4096
    block = write_pcm_block(audio)
    try:
        first = read_pcm_block(block.metadata)
        second = read_pcm_block(block.metadata)
        assert first == audio
        assert second == audio
    finally:
        block.close_and_unlink()
