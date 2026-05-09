"""In-memory PCM-to-WAV helper unit tests.

Targets :func:`packages.ml_core.audio_pipe.pcm_to_wav_bytes`. The
helper lives outside ``services.worker.tasks.inference`` so both the
retained worker path and the v4 desktop ``gpu_ml_worker`` path can
call into it without dragging the celery-tied inference module along.
"""

from __future__ import annotations

import os
import struct

import pytest

from packages.ml_core.audio_pipe import pcm_to_wav_bytes


def _silence_pcm(seconds: float = 0.5, sample_rate: int = 16_000) -> bytes:
    """Build a deterministic block of mono s16le PCM silence."""
    n = int(seconds * sample_rate)
    return struct.pack("<" + "h" * n, *([0] * n))


def test_pcm_to_wav_bytes_emits_riff_wav_header() -> None:
    wav = pcm_to_wav_bytes(_silence_pcm())
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert b"fmt " in wav[:64]


def test_pcm_to_wav_bytes_is_pure_in_memory() -> None:
    """The helper must not touch the filesystem."""
    snapshot_before = set(os.listdir(os.getcwd()))
    _ = pcm_to_wav_bytes(_silence_pcm(0.1))
    snapshot_after = set(os.listdir(os.getcwd()))
    assert snapshot_before == snapshot_after


def test_pcm_to_wav_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pcm_to_wav_bytes(b"")


def test_pcm_to_wav_bytes_round_trip_length() -> None:
    """0.5 s @ 16 kHz mono s16le = 16000 samples * 2 bytes = 32_000 PCM bytes.

    The WAV container adds a header (44 bytes for the stdlib ``wave``
    muxer), so the output should be at least the PCM length and at most
    a small constant overhead beyond it.
    """
    pcm = _silence_pcm(0.5)
    wav = pcm_to_wav_bytes(pcm)
    assert len(wav) >= len(pcm)
    assert len(wav) < len(pcm) + 256
