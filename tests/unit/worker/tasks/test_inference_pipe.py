"""WS4 P3 — pipe:0/pipe:1 PCM-to-WAV helper unit tests.

Targets :func:`packages.ml_core.audio_pipe.pcm_to_wav_bytes`. The
helper lives outside ``services.worker.tasks.inference`` precisely so
both the v3.4 worker path and the WS5 P4 desktop ``gpu_ml_worker``
path can call into it without dragging the celery-tied inference
module along.
"""

from __future__ import annotations

import os
import shutil
import struct

import pytest

from packages.ml_core.audio_pipe import pcm_to_wav_bytes

# Skip the entire module if FFmpeg is not on PATH so headless CI
# without the binary stays green.
_FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not on PATH")


def _silence_pcm(seconds: float = 0.5, sample_rate: int = 16_000) -> bytes:
    """Build a deterministic block of mono s16le PCM silence."""
    n = int(seconds * sample_rate)
    return struct.pack("<" + "h" * n, *([0] * n))


def test_pcm_to_wav_bytes_emits_riff_wav_header() -> None:
    wav = pcm_to_wav_bytes(_silence_pcm())
    # The first 12 bytes are the standard RIFF/WAVE header.
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    # FFmpeg's WAV muxer always emits an "fmt " chunk next.
    assert b"fmt " in wav[:64]


def test_pcm_to_wav_bytes_is_pure_in_memory(tmp_path: object) -> None:
    """The helper must not touch the filesystem — pipe:0/pipe:1 only."""
    snapshot_before = set(os.listdir(os.getcwd()))
    _ = pcm_to_wav_bytes(_silence_pcm(0.1))
    snapshot_after = set(os.listdir(os.getcwd()))
    # No tempfile artefacts in the working directory.
    assert snapshot_before == snapshot_after


def test_pcm_to_wav_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pcm_to_wav_bytes(b"")


def test_pcm_to_wav_bytes_round_trip_length() -> None:
    """0.5 s @ 16 kHz mono s16le = 16000 samples * 2 bytes = 32_000 PCM bytes.

    The WAV container adds a header (~78 bytes for FFmpeg's WAV muxer),
    so the output should be at least the PCM length and at most a small
    constant overhead beyond it.
    """
    pcm = _silence_pcm(0.5)
    wav = pcm_to_wav_bytes(pcm)
    assert len(wav) >= len(pcm)
    assert len(wav) < len(pcm) + 256  # header overhead well under 256 bytes
