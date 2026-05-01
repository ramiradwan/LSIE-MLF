"""In-memory PCM → WAV pipe helper (WS4 P3 / §5.2).

Pipes raw PCM s16le @ 16 kHz mono through ``ffmpeg -i pipe:0 ... pipe:1``
so transient biometric audio never touches the filesystem. This is the
WS4 P3 desktop replacement for the v3.4 ``inference.py`` tempfile dance
that wrote ``foo.raw``/``foo.wav`` artefacts during transcription.

Lives under :mod:`packages.ml_core` so both the v3.4
``services.worker.tasks.inference`` legacy path and the v4 desktop
``gpu_ml_worker`` (WS5 P4) can call into it without dragging the
celery-tied inference module in.
"""

from __future__ import annotations

import subprocess


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM s16le @ 16 kHz mono in a WAV container without touching disk.

    Pipes the input through ``ffmpeg -i pipe:0 ... pipe:1`` so the
    transient biometric audio never lands on the filesystem. Per §5.2
    this keeps the Ephemeral Vault perimeter intact even on hosts
    without the WS4 P3 LocalDumps exclusion (e.g. a developer running
    the worker directly).

    Raises ``ValueError`` on empty input. Raises ``RuntimeError`` if
    FFmpeg returns success but emits no output. Raises
    ``subprocess.CalledProcessError`` on FFmpeg failure (non-zero
    exit code).
    """
    if not pcm:
        raise ValueError("pcm_to_wav_bytes: PCM payload is empty")

    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=True,
    )
    if not completed.stdout:
        raise RuntimeError("ffmpeg pipe:0/pipe:1 emitted no output")
    return completed.stdout
