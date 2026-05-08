"""In-memory PCM -> WAV pipe helper (WS4 P3 / §5.2).

Wraps raw PCM s16le @ 16 kHz mono in a WAV container entirely inside a
``BytesIO`` buffer so transient biometric audio never lands on the
filesystem. The Ephemeral Vault perimeter is preserved without spawning
a child process per segment.

Lives under :mod:`packages.ml_core` so both the retained
``services.worker.tasks.inference`` path and the v4 desktop
``gpu_ml_worker`` can call into it without dragging the celery-tied
inference module in.
"""

from __future__ import annotations

import io
import wave

_SAMPLE_RATE_HZ: int = 16_000
_SAMPLE_WIDTH_BYTES: int = 2
_CHANNELS: int = 1


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM s16le @ 16 kHz mono in a WAV container without touching disk.

    Builds the RIFF/WAVE header in memory via ``wave.open`` against an
    ``io.BytesIO`` buffer. Per §5.2 this keeps the Ephemeral Vault
    perimeter intact: no filesystem writes and no child-process pipe
    handles inherit the audio bytes.

    Raises ``ValueError`` on empty input.
    """
    if not pcm:
        raise ValueError("pcm_to_wav_bytes: PCM payload is empty")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(_CHANNELS)
        writer.setsampwidth(_SAMPLE_WIDTH_BYTES)
        writer.setframerate(_SAMPLE_RATE_HZ)
        writer.writeframes(pcm)
    return buffer.getvalue()
