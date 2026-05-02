"""
faster-whisper transcription wrapper for Module D audio input (§4.D.1).

The module lazy-loads the CTranslate2-backed Whisper model and transcribes
16 kHz audio to UTF-8 text. The :meth:`TranscriptionEngine.transcribe`
method accepts either a filesystem path (legacy v3.4 callers) or a
binary file-like object such as :class:`io.BytesIO` — WS4 P3 wires the
desktop runtime's pipe:0/pipe:1 FFmpeg path to feed the bytes in
memory so transient PCM never lands on disk. The production runtime
floor is CUDA 12.x with cuDNN 9 and ``ctranslate2 >= 4.5.0`` on
NVIDIA Turing (SM 7.5+) hardware (v4.0 §10.2 / §11.x); compute type is
fixed to INT8 and is not operator-configurable. WS2 P2 introduces an
``LSIE_DEV_FORCE_CPU_SPEECH`` escape hatch for Pascal developer hosts
that cannot host the production GPU speech path.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import IO, Any

logger = logging.getLogger(__name__)

_TURING_COMPUTE_CAP: float = 7.5
"""Production GPU floor for the v4.0 speech path (NVIDIA Turing)."""

_NVIDIA_SMI_TIMEOUT_S: float = 5.0


def resolve_speech_device() -> str:
    """Pick the faster-whisper device per v4.0 §11.x.

    Resolution order:

    1. ``LSIE_DEV_FORCE_CPU_SPEECH=1`` → ``"cpu"`` (developer escape
       hatch for Pascal hosts that cannot host the cuDNN 9 / CT2 4.5+
       production GPU stack).
    2. ``nvidia-smi`` reports a GPU with compute capability ≥ 7.5
       → ``"cuda"``.
    3. Otherwise → ``"cpu"`` (no NVIDIA GPU detected; CPU INT8 path).

    The cross-encoder semantic scorer is unaffected; it stays on the
    GPU because PyTorch alone (without CTranslate2 loaded in the same
    process) does not trigger the cuDNN-version collision that drove
    this resolver in the first place.

    Returns the literal device string consumed by
    :class:`faster_whisper.WhisperModel`.
    """
    if os.environ.get("LSIE_DEV_FORCE_CPU_SPEECH") == "1":
        logger.info("LSIE_DEV_FORCE_CPU_SPEECH=1 — speech path routed to CPU")
        return "cpu"
    cap = _query_max_compute_capability()
    if cap is not None and cap >= _TURING_COMPUTE_CAP:
        return "cuda"
    if cap is not None:
        logger.warning(
            "speech path falling back to CPU: detected compute capability %.1f < %.1f",
            cap,
            _TURING_COMPUTE_CAP,
        )
    return "cpu"


def _query_max_compute_capability() -> float | None:
    """Return the highest NVIDIA compute capability nvidia-smi reports.

    Returns ``None`` if ``nvidia-smi`` is missing, exits non-zero,
    times out, or every reported row fails to parse as a float.
    Multi-GPU hosts return the maximum of all queried capabilities
    on the assumption that the CUDA runtime selects the most capable
    device by default.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    capabilities: list[float] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            capabilities.append(float(line))
        except ValueError:
            continue
    return max(capabilities) if capabilities else None


class TranscriptionEngine:
    """
    Transcribe 16 kHz audio segments with faster-whisper.

    Accepts a model size and device identifier, lazy-loads the configured
    Whisper model, and produces concatenated UTF-8 transcript text for a
    caller-provided audio path. It does not expose a compute_type override,
    perform semantic matching, or persist transcripts; INT8 is fixed for the
    supported runtime.
    """

    # v4.0 §11.x — compute_type is locked to "int8". On the production
    # Turing (SM 7.5+) floor the INT8 path benefits from DP4A and the
    # Turing/Ampere INT8 Tensor Cores. On the Pascal developer host
    # exposed via LSIE_DEV_FORCE_CPU_SPEECH, INT8 is the right CPU
    # default for faster-whisper too. Allowing operator overrides would
    # silently fall back to FP16 on Pascal (which lacks the kernel) and
    # mask a misconfiguration; pinning the compute type keeps the
    # speech path's accuracy/latency contract reproducible across hosts.
    _COMPUTE_TYPE: str = "int8"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str | None = None,
    ) -> None:
        self.model_size = model_size
        # device=None defers to the v4.0 §11.x resolver so a Pascal
        # developer host with LSIE_DEV_FORCE_CPU_SPEECH=1 routes to CPU
        # automatically without the caller plumbing the env check.
        # Explicit overrides ("cuda" / "cpu") still take precedence —
        # tests and the WS5 cloud worker need the deterministic path.
        self.device = device if device is not None else resolve_speech_device()
        self.compute_type = self._COMPUTE_TYPE
        self._model: Any = None  # Lazy-loaded WhisperModel

    def load_model(self) -> None:
        """Load faster-whisper model into GPU memory. Aborts startup on failure (§4.D contract)."""
        from faster_whisper import WhisperModel

        # §4.D.1 — INT8 quantization. CUDA path is cuDNN 9 / CT2 4.5+ on
        # the Turing+ production floor; CPU path is the Pascal developer
        # escape hatch routed by LSIE_DEV_FORCE_CPU_SPEECH (v4.0 §11.x).
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio: str | IO[bytes],
        language: str | None = None,
    ) -> str:
        """
        Transcribe a 16 kHz audio segment from a path or in-memory stream.

        §4.D.1 — faster-whisper CTranslate2 inference backend.

        Args:
            audio: Either a filesystem path to a 16 kHz audio file
                (legacy v3.4 path) or a binary file-like object
                (WS4 P3 pipe:0/pipe:1 desktop path).
                ``faster_whisper.WhisperModel.transcribe`` accepts both
                shapes natively.
            language: Optional language hint.

        Returns:
            UTF-8 transcription text.
        """
        if self._model is None:
            self.load_model()

        # §4.D.1 — Transcribe with beam_size=5 default
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        # §4.D.1 — Concatenate all segment texts
        return " ".join(segment.text.strip() for segment in segments)
