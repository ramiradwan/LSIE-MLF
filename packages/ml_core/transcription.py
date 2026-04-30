"""
faster-whisper transcription wrapper for Module D audio input (§4.D.1).

The module lazy-loads the CTranslate2-backed Whisper model and transcribes
16 kHz audio files to UTF-8 text from a caller-provided audio path. Runtime is
constrained to CUDA with cuDNN 8 and INT8 compute for the target ML Worker
topology (§9, §10.2); compute type is not operator-configurable.
"""

from __future__ import annotations

from typing import Any


class TranscriptionEngine:
    """
    Transcribe 16 kHz audio segments with faster-whisper.

    Accepts a model size and CUDA device identifier, lazy-loads the configured
    Whisper model, and produces concatenated UTF-8 transcript text for a
    caller-provided audio path. It does not expose a compute_type override,
    perform semantic matching, or persist transcripts; INT8 is fixed for the
    supported CUDA/cuDNN ML Worker runtime.
    """

    # SPEC-AMEND-001: compute_type is hardcoded to "int8" to enforce dp4a
    # vectorization on Pascal (SM 6.1) hardware with cuDNN 8. FP16 is not
    # available on GTX 1080 Ti; allowing overrides would cause silent fallback.
    _COMPUTE_TYPE: str = "int8"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = self._COMPUTE_TYPE
        self._model: Any = None  # Lazy-loaded WhisperModel

    def load_model(self) -> None:
        """Load faster-whisper model into GPU memory. Aborts startup on failure (§4.D contract)."""
        from faster_whisper import WhisperModel

        # §4.D.1 — INT8 quantization on CUDA with cuDNN 8 (SPEC-AMEND-001)
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """
        Transcribe a 16 kHz audio segment from a filesystem path.

        §4.D.1 — faster-whisper CTranslate2 inference backend.

        Args:
            audio_path: Path to a PCM s16le 16 kHz audio file.
            language: Optional language hint.

        Returns:
            UTF-8 transcription text.
        """
        if self._model is None:
            self.load_model()

        # §4.D.1 — Transcribe with beam_size=5 default
        segments, _info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        # §4.D.1 — Concatenate all segment texts
        return " ".join(segment.text.strip() for segment in segments)
