"""
Speech Transcription — §4.D.1

Wraps faster-whisper with CTranslate2 inference backend.
Uses INT8 quantization, requires CUDA 12 and cuDNN 8.
SPEC-AMEND-001: compute_type locked to int8 for dp4a on Pascal (SM 6.1).
"""

from __future__ import annotations

from typing import Any


class TranscriptionEngine:
    """
    §4.D.1 — faster-whisper speech transcription engine.

    Loads the large-v3 model with INT8 quantization on CUDA device.
    Transcribes 16 kHz PCM audio segments into text.
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
        Transcribe a 16 kHz audio segment.

        §4.D.1 — faster-whisper CTranslate2 inference backend.

        Args:
            audio_path: Path to PCM s16le 16 kHz audio file or buffer.
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
