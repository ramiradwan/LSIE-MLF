"""
Tests for packages/ml_core/transcription.py — Phase 1 validation.

Verifies TranscriptionEngine against §4.D.1:
model loading, INT8/CUDA config, transcription output.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from packages.ml_core.transcription import TranscriptionEngine


@pytest.fixture()
def mock_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock faster_whisper into sys.modules."""
    mock_fw = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
    return mock_fw


class TestTranscriptionEngine:
    """§4.D.1 — faster-whisper speech transcription."""

    def test_load_model_uses_int8_cuda(
        self, mock_faster_whisper: MagicMock
    ) -> None:
        """§4.D.1 — Loads large-v3 with INT8 on CUDA."""
        engine = TranscriptionEngine()
        engine.load_model()

        mock_faster_whisper.WhisperModel.assert_called_once_with(
            "large-v3",
            device="cuda",
            compute_type="int8",
        )

    def test_transcribe_concatenates_segments(
        self, mock_faster_whisper: MagicMock
    ) -> None:
        """§4.D.1 — Transcription joins all segment texts."""
        seg1 = MagicMock()
        seg1.text = " Hello world "
        seg2 = MagicMock()
        seg2.text = " How are you "

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine()
        result = engine.transcribe("/tmp/audio.raw")

        assert result == "Hello world How are you"
        mock_model.transcribe.assert_called_once_with(
            "/tmp/audio.raw",
            language=None,
            beam_size=5,
            vad_filter=True,
        )

    def test_transcribe_with_language_hint(
        self, mock_faster_whisper: MagicMock
    ) -> None:
        """§4.D.1 — Language hint passed through."""
        seg = MagicMock()
        seg.text = " Moi "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine()
        result = engine.transcribe("/tmp/audio.raw", language="fi")

        assert result == "Moi"
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "fi"

    def test_transcribe_empty_segments(
        self, mock_faster_whisper: MagicMock
    ) -> None:
        """§4.D.1 — Empty segments produce empty string."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine()
        result = engine.transcribe("/tmp/audio.raw")

        assert result == ""

    def test_default_config(self) -> None:
        """§4.D.1 — Default model_size, device, compute_type."""
        engine = TranscriptionEngine()
        assert engine.model_size == "large-v3"
        assert engine.device == "cuda"
        assert engine.compute_type == "int8"

    def test_compute_type_enforced_int8(self) -> None:
        """SPEC-AMEND-001 — compute_type is always int8, cannot be overridden."""
        engine = TranscriptionEngine(model_size="small", device="cpu")
        assert engine.compute_type == "int8"
