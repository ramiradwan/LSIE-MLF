"""
Tests for packages/ml_core/transcription.py — Phase 1 validation.

Verifies TranscriptionEngine against §4.D.1:
model loading, INT8/CUDA config, transcription output. WS2 P2 adds
the `resolve_speech_device` resolver that picks the device per
v4.0 §11.x — the existing engine tests pin `device="cuda"` explicitly
so they exercise the WhisperModel-call shape independently of the
resolver's host-dependent return.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from packages.ml_core import transcription
from packages.ml_core.transcription import TranscriptionEngine, resolve_speech_device


@pytest.fixture()
def mock_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock faster_whisper into sys.modules."""
    mock_fw = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
    return mock_fw


@pytest.fixture()
def clear_speech_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop LSIE_DEV_FORCE_CPU_SPEECH so resolver tests start clean."""
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)


class TestTranscriptionEngine:
    """§4.D.1 — faster-whisper speech transcription."""

    def test_load_model_uses_int8_cuda(self, mock_faster_whisper: MagicMock) -> None:
        """§4.D.1 — Loads large-v3 with INT8 on CUDA."""
        engine = TranscriptionEngine(device="cuda")
        engine.load_model()

        mock_faster_whisper.WhisperModel.assert_called_once_with(
            "large-v3",
            device="cuda",
            compute_type="int8",
        )

    def test_transcribe_concatenates_segments(self, mock_faster_whisper: MagicMock) -> None:
        """§4.D.1 — Transcription joins all segment texts."""
        seg1 = MagicMock()
        seg1.text = " Hello world "
        seg2 = MagicMock()
        seg2.text = " How are you "

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine(device="cuda")
        result = engine.transcribe("/tmp/audio.raw")

        assert result == "Hello world How are you"
        mock_model.transcribe.assert_called_once_with(
            "/tmp/audio.raw",
            language=None,
            beam_size=5,
            vad_filter=True,
        )

    def test_transcribe_with_language_hint(self, mock_faster_whisper: MagicMock) -> None:
        """§4.D.1 — Language hint passed through."""
        seg = MagicMock()
        seg.text = " Moi "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine(device="cuda")
        result = engine.transcribe("/tmp/audio.raw", language="fi")

        assert result == "Moi"
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "fi"

    def test_transcribe_empty_segments(self, mock_faster_whisper: MagicMock) -> None:
        """§4.D.1 — Empty segments produce empty string."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        mock_faster_whisper.WhisperModel.return_value = mock_model

        engine = TranscriptionEngine(device="cuda")
        result = engine.transcribe("/tmp/audio.raw")

        assert result == ""

    def test_default_device_routes_through_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """device=None defers to resolve_speech_device() (v4.0 §11.x)."""
        monkeypatch.setattr(transcription, "resolve_speech_device", lambda: "cpu")
        engine = TranscriptionEngine()
        assert engine.device == "cpu"

        monkeypatch.setattr(transcription, "resolve_speech_device", lambda: "cuda")
        engine = TranscriptionEngine()
        assert engine.device == "cuda"

    def test_explicit_device_overrides_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit device kwarg short-circuits the resolver call."""
        called: list[bool] = []

        def _fail() -> str:
            called.append(True)
            return "cpu"

        monkeypatch.setattr(transcription, "resolve_speech_device", _fail)
        engine = TranscriptionEngine(device="cuda")
        assert engine.device == "cuda"
        assert called == []

    def test_default_config_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default model_size + compute_type are stable across hosts."""
        monkeypatch.setattr(transcription, "resolve_speech_device", lambda: "cuda")
        engine = TranscriptionEngine()
        assert engine.model_size == "large-v3"
        assert engine.compute_type == "int8"

    def test_compute_type_enforced_int8(self) -> None:
        """v4.0 §11.x — compute_type is always int8, cannot be overridden.

        Locked at the class level so the production Turing+ GPU path
        (DP4A + INT8 Tensor Cores) and the Pascal CPU developer escape
        hatch (LSIE_DEV_FORCE_CPU_SPEECH) share the same accuracy and
        latency contract.
        """
        engine = TranscriptionEngine(model_size="small", device="cpu")
        assert engine.compute_type == "int8"


class TestResolveSpeechDevice:
    """v4.0 §11.x — device resolver feeds TranscriptionEngine.__init__."""

    def test_env_override_forces_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_speech_env: None,
    ) -> None:
        monkeypatch.setenv("LSIE_DEV_FORCE_CPU_SPEECH", "1")
        # Must short-circuit before nvidia-smi is even consulted.
        monkeypatch.setattr(
            transcription,
            "query_max_compute_capability",
            lambda: pytest.fail("resolver must not query nvidia-smi when env override is set"),
        )
        assert resolve_speech_device() == "cpu"

    def test_env_override_only_triggers_on_exact_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_speech_env: None,
    ) -> None:
        for sneaky in ("0", "true", "True", "yes", "  1  ", ""):
            monkeypatch.setenv("LSIE_DEV_FORCE_CPU_SPEECH", sneaky)
            monkeypatch.setattr(
                transcription,
                "query_max_compute_capability",
                lambda: 7.5,
            )
            assert resolve_speech_device() == "cuda", (
                f"value {sneaky!r} unexpectedly triggered CPU override"
            )

    def test_turing_or_newer_routes_to_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_speech_env: None,
    ) -> None:
        for cap in (7.5, 8.0, 8.6, 8.9, 9.0):
            monkeypatch.setattr(transcription, "query_max_compute_capability", lambda c=cap: c)
            assert resolve_speech_device() == "cuda", f"cap {cap} should route to cuda"

    def test_pascal_routes_to_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_speech_env: None,
    ) -> None:
        monkeypatch.setattr(transcription, "query_max_compute_capability", lambda: 6.1)
        assert resolve_speech_device() == "cpu"

    def test_no_gpu_routes_to_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_speech_env: None,
    ) -> None:
        monkeypatch.setattr(transcription, "query_max_compute_capability", lambda: None)
        assert resolve_speech_device() == "cpu"
