"""
Tests for services/worker/tasks/inference.py — Phase 3.5 v3.0 gap-fix supplement.

Supplements test_inference.py with coverage for:
  Gap 1 fix: _FORWARD_FIELDS forwarding to persist_metrics
  Gap 2 fix: base64 decode of _audio_data/_frame_data from JSON transport
"""

from __future__ import annotations

import base64
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch


def _get_inference_module() -> Any:
    """Import inference module with mocked celery decorator."""
    mock_app = MagicMock()
    mock_app.task.return_value = lambda f: f

    with patch("services.worker.celery_app.celery_app", mock_app):
        mod_name = "services.worker.tasks.inference"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return importlib.import_module(mod_name)


def _make_payload(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid payload with no audio/frame data."""
    base: dict[str, Any] = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "_segment_id": "seg-0001",
        "timestamp_utc": "2026-03-13T12:00:00+00:00",
        "_audio_data": None,
        "segments": [],
    }
    base.update(overrides)
    return base


def _assert_null_acoustic_contract(payload: dict[str, Any]) -> None:
    """Assert the deterministic v3.3 null acoustic payload at the D→E boundary."""
    assert payload["f0_valid_measure"] is False
    assert payload["f0_valid_baseline"] is False
    assert payload["perturbation_valid_measure"] is False
    assert payload["perturbation_valid_baseline"] is False
    assert payload["voiced_coverage_measure_s"] == 0.0
    assert payload["voiced_coverage_baseline_s"] == 0.0
    assert payload["f0_mean_measure_hz"] is None
    assert payload["f0_mean_baseline_hz"] is None
    assert payload["f0_delta_semitones"] is None
    assert payload["jitter_mean_measure"] is None
    assert payload["jitter_mean_baseline"] is None
    assert payload["jitter_delta"] is None
    assert payload["shimmer_mean_measure"] is None
    assert payload["shimmer_mean_baseline"] is None
    assert payload["shimmer_delta"] is None
    assert payload.get("pitch_f0") is None
    assert payload.get("jitter") is None
    assert payload.get("shimmer") is None


class TestForwardFields:
    """Gap 1 fix: _FORWARD_FIELDS forwarding from input to persist_metrics dispatch."""

    def test_all_forward_fields_present_in_output(self) -> None:
        """All six _FORWARD_FIELDS are forwarded when present in input."""
        mod = _get_inference_module()
        forward_data = {
            "_active_arm": "greeting_A",
            "_experiment_id": "greeting_line_v1",
            "_expected_greeting": "Hello, welcome!",
            "_au12_series": [{"timestamp_s": 0.0, "intensity": 0.5}],
            "_stimulus_time": 15.0,
            "_x_max": 0.8,
        }
        payload = _make_payload(**forward_data)
        mock_persist = MagicMock()

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(MagicMock(), payload)

        # Verify fields are in the result dict
        for key in mod._FORWARD_FIELDS:
            assert key in result, f"Missing forward field: {key}"
            assert result[key] == forward_data[key]
        _assert_null_acoustic_contract(result)

        # Verify persist_metrics receives them
        mock_persist.delay.assert_called_once()
        dispatched = mock_persist.delay.call_args[0][0]
        for key in mod._FORWARD_FIELDS:
            assert key in dispatched, f"Missing in persist dispatch: {key}"
        _assert_null_acoustic_contract(dispatched)

    def test_missing_forward_fields_not_added(self) -> None:
        """Fields not present in input are NOT added to output."""
        mod = _get_inference_module()
        # Only provide one forward field
        payload = _make_payload(_active_arm="greeting_A")
        mock_persist = MagicMock()

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(MagicMock(), payload)

        assert "_active_arm" in result
        assert "_experiment_id" not in result
        assert "_expected_greeting" not in result
        assert "_au12_series" not in result
        assert "_stimulus_time" not in result
        assert "_x_max" not in result
        _assert_null_acoustic_contract(result)

    def test_partial_forward_fields(self) -> None:
        """Only present forward fields are forwarded, others absent."""
        mod = _get_inference_module()
        payload = _make_payload(
            _active_arm="arm_B",
            _experiment_id="exp_v2",
            _stimulus_time=10.0,
        )
        mock_persist = MagicMock()

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(MagicMock(), payload)

        assert result["_active_arm"] == "arm_B"
        assert result["_experiment_id"] == "exp_v2"
        assert result["_stimulus_time"] == 10.0
        assert "_expected_greeting" not in result
        assert "_au12_series" not in result
        assert "_x_max" not in result
        _assert_null_acoustic_contract(result)


class TestBase64Decode:
    """Gap 2 fix: base64 decode of binary fields from Celery JSON transport."""

    def test_base64_audio_decoded_before_pipeline(self) -> None:
        """base64-encoded _audio_data is decoded to bytes before ML processing."""
        mod = _get_inference_module()
        raw_audio = b"\x00\x01\x02\x03" * 800  # 3200 bytes of PCM
        b64_audio = base64.b64encode(raw_audio).decode("ascii")

        payload = _make_payload(_audio_data=b64_audio)
        mock_persist = MagicMock()

        # Mock the transcription pipeline to verify it receives bytes
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            mod.process_segment(MagicMock(), payload)

            # The raw PCM bytes should have been written to the temp file
            mock_file.write.assert_called_once_with(raw_audio)

    def test_none_audio_passes_through(self) -> None:
        """None _audio_data passes through decode unchanged."""
        mod = _get_inference_module()
        payload = _make_payload(_audio_data=None)
        mock_persist = MagicMock()

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(MagicMock(), payload)

        # No transcription without audio
        assert result["transcription"] == ""
        _assert_null_acoustic_contract(result)

    def test_raw_bytes_audio_passes_through(self) -> None:
        """bytes _audio_data (already decoded) passes through unchanged."""
        mod = _get_inference_module()
        raw_audio = b"\x00" * 3200
        payload = _make_payload(_audio_data=raw_audio)
        mock_persist = MagicMock()

        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "test"

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            mod.process_segment(MagicMock(), payload)

            # Original bytes should pass through serialization.decode_bytes_fields
            # (bytes is not str, so decode is a no-op) and be written as-is
            mock_file.write.assert_called_once_with(raw_audio)
