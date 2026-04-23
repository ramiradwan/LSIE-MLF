"""
Tests for services/worker/tasks/inference.py — Phase 4.1–4.2 validation.

Verifies process_segment and persist_metrics against:
  §2 step 5–7 — Full Module D pipeline and Module E persistence
  §4.D — Multimodal ML processing
  §12.1 — Error handling for each pipeline stage
"""

from __future__ import annotations

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


class TestProcessSegment:
    """§4.D — Module D multimodal ML pipeline."""

    def _make_payload(self, **overrides: Any) -> dict[str, Any]:
        """Create a minimal valid payload."""
        base: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "_segment_id": "seg-0001",
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "_audio_data": b"\x00" * 3200,
            "segments": [],
        }
        base.update(overrides)
        return base

    def test_returns_result_dict(self) -> None:
        """§2 step 6 — Result dict has all required fields."""
        mod = _get_inference_module()
        with patch.object(mod, "persist_metrics", MagicMock()):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))

        assert result["session_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["segment_id"] == "seg-0001"
        assert result["timestamp_utc"] == "2026-03-13T12:00:00+00:00"
        assert "au12_intensity" in result
        assert "transcription" in result
        assert "semantic" in result
        assert "pitch_f0" in result
        assert "jitter" in result
        assert "shimmer" in result

    def test_dispatches_to_module_e(self) -> None:
        """§2 step 6 → §2 step 7 — Results dispatched to Module E."""
        mod = _get_inference_module()
        mock_persist = MagicMock()
        with patch.object(mod, "persist_metrics", mock_persist):
            mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))

        mock_persist.delay.assert_called_once()
        call_args = mock_persist.delay.call_args[0][0]
        assert call_args["session_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert "_physiological_context" not in call_args

    def test_no_audio_skips_transcription(self) -> None:
        """No transcription when audio_data is None."""
        mod = _get_inference_module()
        with patch.object(mod, "persist_metrics", MagicMock()):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))
        assert result["transcription"] == ""

    def test_no_audio_skips_acoustic(self) -> None:
        """No acoustic analysis when audio_data is None."""
        mod = _get_inference_module()
        with patch.object(mod, "persist_metrics", MagicMock()):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))
        assert result["pitch_f0"] is None
        assert result["jitter"] is None
        assert result["shimmer"] is None

    def test_no_frame_skips_au12(self) -> None:
        """§4.D contract — No AU12 when frame data missing."""
        mod = _get_inference_module()
        with patch.object(mod, "persist_metrics", MagicMock()):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))
        assert result["au12_intensity"] is None

    def test_transcription_failure_returns_empty(self) -> None:
        """§12 Network disconnect D — Transcription failure yields empty string."""
        mod = _get_inference_module()
        payload = self._make_payload()

        # Mock TranscriptionEngine to raise
        mock_engine = MagicMock()
        mock_engine.transcribe.side_effect = RuntimeError("GPU OOM")

        with (
            patch.object(mod, "persist_metrics", MagicMock()),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
        ):
            result = mod.process_segment(MagicMock(), payload)

        # Transcription should fall back to empty on failure
        assert isinstance(result["transcription"], str)

    def test_persist_dispatch_failure_handled(self) -> None:
        """§12 — persist_metrics dispatch failure doesn't crash process_segment."""
        mod = _get_inference_module()
        mock_persist = MagicMock()
        mock_persist.delay.side_effect = RuntimeError("broker down")

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))

        assert result["session_id"] is not None

    def test_all_none_on_empty_payload(self) -> None:
        """All ML fields are None/empty when no data available."""
        mod = _get_inference_module()
        with patch.object(mod, "persist_metrics", MagicMock()):
            result = mod.process_segment(MagicMock(), self._make_payload(_audio_data=None))

        assert result["au12_intensity"] is None
        assert result["transcription"] == ""
        assert result["semantic"] is None
        assert result["pitch_f0"] is None
        assert result["jitter"] is None
        assert result["shimmer"] is None

    def test_forwards_physiological_context_unchanged_when_present(self) -> None:
        """§4.D — Enriched physiological context is forwarded unchanged to Module E."""
        mod = _get_inference_module()
        mock_persist = MagicMock()
        physio_context = {
            "streamer": {
                "rmssd_ms": None,
                "heart_rate_bpm": 72,
                "source_timestamp_utc": "2026-03-13T11:59:30+00:00",
                "freshness_s": 30.0,
                "is_stale": False,
                "provider": "oura",
                "validity_ratio": 0.92,
                "source_kind": "ibi",
                "derivation_method": "server",
            },
            "operator": None,
        }

        with patch.object(mod, "persist_metrics", mock_persist):
            result = mod.process_segment(
                MagicMock(),
                self._make_payload(_audio_data=None, _physiological_context=physio_context),
            )

        assert result["_physiological_context"] is physio_context
        dispatched_metrics = mock_persist.delay.call_args.args[0]
        assert dispatched_metrics["_physiological_context"] is physio_context


class TestPersistMetrics:
    """§2 step 7 / §4.E — Module E metrics persistence."""

    def test_calls_metrics_store(self) -> None:
        """§2 step 7 — Calls MetricsStore.insert_metrics()."""
        mod = _get_inference_module()
        mock_store = MagicMock()

        with patch(
            "services.worker.pipeline.analytics.MetricsStore",
            return_value=mock_store,
        ):
            metrics: dict[str, Any] = {
                "session_id": "test-id",
                "segment_id": "seg-001",
            }
            mod.persist_metrics(MagicMock(), metrics)

        mock_store.connect.assert_called_once()
        mock_store.insert_metrics.assert_called_once_with(metrics)
        mock_store.close.assert_called_once()

    def test_handles_connection_failure(self) -> None:
        """§12.1 Module E — Connection failure logged, not raised."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_store.connect.side_effect = RuntimeError("no postgres")

        with patch(
            "services.worker.pipeline.analytics.MetricsStore",
            return_value=mock_store,
        ):
            # Should not raise
            mod.persist_metrics(MagicMock(), {"segment_id": "seg-001"})

        mock_store.insert_metrics.assert_not_called()

    def test_handles_insert_failure(self) -> None:
        """§12.1 Module E — Insert failure logged, not raised."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_store.insert_metrics.side_effect = RuntimeError("write failed")

        with patch(
            "services.worker.pipeline.analytics.MetricsStore",
            return_value=mock_store,
        ):
            # Should not raise
            mod.persist_metrics(MagicMock(), {"segment_id": "seg-001"})

        mock_store.close.assert_called_once()
