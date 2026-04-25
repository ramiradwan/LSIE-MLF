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
from datetime import datetime
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


def _assert_null_acoustic_contract(payload: dict[str, Any]) -> None:
    """Assert the deterministic v3.3 no-audio/null-stimulus acoustic payload."""
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
    # Legacy scalar compatibility fields remain optional/deprecated.
    assert payload.get("pitch_f0") is None
    assert payload.get("jitter") is None
    assert payload.get("shimmer") is None


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
        assert "f0_valid_measure" in result
        assert "f0_valid_baseline" in result
        assert "perturbation_valid_measure" in result
        assert "perturbation_valid_baseline" in result
        assert "voiced_coverage_measure_s" in result
        assert "voiced_coverage_baseline_s" in result
        assert "f0_mean_measure_hz" in result
        assert "f0_mean_baseline_hz" in result
        assert "f0_delta_semitones" in result
        assert "jitter_mean_measure" in result
        assert "jitter_mean_baseline" in result
        assert "jitter_delta" in result
        assert "shimmer_mean_measure" in result
        assert "shimmer_mean_baseline" in result
        assert "shimmer_delta" in result
        _assert_null_acoustic_contract(result)

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
        _assert_null_acoustic_contract(call_args)

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
        _assert_null_acoustic_contract(result)

    def test_null_stimulus_time_skips_acoustic_even_with_audio(self) -> None:
        """Null _stimulus_time forces default acoustic output despite audio bytes."""
        mod = _get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer") as mock_acoustic_cls,
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(
                MagicMock(),
                self._make_payload(_stimulus_time=None),
            )

        _assert_null_acoustic_contract(result)
        dispatched_metrics = mock_persist.delay.call_args.args[0]
        _assert_null_acoustic_contract(dispatched_metrics)
        mock_acoustic_cls.assert_not_called()

    def test_stimulus_locked_acoustic_analysis_uses_timing_context(self) -> None:
        """§4.D.3 / §7D — AcousticAnalyzer receives 16 kHz stimulus-locked timing."""
        from packages.ml_core.acoustic import AcousticMetrics

        mod = _get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        mock_acoustic = MagicMock()
        mock_acoustic.analyze.return_value = AcousticMetrics(
            pitch_f0=210.0,
            jitter=0.01,
            shimmer=0.02,
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=2.0,
            voiced_coverage_baseline_s=1.5,
            f0_mean_measure_hz=220.0,
            f0_mean_baseline_hz=180.0,
            f0_delta_semitones=3.468,
            jitter_mean_measure=0.011,
            jitter_mean_baseline=0.009,
            jitter_delta=0.002,
            shimmer_mean_measure=0.021,
            shimmer_mean_baseline=0.018,
            shimmer_delta=0.003,
        )
        payload = self._make_payload(_stimulus_time=100.0)
        expected_segment_start_time_s = datetime.fromisoformat(
            payload["timestamp_utc"]
        ).timestamp() - (len(payload["_audio_data"]) / (16000 * 2))

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", return_value=mock_acoustic),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(MagicMock(), payload)

        mock_acoustic.analyze.assert_called_once_with(
            payload["_audio_data"],
            sample_rate=16000,
            stimulus_time_s=100.0,
            segment_start_time_s=expected_segment_start_time_s,
        )
        assert result["f0_valid_measure"] is True
        assert result["f0_valid_baseline"] is True
        assert result["f0_mean_measure_hz"] == 220.0
        assert result["f0_mean_baseline_hz"] == 180.0
        assert result["jitter_mean_measure"] == 0.011
        assert result["shimmer_mean_measure"] == 0.021
        assert result["pitch_f0"] == 210.0
        assert result["jitter"] == 0.01
        assert result["shimmer"] == 0.02
        dispatched_metrics = mock_persist.delay.call_args.args[0]
        assert dispatched_metrics["f0_valid_measure"] is True
        assert dispatched_metrics["pitch_f0"] == 210.0
        assert dispatched_metrics["jitter"] == 0.01
        assert dispatched_metrics["shimmer"] == 0.02

    def test_acoustic_invalidity_returns_default_payload_without_failure(self) -> None:
        """§4.D.contract / §12.4 — local acoustic failures degrade to false/null outputs."""
        mod = _get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        mock_acoustic = MagicMock()
        mock_acoustic.analyze.side_effect = RuntimeError("praat extraction failed")

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", return_value=mock_acoustic),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(MagicMock(), self._make_payload(_stimulus_time=100.0))

        _assert_null_acoustic_contract(result)
        dispatched_metrics = mock_persist.delay.call_args.args[0]
        _assert_null_acoustic_contract(dispatched_metrics)

    def test_canonical_invalidity_payloads_dispatch_without_crashing(self) -> None:
        """Silence, sparse voicing, and perturbation invalidity stay local to acoustics."""
        from packages.ml_core.acoustic import AcousticMetrics

        mod = _get_inference_module()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        expected_cases: list[tuple[str, AcousticMetrics, dict[str, Any]]] = [
            (
                "silence",
                AcousticMetrics(),
                {
                    "f0_valid_measure": False,
                    "f0_valid_baseline": False,
                    "perturbation_valid_measure": False,
                    "perturbation_valid_baseline": False,
                    "voiced_coverage_measure_s": 0.0,
                    "voiced_coverage_baseline_s": 0.0,
                    "f0_mean_measure_hz": None,
                    "f0_mean_baseline_hz": None,
                    "f0_delta_semitones": None,
                    "jitter_mean_measure": None,
                    "jitter_mean_baseline": None,
                    "jitter_delta": None,
                    "shimmer_mean_measure": None,
                    "shimmer_mean_baseline": None,
                    "shimmer_delta": None,
                    "pitch_f0": None,
                    "jitter": None,
                    "shimmer": None,
                },
            ),
            (
                "sparse_voicing",
                AcousticMetrics(
                    pitch_f0=205.0,
                    jitter=0.011,
                    shimmer=0.019,
                    voiced_coverage_measure_s=0.6,
                    voiced_coverage_baseline_s=0.8,
                ),
                {
                    "f0_valid_measure": False,
                    "f0_valid_baseline": False,
                    "perturbation_valid_measure": False,
                    "perturbation_valid_baseline": False,
                    "voiced_coverage_measure_s": 0.6,
                    "voiced_coverage_baseline_s": 0.8,
                    "f0_mean_measure_hz": None,
                    "f0_mean_baseline_hz": None,
                    "f0_delta_semitones": None,
                    "jitter_mean_measure": None,
                    "jitter_mean_baseline": None,
                    "jitter_delta": None,
                    "shimmer_mean_measure": None,
                    "shimmer_mean_baseline": None,
                    "shimmer_delta": None,
                    "pitch_f0": 205.0,
                    "jitter": 0.011,
                    "shimmer": 0.019,
                },
            ),
            (
                "perturbation_invalidity",
                AcousticMetrics(
                    pitch_f0=210.0,
                    jitter=0.010,
                    shimmer=0.020,
                    f0_valid_measure=True,
                    f0_valid_baseline=True,
                    voiced_coverage_measure_s=2.4,
                    voiced_coverage_baseline_s=2.0,
                    f0_mean_measure_hz=220.0,
                    f0_mean_baseline_hz=180.0,
                    f0_delta_semitones=3.468,
                ),
                {
                    "f0_valid_measure": True,
                    "f0_valid_baseline": True,
                    "perturbation_valid_measure": False,
                    "perturbation_valid_baseline": False,
                    "voiced_coverage_measure_s": 2.4,
                    "voiced_coverage_baseline_s": 2.0,
                    "f0_mean_measure_hz": 220.0,
                    "f0_mean_baseline_hz": 180.0,
                    "f0_delta_semitones": 3.468,
                    "jitter_mean_measure": None,
                    "jitter_mean_baseline": None,
                    "jitter_delta": None,
                    "shimmer_mean_measure": None,
                    "shimmer_mean_baseline": None,
                    "shimmer_delta": None,
                    "pitch_f0": 210.0,
                    "jitter": 0.010,
                    "shimmer": 0.020,
                },
            ),
        ]

        for label, metrics, expected in expected_cases:
            mock_persist = MagicMock()
            mock_acoustic = MagicMock()
            mock_acoustic.analyze.return_value = metrics

            with (
                patch.object(mod, "persist_metrics", mock_persist),
                patch(
                    "packages.ml_core.transcription.TranscriptionEngine",
                    return_value=mock_engine,
                ),
                patch(
                    "packages.ml_core.acoustic.AcousticAnalyzer",
                    return_value=mock_acoustic,
                ),
                patch("subprocess.run"),
                patch("os.remove"),
                patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
            ):
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = "/tmp/test.raw"
                mock_tmpfile.return_value = mock_file

                result = mod.process_segment(MagicMock(), self._make_payload(_stimulus_time=100.0))

            dispatched_metrics = mock_persist.delay.call_args.args[0]
            for key, value in expected.items():
                assert result[key] == value, f"{label}: result[{key}]"
                assert dispatched_metrics[key] == value, f"{label}: dispatch[{key}]"

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
        _assert_null_acoustic_contract(result)

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

    def test_non_finite_acoustic_scalars_sanitized_before_dispatch(self) -> None:
        """NaN/Infinity acoustic outputs are converted to null/0.0 before JSON handoff."""
        from packages.ml_core.acoustic import AcousticMetrics

        mod = _get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello"
        mock_acoustic = MagicMock()
        mock_acoustic.analyze.return_value = AcousticMetrics(
            pitch_f0=float("nan"),
            jitter=float("inf"),
            shimmer=float("-inf"),
            voiced_coverage_measure_s=float("nan"),
            voiced_coverage_baseline_s=float("inf"),
            f0_mean_measure_hz=float("nan"),
            f0_mean_baseline_hz=float("inf"),
            f0_delta_semitones=float("nan"),
            jitter_mean_measure=float("nan"),
            jitter_mean_baseline=float("inf"),
            jitter_delta=float("nan"),
            shimmer_mean_measure=float("nan"),
            shimmer_mean_baseline=float("inf"),
            shimmer_delta=float("-inf"),
        )

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", return_value=mock_acoustic),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(MagicMock(), self._make_payload(_stimulus_time=100.0))

        _assert_null_acoustic_contract(result)
        dispatched_metrics = mock_persist.delay.call_args.args[0]
        _assert_null_acoustic_contract(dispatched_metrics)

    def test_default_acoustic_payload_freezes_public_field_names(self) -> None:
        """§2.6 / §4.D.contract — D→E payload field names stay stable."""
        mod = _get_inference_module()
        payload = mod._default_acoustic_payload()

        assert tuple(payload.keys()) == (
            "f0_valid_measure",
            "f0_valid_baseline",
            "perturbation_valid_measure",
            "perturbation_valid_baseline",
            "voiced_coverage_measure_s",
            "voiced_coverage_baseline_s",
            "f0_mean_measure_hz",
            "f0_mean_baseline_hz",
            "f0_delta_semitones",
            "jitter_mean_measure",
            "jitter_mean_baseline",
            "jitter_delta",
            "shimmer_mean_measure",
            "shimmer_mean_baseline",
            "shimmer_delta",
            "pitch_f0",
            "jitter",
            "shimmer",
        )
        for forbidden_field in (
            "gated_reward",
            "p90_intensity",
            "semantic_gate",
            "reward_z",
            "z_score",
            "z_scored",
        ):
            assert forbidden_field not in payload


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

    def test_sanitizes_non_finite_metrics_before_store_insert(self) -> None:
        """Persist path converts non-finite numerics to JSON-null-compatible values."""
        mod = _get_inference_module()
        mock_store = MagicMock()

        with patch(
            "services.worker.pipeline.analytics.MetricsStore",
            return_value=mock_store,
        ):
            mod.persist_metrics(
                MagicMock(),
                {
                    "session_id": "test-id",
                    "segment_id": "seg-001",
                    "pitch_f0": float("nan"),
                    "jitter": float("inf"),
                    "semantic": {"confidence_score": float("nan")},
                },
            )

        inserted = mock_store.insert_metrics.call_args.args[0]
        assert inserted["pitch_f0"] is None
        assert inserted["jitter"] is None
        assert inserted["semantic"]["confidence_score"] is None

    def _make_reward_metrics(self, **overrides: Any) -> dict[str, Any]:
        """Create a reward-eligible metrics payload with a stable AU12 series."""
        base: dict[str, Any] = {
            "session_id": "test-session",
            "segment_id": "seg-reward",
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "semantic": {"is_match": True, "confidence_score": 0.84},
            "_active_arm": "arm-a",
            "_experiment_id": "exp-1",
            "_au12_series": [
                {"timestamp_s": 95.2, "intensity": 0.11},
                {"timestamp_s": 96.0, "intensity": 0.12},
                {"timestamp_s": 97.5, "intensity": 0.10},
                {"timestamp_s": 100.5, "intensity": 0.20},
                {"timestamp_s": 100.8, "intensity": 0.24},
                {"timestamp_s": 101.1, "intensity": 0.28},
                {"timestamp_s": 101.4, "intensity": 0.32},
                {"timestamp_s": 101.7, "intensity": 0.36},
                {"timestamp_s": 102.0, "intensity": 0.40},
                {"timestamp_s": 102.3, "intensity": 0.44},
                {"timestamp_s": 102.6, "intensity": 0.48},
                {"timestamp_s": 102.9, "intensity": 0.52},
                {"timestamp_s": 103.2, "intensity": 0.56},
                {"timestamp_s": 103.5, "intensity": 0.60},
                {"timestamp_s": 103.8, "intensity": 0.64},
            ],
            "_stimulus_time": 100.0,
            "_x_max": None,
        }
        base.update(overrides)
        return base

    def _make_observational_acoustic_payload(self, **overrides: Any) -> dict[str, Any]:
        """Create a populated observational acoustic payload for regression tests."""
        payload: dict[str, Any] = {
            "f0_valid_measure": True,
            "f0_valid_baseline": True,
            "perturbation_valid_measure": True,
            "perturbation_valid_baseline": True,
            "voiced_coverage_measure_s": 2.4,
            "voiced_coverage_baseline_s": 1.8,
            "f0_mean_measure_hz": 220.0,
            "f0_mean_baseline_hz": 180.0,
            "f0_delta_semitones": 3.468,
            "jitter_mean_measure": 0.011,
            "jitter_mean_baseline": 0.009,
            "jitter_delta": 0.002,
            "shimmer_mean_measure": 0.021,
            "shimmer_mean_baseline": 0.018,
            "shimmer_delta": 0.003,
            "pitch_f0": 210.0,
            "jitter": 0.010,
            "shimmer": 0.020,
        }
        payload.update(overrides)
        return payload

    def test_reward_update_is_invariant_to_observational_acoustic_payloads(self) -> None:
        """§7B / §7D — populated acoustic analytics never alter AU12 reward inputs."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_engine = MagicMock()
        base_metrics = self._make_reward_metrics()
        acoustic_payload = self._make_observational_acoustic_payload()
        metrics_with_acoustics = {**base_metrics, **acoustic_payload}

        from services.worker.pipeline.reward import TimestampedAU12
        from services.worker.pipeline.reward import compute_reward as real_compute_reward

        expected_series = [TimestampedAU12(**point) for point in base_metrics["_au12_series"]]
        expected_reward_kwargs = {
            "au12_series": expected_series,
            "stimulus_time_s": 100.0,
            "is_match": True,
            "confidence_score": 0.84,
            "x_max": None,
        }
        expected_reward = real_compute_reward(
            au12_series=expected_series,
            stimulus_time_s=100.0,
            is_match=True,
            confidence_score=0.84,
            x_max=None,
        )

        with (
            patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
            patch(
                "services.worker.pipeline.analytics.ThompsonSamplingEngine",
                return_value=mock_engine,
            ),
            patch(
                "services.worker.pipeline.reward.compute_reward",
                side_effect=real_compute_reward,
            ) as mock_compute_reward,
            patch.object(mod, "_log_encounter", MagicMock()),
        ):
            mod.persist_metrics(MagicMock(), base_metrics)
            mod.persist_metrics(MagicMock(), metrics_with_acoustics)

        assert [reward_call.kwargs for reward_call in mock_compute_reward.call_args_list] == [
            expected_reward_kwargs,
            expected_reward_kwargs,
        ]
        assert [update_call.args for update_call in mock_engine.update.call_args_list] == [
            ("exp-1", "arm-a", expected_reward.gated_reward),
            ("exp-1", "arm-a", expected_reward.gated_reward),
        ]
        for reward_call in mock_compute_reward.call_args_list:
            for acoustic_field in acoustic_payload:
                assert acoustic_field not in reward_call.kwargs

        persisted_without_acoustics = mock_store.insert_metrics.call_args_list[0].args[0]
        persisted_with_acoustics = mock_store.insert_metrics.call_args_list[1].args[0]
        for acoustic_field, value in acoustic_payload.items():
            assert acoustic_field not in persisted_without_acoustics
            assert persisted_with_acoustics[acoustic_field] == value

    def test_semantic_gate_reward_stays_closed_even_with_observational_acoustics(self) -> None:
        """§7B / §7D — acoustic payloads cannot reopen the semantic reward gate."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_engine = MagicMock()
        base_metrics = self._make_reward_metrics(
            semantic={"is_match": False, "confidence_score": 0.84},
        )
        acoustic_payload = self._make_observational_acoustic_payload(
            f0_delta_semitones=9.9,
            jitter_delta=0.25,
            shimmer_delta=0.5,
            pitch_f0=420.0,
        )
        metrics_with_acoustics = {**base_metrics, **acoustic_payload}

        from services.worker.pipeline.reward import TimestampedAU12
        from services.worker.pipeline.reward import compute_reward as real_compute_reward

        expected_series = [TimestampedAU12(**point) for point in base_metrics["_au12_series"]]
        expected_reward_kwargs = {
            "au12_series": expected_series,
            "stimulus_time_s": 100.0,
            "is_match": False,
            "confidence_score": 0.84,
            "x_max": None,
        }
        expected_reward = real_compute_reward(
            au12_series=expected_series,
            stimulus_time_s=100.0,
            is_match=False,
            confidence_score=0.84,
            x_max=None,
        )
        assert expected_reward.gated_reward == 0.0
        assert expected_reward.semantic_gate == 0

        with (
            patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
            patch(
                "services.worker.pipeline.analytics.ThompsonSamplingEngine",
                return_value=mock_engine,
            ),
            patch(
                "services.worker.pipeline.reward.compute_reward",
                side_effect=real_compute_reward,
            ) as mock_compute_reward,
            patch.object(mod, "_log_encounter", MagicMock()),
        ):
            mod.persist_metrics(MagicMock(), base_metrics)
            mod.persist_metrics(MagicMock(), metrics_with_acoustics)

        assert [reward_call.kwargs for reward_call in mock_compute_reward.call_args_list] == [
            expected_reward_kwargs,
            expected_reward_kwargs,
        ]
        assert [update_call.args for update_call in mock_engine.update.call_args_list] == [
            ("exp-1", "arm-a", 0.0),
            ("exp-1", "arm-a", 0.0),
        ]
        for reward_call in mock_compute_reward.call_args_list:
            for acoustic_field in acoustic_payload:
                assert acoustic_field not in reward_call.kwargs

        persisted_with_acoustics = mock_store.insert_metrics.call_args_list[1].args[0]
        for acoustic_field, value in acoustic_payload.items():
            assert persisted_with_acoustics[acoustic_field] == value
