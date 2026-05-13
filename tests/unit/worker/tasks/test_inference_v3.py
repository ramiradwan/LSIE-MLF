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
from dataclasses import asdict
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import pytest


def _get_inference_module() -> Any:
    """Import inference module with mocked celery decorator."""
    mock_app = MagicMock()
    mock_app.task.return_value = lambda f: f

    celery_dependency = cast(Any, ModuleType("celery"))
    celery_dependency.Task = type("Task", (), {})
    celery_app_module = cast(Any, ModuleType("services.worker.celery_app"))
    celery_app_module.celery_app = mock_app
    worker_pkg = importlib.import_module("services.worker")

    with (
        patch.dict(
            sys.modules,
            {
                "celery": celery_dependency,
                "services.worker.celery_app": celery_app_module,
            },
        ),
        patch.object(worker_pkg, "celery_app", celery_app_module, create=True),
    ):
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


class TestForwardFields:
    """Gap 1 fix: _FORWARD_FIELDS forwarding from input to persist_metrics dispatch."""

    def test_all_forward_fields_present_in_output(self) -> None:
        """All six _FORWARD_FIELDS are forwarded when present in input."""
        mod = _get_inference_module()
        forward_data = {
            "_active_arm": "greeting_A",
            "_experiment_id": "greeting_line_v1",
            "_stimulus_modality": "spoken_greeting",
            "_stimulus_payload": {"content_type": "text", "text": "Hello, welcome!"},
            "_expected_stimulus_rule": "Deliver the spoken greeting to the creator",
            "_expected_response_rule": "The streamer acknowledges the stimulus.",
            "_stimulus_id": "11111111-1111-4111-8111-111111111111",
            "_response_observation_horizon_s": 5.0,
            "response_inference": {
                "is_match": True,
                "confidence_score": 0.9,
                "registration_status": "observable_response",
                "response_reason_code": "response_semantic_ack",
            },
            "_au12_series": [{"timestamp_s": 0.0, "intensity": 0.5}],
            "_stimulus_time": 15.0,
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
        assert "_stimulus_modality" not in result
        assert "_stimulus_payload" not in result
        assert "_expected_stimulus_rule" not in result
        assert "_expected_response_rule" not in result
        assert "_au12_series" not in result
        assert "_stimulus_time" not in result
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
        assert "_stimulus_modality" not in result
        assert "_stimulus_payload" not in result
        assert "_expected_stimulus_rule" not in result
        assert "_expected_response_rule" not in result
        assert "_au12_series" not in result
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
            patch(
                "packages.ml_core.audio_pipe.pcm_to_wav_bytes",
                return_value=b"RIFFwav",
            ) as mock_pcm_to_wav,
        ):
            mod.process_segment(MagicMock(), payload)

        mock_pcm_to_wav.assert_called_once_with(raw_audio)
        audio_stream = mock_engine.transcribe.call_args.args[0]
        assert hasattr(audio_stream, "read")
        assert audio_stream.read() == b"RIFFwav"

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
            patch(
                "packages.ml_core.audio_pipe.pcm_to_wav_bytes",
                return_value=b"RIFFwav",
            ) as mock_pcm_to_wav,
        ):
            mod.process_segment(MagicMock(), payload)

        mock_pcm_to_wav.assert_called_once_with(raw_audio)
        audio_stream = mock_engine.transcribe.call_args.args[0]
        assert hasattr(audio_stream, "read")
        assert audio_stream.read() == b"RIFFwav"

    @pytest.mark.audit_item("13.27")
    def test_semantic_payload_contains_only_live_channel(self) -> None:
        """§8 — v3 handoff enriches only the live semantic payload."""
        mod = _get_inference_module()
        raw_audio = b"\x00\x01" * 1600
        b64_audio = base64.b64encode(raw_audio).decode("ascii")
        payload = _make_payload(_audio_data=b64_audio)
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "hello welcome"
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = "hello welcome"
        method_version = "lsie-greeting-cross-encoder-v1.0.0+semantic-greeting-calibration-v1.0.0"
        live_semantic = {
            "reasoning": "cross_encoder_high_nonmatch",
            "is_match": False,
            "confidence_score": 0.57,
        }
        expected_live_semantic = {
            **live_semantic,
            "semantic_method": "cross_encoder",
            "semantic_method_version": method_version,
        }
        mock_semantic = MagicMock()
        mock_semantic.evaluate.return_value = live_semantic

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch(
                "packages.ml_core.preprocessing.TextPreprocessor",
                return_value=mock_preprocessor,
            ),
            patch("packages.ml_core.semantic.SemanticEvaluator", return_value=mock_semantic),
            patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFFwav"),
        ):
            result = mod.process_segment(MagicMock(), payload)

        assert "semantic_method" not in live_semantic
        assert "semantic_method_version" not in live_semantic
        assert result["semantic"] == expected_live_semantic
        assert set(result["semantic"]) == {
            "reasoning",
            "is_match",
            "confidence_score",
            "semantic_method",
            "semantic_method_version",
        }
        assert result["semantic"]["is_match"] is False
        assert result["semantic"]["confidence_score"] == 0.57
        dispatched = mock_persist.delay.call_args.args[0]
        assert dispatched["semantic"] == expected_live_semantic
        assert set(dispatched["semantic"]) == set(result["semantic"])

    @pytest.mark.audit_item("13.27")
    def test_semantic_shadow_mode_is_observational_for_reward_and_updates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """§8/§13.27 — Divergent semantic shadow output cannot change live reward/update paths."""
        mod = _get_inference_module()
        from packages.ml_core.semantic import SemanticEvaluator
        from services.worker.pipeline.reward import RewardResult

        raw_audio = b"\x00\x01" * 1600
        b64_audio = base64.b64encode(raw_audio).decode("ascii")
        payload = _make_payload(
            _audio_data=b64_audio,
            _active_arm="arm-a",
            _experiment_id="exp-1",
            _stimulus_modality="spoken_greeting",
            _stimulus_payload={"content_type": "text", "text": "hello welcome"},
            _expected_stimulus_rule="Deliver the spoken greeting to the creator",
            _expected_response_rule="The streamer acknowledges the stimulus.",
            _stimulus_time=100.0,
            _au12_series=[
                {"timestamp_s": 100.5, "intensity": 0.40},
                {"timestamp_s": 101.0, "intensity": 0.70},
            ],
        )

        def _fixed_score(value: float) -> Any:
            def scorer(_expected: str, _actual: str) -> float:
                return value

            return scorer

        def _process_with_shadow_flag(enabled: bool) -> tuple[dict[str, Any], dict[str, Any], Any]:
            if enabled:
                monkeypatch.setenv("SEMANTIC_SHADOW_MODE_ENABLED", "1")
            else:
                monkeypatch.delenv("SEMANTIC_SHADOW_MODE_ENABLED", raising=False)

            semantic_instances: list[SemanticEvaluator] = []

            def evaluator_factory() -> SemanticEvaluator:
                evaluator = SemanticEvaluator(
                    primary_scorer=_fixed_score(0.84),
                    gray_band_fallback_enabled=False,
                    shadow_scorer=_fixed_score(0.12),
                    shadow_mode_enabled=None,
                )
                semantic_instances.append(evaluator)
                return evaluator

            mock_persist = MagicMock()
            mock_engine = MagicMock()
            mock_engine.transcribe.return_value = "hello welcome"
            mock_preprocessor = MagicMock()
            mock_preprocessor.preprocess.return_value = "hello welcome"

            with (
                patch.object(mod, "persist_metrics", mock_persist),
                patch(
                    "packages.ml_core.transcription.TranscriptionEngine",
                    return_value=mock_engine,
                ),
                patch(
                    "packages.ml_core.preprocessing.TextPreprocessor",
                    return_value=mock_preprocessor,
                ),
                patch("packages.ml_core.semantic.SemanticEvaluator", side_effect=evaluator_factory),
                patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", return_value=b"RIFFwav"),
            ):
                result = mod.process_segment(MagicMock(), dict(payload))

            assert len(semantic_instances) == 1
            dispatched = mock_persist.delay.call_args.args[0]
            return result, dispatched, semantic_instances[0]

        live_result, live_dispatched, live_evaluator = _process_with_shadow_flag(False)
        shadow_result, shadow_dispatched, shadow_evaluator = _process_with_shadow_flag(True)

        expected_live_semantic = {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.84,
            "semantic_method": "cross_encoder",
            "semantic_method_version": (
                "lsie-greeting-cross-encoder-v1.0.0+semantic-greeting-calibration-v1.0.0"
            ),
        }
        assert live_evaluator.last_shadow_semantic is None
        assert shadow_evaluator.last_shadow_semantic == {
            "reasoning": "shadow_candidate_nonmatch",
            "is_match": False,
            "confidence_score": 0.12,
        }
        assert live_result["semantic"] == expected_live_semantic
        assert shadow_result["semantic"] == expected_live_semantic
        assert live_dispatched["semantic"] == expected_live_semantic
        assert shadow_dispatched["semantic"] == expected_live_semantic
        assert "_semantic_shadow" not in shadow_result
        assert "semantic_shadow" not in shadow_result
        assert "_semantic_shadow" not in shadow_dispatched
        assert "semantic_shadow" not in shadow_dispatched

        reward_result = RewardResult(
            gated_reward=0.42,
            p90_intensity=0.70,
            semantic_gate=1,
            n_frames_in_window=2,
            au12_baseline_pre=None,
        )
        mock_store = MagicMock()
        mock_store.compute_comodulation.return_value = None
        mock_engine = MagicMock()

        with (
            patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
            patch(
                "services.worker.pipeline.analytics.ThompsonSamplingEngine",
                return_value=mock_engine,
            ),
            patch(
                "services.worker.pipeline.reward.compute_reward",
                return_value=reward_result,
            ) as mock_compute_reward,
            patch.object(mod, "_log_encounter", MagicMock()),
        ):
            mod.persist_metrics(MagicMock(), live_result)
            mod.persist_metrics(MagicMock(), shadow_result)

        assert len(mock_compute_reward.call_args_list) == 2
        assert (
            mock_compute_reward.call_args_list[0].kwargs
            == mock_compute_reward.call_args_list[1].kwargs
        )
        reward_kwargs = mock_compute_reward.call_args_list[1].kwargs
        assert set(reward_kwargs) == {"au12_series", "stimulus_time_s", "is_match"}
        assert reward_kwargs["is_match"] is True
        assert "confidence_score" not in reward_kwargs
        assert "semantic_shadow" not in reward_kwargs
        assert mock_engine.update.call_args_list == [
            call("exp-1", "arm-a", 0.42),
            call("exp-1", "arm-a", 0.42),
        ]


class TestPersistMetricsRewardInvariance:
    """§7B — Module E reward path ignores observational side channels."""

    def _reward_eligible_metrics(self, **overrides: Any) -> dict[str, Any]:
        """Build a reward-eligible Module D → E payload with stable AU12/is_match."""
        metrics: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "d" * 64,
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "semantic": {
                "reasoning": "cross_encoder_high_match",
                "is_match": True,
                "confidence_score": 0.25,
                "semantic_method": "cross_encoder",
                "semantic_method_version": "ce-v1",
            },
            "_active_arm": "arm-a",
            "_experiment_id": "exp-1",
            "_stimulus_modality": "spoken_greeting",
            "_stimulus_payload": {"content_type": "text", "text": "hello welcome"},
            "_expected_stimulus_rule": "Deliver the spoken greeting to the creator",
            "_expected_response_rule": "The streamer acknowledges the stimulus.",
            "_stimulus_time": 100.0,
            "stimulus_time_utc": "2026-03-13T12:00:00+00:00",
            "_bandit_decision_snapshot": {
                "selection_method": "thompson_sampling",
                "selection_time_utc": "2026-03-13T12:00:00+00:00",
                "experiment_id": 1,
                "policy_version": "policy-v1",
                "selected_arm_id": "arm-a",
                "candidate_arm_ids": ["arm-a", "arm-b"],
                "posterior_by_arm": {
                    "arm-a": {"alpha": 1.0, "beta": 1.0},
                    "arm-b": {"alpha": 2.0, "beta": 3.0},
                },
                "sampled_theta_by_arm": {"arm-a": 0.6, "arm-b": 0.4},
                "stimulus_modality": "spoken_greeting",
                "stimulus_payload": {"content_type": "text", "text": "hello welcome"},
                "expected_stimulus_rule": "Deliver the spoken greeting to the creator",
                "expected_response_rule": "The streamer acknowledges the stimulus.",
                "decision_context_hash": "e" * 64,
                "random_seed": 42,
            },
            "_au12_series": [
                {"timestamp_s": 95.1, "intensity": 0.10},
                {"timestamp_s": 96.0, "intensity": 0.12},
                {"timestamp_s": 97.4, "intensity": 0.11},
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
        }
        metrics.update(overrides)
        return metrics

    def _side_channel_overrides(self) -> dict[str, Any]:
        """Payload differences that must remain observational-only for reward."""
        return {
            "semantic": {
                "reasoning": "gray_band_llm_match",
                "is_match": True,
                "confidence_score": 0.99,
                "semantic_method": "llm_gray_band",
                "semantic_method_version": "fallback-v9",
            },
            "_physiological_context": {
                "streamer": {
                    "rmssd_ms": 88.0,
                    "heart_rate_bpm": 60,
                    "freshness_s": 5.0,
                    "is_stale": False,
                    "provider": "oura",
                    "source_kind": "physiology_chunk",
                    "derivation_method": "rolling_rmssd",
                    "window_s": 300.0,
                    "validity_ratio": 0.96,
                    "is_valid": True,
                    "source_timestamp_utc": "2026-03-13T11:59:55+00:00",
                },
                "operator": {
                    "rmssd_ms": 22.0,
                    "heart_rate_bpm": 82,
                    "freshness_s": 4.0,
                    "is_stale": False,
                    "provider": "oura",
                    "source_kind": "physiology_chunk",
                    "derivation_method": "rolling_rmssd",
                    "window_s": 300.0,
                    "validity_ratio": 0.91,
                    "is_valid": True,
                    "source_timestamp_utc": "2026-03-13T11:59:56+00:00",
                },
            },
            "co_modulation_index": -0.8,
            "n_paired_observations": 9,
            "coverage_ratio": 0.9,
            "f0_valid_measure": True,
            "f0_valid_baseline": True,
            "perturbation_valid_measure": True,
            "perturbation_valid_baseline": True,
            "voiced_coverage_measure_s": 2.4,
            "voiced_coverage_baseline_s": 1.8,
            "f0_mean_measure_hz": 260.0,
            "f0_mean_baseline_hz": 170.0,
            "f0_delta_semitones": 7.35,
            "jitter_mean_measure": 0.04,
            "jitter_mean_baseline": 0.01,
            "jitter_delta": 0.03,
            "shimmer_mean_measure": 0.08,
            "shimmer_mean_baseline": 0.02,
            "shimmer_delta": 0.06,
            "_bandit_decision_snapshot": {
                "selection_method": "thompson_sampling",
                "selection_time_utc": "2026-03-13T12:00:00+00:00",
                "experiment_id": 1,
                "policy_version": "policy-v9",
                "selected_arm_id": "arm-a",
                "candidate_arm_ids": ["arm-a", "arm-b"],
                "posterior_by_arm": {
                    "arm-a": {"alpha": 1.0, "beta": 1.0},
                    "arm-b": {"alpha": 9.0, "beta": 1.0},
                },
                "sampled_theta_by_arm": {"arm-a": 0.2, "arm-b": 0.9},
                "stimulus_modality": "spoken_greeting",
                "stimulus_payload": {"content_type": "text", "text": "hello welcome"},
                "expected_stimulus_rule": "Deliver the spoken greeting to the creator",
                "expected_response_rule": "The streamer acknowledges the stimulus.",
                "decision_context_hash": "f" * 64,
                "random_seed": 99,
            },
            "outcome_event": {
                "outcome_type": "creator_follow",
                "outcome_value": 1.0,
                "outcome_time_utc": "2026-03-13T12:02:00+00:00",
                "source_system": "tiktok_webcast",
                "source_event_ref": "follow-differential",
                "confidence": 1.0,
            },
            "attribution_event": {"event_id": "evt-1", "evidence_flags": ["semantic"]},
            "attribution_score": {
                "attribution_method": "lagged_correlation",
                "soft_reward_candidate": 0.99,
                "au12_baseline_pre": 0.0,
                "sync_peak_corr": 0.95,
            },
        }

    def test_empty_au12_list_flows_to_reward_and_updates_from_gated_reward(self) -> None:
        """§7B/§7E — Empty AU12 updates reward while missing outcomes stay non-fatal."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_engine = MagicMock()
        metrics = self._reward_eligible_metrics(_au12_series=[])

        from services.worker.pipeline.reward import RewardResult

        reward_result = RewardResult(
            gated_reward=0.37,
            p90_intensity=0.91,
            semantic_gate=1,
            n_frames_in_window=0,
            au12_baseline_pre=None,
        )
        mock_log_encounter = MagicMock()

        with (
            patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
            patch(
                "services.worker.pipeline.analytics.ThompsonSamplingEngine",
                return_value=mock_engine,
            ),
            patch(
                "services.worker.pipeline.reward.compute_reward",
                return_value=reward_result,
            ) as mock_compute_reward,
            patch.object(mod, "_log_encounter", mock_log_encounter),
        ):
            mod.persist_metrics(MagicMock(), metrics)

        mock_compute_reward.assert_called_once()
        reward_call = mock_compute_reward.call_args
        assert reward_call.kwargs["au12_series"] == []
        assert reward_call.kwargs["stimulus_time_s"] == 100.0
        assert reward_call.kwargs["is_match"] is True
        mock_engine.update.assert_called_once_with("exp-1", "arm-a", 0.37)
        mock_log_encounter.assert_called_once()
        assert mock_log_encounter.call_args.args[4] is reward_result

        mock_store.persist_attribution_ledger.assert_called_once()
        ledger = mock_store.persist_attribution_ledger.call_args.args[0]
        assert ledger.event.semantic_method == "cross_encoder"
        assert ledger.event.semantic_method_version == "ce-v1"
        assert ledger.outcomes == ()
        assert ledger.links == ()
        assert len(ledger.scores) == 6
        assert {score.outcome_id for score in ledger.scores} == {None}

    @pytest.mark.audit_item("13.31")
    def test_differential_payloads_produce_identical_reward_result(self) -> None:
        """§7E writes preserve §7B reward result and posterior invariance."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_store.compute_comodulation.return_value = None
        mock_engine = MagicMock()
        base_metrics = self._reward_eligible_metrics()
        differential_metrics = self._reward_eligible_metrics(**self._side_channel_overrides())

        from services.worker.pipeline.analytics import ThompsonSamplingEngine
        from services.worker.pipeline.reward import compute_reward as real_compute_reward

        arm_priors = [
            {"arm": "arm-a", "alpha_param": 3.0, "beta_param": 2.0},
            {"arm": "arm-b", "alpha_param": 2.0, "beta_param": 3.0},
        ]
        base_selection_store = MagicMock()
        differential_selection_store = MagicMock()
        base_selection_store.get_experiment_arms.return_value = arm_priors
        differential_selection_store.get_experiment_arms.return_value = arm_priors
        with patch("scipy.stats.beta.rvs", side_effect=[0.73, 0.44, 0.73, 0.44]) as mock_beta_rvs:
            base_selected_arm = ThompsonSamplingEngine(base_selection_store).select_arm(
                base_metrics["_experiment_id"]
            )
            differential_selected_arm = ThompsonSamplingEngine(
                differential_selection_store
            ).select_arm(differential_metrics["_experiment_id"])

        assert "attribution_score" not in base_metrics
        assert differential_metrics["attribution_score"]["soft_reward_candidate"] == 0.99
        assert base_selected_arm == differential_selected_arm == "arm-a"
        assert base_selection_store.get_experiment_arms.call_args_list == [call("exp-1")]
        assert differential_selection_store.get_experiment_arms.call_args_list == [call("exp-1")]
        assert mock_beta_rvs.call_args_list == [
            call(3.0, 2.0),
            call(2.0, 3.0),
            call(3.0, 2.0),
            call(2.0, 3.0),
        ]

        reward_results: list[Any] = []

        def _record_reward(**kwargs: Any) -> Any:
            reward_result = real_compute_reward(**kwargs)
            reward_results.append(reward_result)
            return reward_result

        with (
            patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
            patch(
                "services.worker.pipeline.analytics.ThompsonSamplingEngine",
                return_value=mock_engine,
            ),
            patch(
                "services.worker.pipeline.reward.compute_reward",
                side_effect=_record_reward,
            ) as mock_compute_reward,
            patch.object(mod, "_log_encounter", MagicMock()),
        ):
            mod.persist_metrics(MagicMock(), base_metrics)
            mod.persist_metrics(MagicMock(), differential_metrics)

        assert len(reward_results) == 2
        assert reward_results[0] == reward_results[1]
        # Differential payload equality check: the serialized RewardResult is
        # identical even though semantic confidence/method, physiology,
        # co-modulation, acoustics, and attribution differ.
        assert asdict(reward_results[0]) == asdict(reward_results[1])
        assert "au12_baseline_pre" in asdict(reward_results[0])

        for reward_call in mock_compute_reward.call_args_list:
            assert set(reward_call.kwargs) == {"au12_series", "stimulus_time_s", "is_match"}
            assert reward_call.kwargs["is_match"] is True
            assert "confidence_score" not in reward_call.kwargs
            assert "semantic_method" not in reward_call.kwargs
            assert "_physiological_context" not in reward_call.kwargs
            assert "co_modulation_index" not in reward_call.kwargs
            assert "attribution_score" not in reward_call.kwargs
            assert "x_max" not in reward_call.kwargs

        expected_reward = reward_results[0].gated_reward
        assert mock_engine.update.call_args_list == [
            call("exp-1", "arm-a", expected_reward),
            call("exp-1", "arm-a", expected_reward),
        ]

        persisted_differential = mock_store.insert_metrics.call_args_list[1].args[0]
        assert persisted_differential["semantic"]["confidence_score"] == 0.99
        assert persisted_differential["semantic"]["semantic_method"] == "llm_gray_band"
        assert set(persisted_differential["semantic"]) == {
            "reasoning",
            "is_match",
            "confidence_score",
            "semantic_method",
            "semantic_method_version",
        }
        assert persisted_differential["_physiological_context"]["streamer"]["rmssd_ms"] == 88.0
        assert persisted_differential["attribution_score"]["soft_reward_candidate"] == 0.99

        assert mock_store.persist_attribution_ledger.call_count == 2
        base_ledger = mock_store.persist_attribution_ledger.call_args_list[0].args[0]
        differential_ledger = mock_store.persist_attribution_ledger.call_args_list[1].args[0]
        assert base_ledger.event.event_id == differential_ledger.event.event_id
        assert base_ledger.event.semantic_method == "cross_encoder"
        assert base_ledger.event.semantic_method_version == "ce-v1"
        assert base_ledger.outcomes == ()
        assert base_ledger.links == ()
        assert {score.outcome_id for score in base_ledger.scores} == {None}
        assert differential_ledger.event.semantic_method == "llm_gray_band"
        assert differential_ledger.event.semantic_method_version == "fallback-v9"
        assert len(differential_ledger.outcomes) == 1
        assert len(differential_ledger.links) == 1
        assert len(differential_ledger.scores) == 6
        assert differential_ledger.links[0].event_id == differential_ledger.event.event_id
        assert differential_ledger.links[0].outcome_id == differential_ledger.outcomes[0].outcome_id
        assert {score.event_id for score in differential_ledger.scores} == {
            differential_ledger.event.event_id
        }
        assert {score.outcome_id for score in differential_ledger.scores} == {
            differential_ledger.outcomes[0].outcome_id
        }
        differential_scores = {
            score.attribution_method: score for score in differential_ledger.scores
        }
        assert differential_scores["sync_peak_corr"].score_raw is None
        assert differential_scores["sync_peak_lag"].score_raw is None
        assert differential_scores["sync_peak_corr"].evidence_flags == []
        assert differential_scores["sync_peak_lag"].evidence_flags == []

    def test_connect_failure_routes_attribution_payload_to_store_contract(self) -> None:
        """§2.7/§7E — Connect outage still hands attribution ledger to store buffering."""
        mod = _get_inference_module()
        mock_store = MagicMock()
        mock_store.connect.side_effect = RuntimeError("persistent store unavailable")
        metrics = self._reward_eligible_metrics()

        with patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store):
            mod.persist_metrics(MagicMock(), metrics)

        mock_store.connect.assert_called_once()
        mock_store.insert_metrics.assert_not_called()
        mock_store.persist_attribution_ledger.assert_called_once()
        ledger = mock_store.persist_attribution_ledger.call_args.args[0]
        assert str(ledger.event.event_id)
        assert str(ledger.event.session_id) == metrics["session_id"]
        assert ledger.event.segment_id == metrics["segment_id"]
        assert ledger.event.selected_arm_id == "arm-a"
        assert len(ledger.scores) == 6
        mock_store.close.assert_called_once()
