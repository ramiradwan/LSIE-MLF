"""Targeted tests for physiological context handling in persist_metrics (§4.D, §7B, §12)."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any
from unittest.mock import MagicMock, call, patch


def _get_inference_module() -> Any:
    """Import inference module with mocked celery decorator."""
    mock_app = MagicMock()
    mock_app.task.return_value = lambda f: f

    with patch("services.worker.celery_app.celery_app", mock_app):
        mod_name = "services.worker.tasks.inference"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return importlib.import_module(mod_name)


def test_persist_metrics_logs_physio_availability(caplog: Any) -> None:
    """§4.D / §12 — persist_metrics logs physio availability for enriched snapshots."""
    mod = _get_inference_module()
    mock_store = MagicMock()
    streamer_snapshot = {
        "rmssd_ms": None,
        "heart_rate_bpm": 72,
        "source_timestamp_utc": "2026-03-13T11:59:30+00:00",
        "freshness_s": 30.0,
        "is_stale": False,
        "provider": "oura",
        "validity_ratio": 0.92,
        "source_kind": "ibi",
        "derivation_method": "server",
        "window_length_s": 300,
    }
    metrics = {
        "session_id": "test-session",
        "segment_id": "seg-001",
        "_physiological_context": {
            "streamer": streamer_snapshot,
            "operator": None,
        },
    }

    with (
        patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
        caplog.at_level(logging.INFO),
    ):
        mod.persist_metrics(MagicMock(), metrics)

    assert any(
        record.message
        == "Physiological context available: physio_available=True streamer=True operator=False"
        for record in caplog.records
    )
    mock_store.insert_metrics.assert_called_once_with(metrics)
    persisted_metrics = mock_store.insert_metrics.call_args.args[0]
    assert persisted_metrics["_physiological_context"] is metrics["_physiological_context"]
    mock_store.persist_physiology_snapshot.assert_called_once_with(
        session_id="test-session",
        segment_id="seg-001",
        subject_role="streamer",
        snapshot=streamer_snapshot,
    )
    mock_store.close.assert_called_once()


def test_persist_metrics_without_physio_context_does_not_error(caplog: Any) -> None:
    """persist_metrics omits explicit null physiology and completes normally."""
    mod = _get_inference_module()
    mock_store = MagicMock()
    metrics = {
        "session_id": "test-session",
        "segment_id": "seg-002",
        "_physiological_context": None,
    }

    with (
        patch("services.worker.pipeline.analytics.MetricsStore", return_value=mock_store),
        caplog.at_level(logging.INFO),
    ):
        mod.persist_metrics(MagicMock(), metrics)

    assert not any(
        "Physiological context available:" in record.message for record in caplog.records
    )
    persisted_metrics = mock_store.insert_metrics.call_args.args[0]
    assert "_physiological_context" not in persisted_metrics
    mock_store.persist_physiology_snapshot.assert_not_called()
    mock_store.close.assert_called_once()


def test_persist_metrics_keeps_reward_pipeline_unchanged() -> None:
    """§4.D / §7B — Physiology changes do not alter reward or TS update inputs."""
    mod = _get_inference_module()
    mock_store = MagicMock()
    mock_store.compute_comodulation.return_value = None
    mock_engine = MagicMock()

    from services.worker.pipeline.reward import RewardResult, TimestampedAU12

    reward_result = RewardResult(
        gated_reward=0.75,
        p90_intensity=0.75,
        semantic_gate=1,
        is_valid=True,
        n_frames_in_window=12,
        baseline_b_neutral=0.1,
        au12_window_series=[0.2, 0.5],
    )
    expected_series = [
        TimestampedAU12(timestamp_s=100.5, intensity=0.2),
        TimestampedAU12(timestamp_s=101.0, intensity=0.5),
    ]
    base_metrics = {
        "session_id": "test-session",
        "segment_id": "seg-003",
        "timestamp_utc": "2026-03-13T12:00:00+00:00",
        "semantic": {
            "is_match": True,
            "confidence_score": 0.8,
        },
        "_active_arm": "arm-a",
        "_experiment_id": "exp-1",
        "_au12_series": [
            {"timestamp_s": 100.5, "intensity": 0.2},
            {"timestamp_s": 101.0, "intensity": 0.5},
        ],
        "_stimulus_time": 100.0,
        "_x_max": 0.9,
    }
    streamer_snapshot = {
        "rmssd_ms": None,
        "heart_rate_bpm": 71,
        "source_timestamp_utc": "2026-03-13T11:59:30+00:00",
        "freshness_s": 30.0,
        "is_stale": False,
        "provider": "oura",
        "validity_ratio": 0.92,
    }
    operator_snapshot = {
        "rmssd_ms": 35.7,
        "heart_rate_bpm": 69,
        "source_timestamp_utc": "2026-03-13T11:59:32+00:00",
        "freshness_s": 28.0,
        "is_stale": False,
        "provider": "oura",
        "validity_ratio": 0.88,
    }
    metrics_without_physio = dict(base_metrics)
    metrics_with_physio = {
        **base_metrics,
        "_physiological_context": {
            "streamer": streamer_snapshot,
            "operator": operator_snapshot,
        },
    }
    expected_reward_call = call(
        au12_series=expected_series,
        stimulus_time_s=100.0,
        is_match=True,
        confidence_score=0.8,
        x_max=0.9,
    )

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
        mod.persist_metrics(MagicMock(), metrics_without_physio)
        mod.persist_metrics(MagicMock(), metrics_with_physio)

    assert mock_compute_reward.call_args_list == [expected_reward_call, expected_reward_call]
    assert mock_engine.update.call_args_list == [
        call("exp-1", "arm-a", 0.75),
        call("exp-1", "arm-a", 0.75),
    ]
    for reward_call in mock_compute_reward.call_args_list:
        assert "_physiological_context" not in reward_call.kwargs
        assert "rmssd_ms" not in reward_call.kwargs
    persisted_without_physio = mock_store.insert_metrics.call_args_list[0].args[0]
    persisted_with_physio = mock_store.insert_metrics.call_args_list[1].args[0]
    assert "_physiological_context" not in persisted_without_physio
    assert (
        persisted_with_physio["_physiological_context"]
        is metrics_with_physio["_physiological_context"]
    )
    mock_store.persist_physiology_snapshot.assert_has_calls(
        [
            call(
                session_id="test-session",
                segment_id="seg-003",
                subject_role="streamer",
                snapshot=streamer_snapshot,
            ),
            call(
                session_id="test-session",
                segment_id="seg-003",
                subject_role="operator",
                snapshot=operator_snapshot,
            ),
        ],
        any_order=False,
    )
    assert mock_store.persist_physiology_snapshot.call_count == 2
    assert mock_store.close.call_count == 2
