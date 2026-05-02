from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.ipc.control_messages import AnalyticsResultMessage
from services.desktop_app.processes.analytics_state_worker import LocalAnalyticsProcessor
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _handoff_payload(*, stimulus_time: float | None = 100.0) -> dict[str, Any]:
    sample_timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    return {
        "session_id": SESSION_ID,
        "segment_id": SEGMENT_ID,
        "segment_window_start_utc": sample_timestamp.isoformat(),
        "segment_window_end_utc": sample_timestamp.isoformat(),
        "timestamp_utc": sample_timestamp.isoformat(),
        "media_source": {
            "stream_url": "https://example.com/stream",
            "codec": "h264",
            "resolution": [1920, 1080],
        },
        "segments": [],
        "_active_arm": "warm_welcome",
        "_experiment_id": 1,
        "_expected_greeting": "Say hello to the creator",
        "_stimulus_time": stimulus_time,
        "_au12_series": [
            {"timestamp_s": 96.0, "intensity": 0.2},
            {"timestamp_s": 97.5, "intensity": 0.4},
            {"timestamp_s": 100.4, "intensity": 1.0},
            {"timestamp_s": 100.5, "intensity": 0.1},
            {"timestamp_s": 101.0, "intensity": 0.5},
            {"timestamp_s": 105.0, "intensity": 0.9},
            {"timestamp_s": 105.1, "intensity": 1.0},
        ],
        "_bandit_decision_snapshot": {
            "selection_method": "thompson_sampling",
            "selection_time_utc": sample_timestamp.isoformat(),
            "experiment_id": 1,
            "policy_version": "ts-v1",
            "selected_arm_id": "warm_welcome",
            "candidate_arm_ids": ["warm_welcome", "direct_question"],
            "posterior_by_arm": {
                "warm_welcome": {"alpha": 1.0, "beta": 1.0},
                "direct_question": {"alpha": 1.0, "beta": 1.0},
            },
            "sampled_theta_by_arm": {"warm_welcome": 0.72, "direct_question": 0.44},
            "expected_greeting": "Say hello to the creator",
            "decision_context_hash": DECISION_CONTEXT_HASH,
            "random_seed": 42,
        },
    }


def _analytics_message(
    *,
    handoff: dict[str, Any] | None = None,
    is_match: bool = True,
    confidence_score: float = 0.5,
) -> AnalyticsResultMessage:
    return AnalyticsResultMessage.model_validate(
        {
            "message_id": "00000000-0000-4000-8000-000000000010",
            "handoff": handoff or _handoff_payload(),
            "semantic": {
                "reasoning": "cross_encoder_high_match",
                "is_match": is_match,
                "confidence_score": confidence_score,
                "semantic_method": "cross_encoder",
                "semantic_method_version": "test-v1",
            },
        }
    )


def _prepare_db(db: Path) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, stream_url) VALUES (?, ?)",
        (SESSION_ID, "https://example.com/stream"),
    )
    conn.close()


def test_processor_updates_local_posterior_and_returns_delta(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    delta = processor.process(_analytics_message())

    assert delta is not None
    assert delta.experiment_id == 1
    assert delta.arm_id == "warm_welcome"
    assert delta.delta_alpha == pytest.approx(0.82, abs=1e-12)
    assert delta.delta_beta == pytest.approx(0.18, abs=1e-12)
    assert delta.decision_context_hash == DECISION_CONTEXT_HASH

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    experiment = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    encounter = conn.execute(
        "SELECT gated_reward, p90_intensity, semantic_gate, n_frames_in_window, au12_baseline_pre "
        "FROM encounter_log WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    conn.close()

    assert experiment is not None
    assert experiment["alpha_param"] == pytest.approx(1.82, abs=1e-12)
    assert experiment["beta_param"] == pytest.approx(1.18, abs=1e-12)
    assert encounter is not None
    assert encounter["gated_reward"] == pytest.approx(0.82, abs=1e-12)
    assert encounter["p90_intensity"] == pytest.approx(0.82, abs=1e-12)
    assert encounter["semantic_gate"] == 1
    assert encounter["n_frames_in_window"] == 3
    assert encounter["au12_baseline_pre"] == pytest.approx(0.3, abs=1e-12)


def test_processor_duplicate_message_does_not_reapply_posterior(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    message = _analytics_message()

    first = processor.process(message)
    second = processor.process(message)

    assert first is not None
    assert second is None
    conn = sqlite3.connect(str(db), isolation_level=None)
    experiment = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    encounter_count = conn.execute(
        "SELECT COUNT(*) FROM encounter_log WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    ledger_count = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()[0]
    conn.close()

    assert experiment == pytest.approx((1.82, 1.18), abs=1e-12)
    assert encounter_count == 1
    assert ledger_count == 1


def test_processor_rolls_back_ledger_when_local_update_fails(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    payload = _handoff_payload()
    payload["_active_arm"] = "missing_arm"

    with pytest.raises(ValueError, match="missing_arm"):
        processor.process(_analytics_message(handoff=payload))

    conn = sqlite3.connect(str(db), isolation_level=None)
    ledger_count = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()[0]
    encounter_count = conn.execute("SELECT COUNT(*) FROM encounter_log").fetchone()[0]
    conn.close()
    assert ledger_count == 0
    assert encounter_count == 0


def test_processor_uses_semantic_gate_without_confidence_or_physiology(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    payload = _handoff_payload()
    payload["_physiological_context"] = {
        "streamer": {
            "rmssd_ms": 99.0,
            "heart_rate_bpm": 70,
            "source_timestamp_utc": "2026-05-02T12:00:00Z",
            "freshness_s": 1.0,
            "is_stale": False,
            "provider": "oura",
            "source_kind": "ibi",
            "derivation_method": "provider",
            "window_s": 300,
            "validity_ratio": 1.0,
            "is_valid": True,
        }
    }

    delta = processor.process(
        _analytics_message(handoff=payload, is_match=False, confidence_score=0.99)
    )

    assert delta is not None
    assert delta.delta_alpha == 0.0
    assert delta.delta_beta == 1.0

    conn = sqlite3.connect(str(db), isolation_level=None)
    row = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    conn.close()
    assert row == pytest.approx((1.0, 2.0), abs=1e-12)


def test_processor_skips_segments_without_stimulus(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    delta = processor.process(_analytics_message(handoff=_handoff_payload(stimulus_time=None)))

    assert delta is None
    conn = sqlite3.connect(str(db), isolation_level=None)
    count = conn.execute("SELECT COUNT(*) FROM encounter_log").fetchone()[0]
    conn.close()
    assert count == 0


def test_delta_enqueues_after_local_update(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    outbox = CloudOutbox(db)
    try:
        delta = processor.process(_analytics_message())
        assert delta is not None
        outbox.enqueue_posterior_delta(delta)
        upload = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=1)[0]
    finally:
        outbox.close()

    stored = json.loads(upload.payload_json)
    assert upload.payload_type == "posterior_delta"
    assert stored["segment_id"] == SEGMENT_ID
    assert stored["delta_alpha"] == pytest.approx(0.82, abs=1e-12)
    assert stored["delta_beta"] == pytest.approx(0.18, abs=1e-12)
