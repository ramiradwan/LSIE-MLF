from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from packages.schemas.attribution import AttributionEvent
from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.processes.analytics_state_worker import (
    POSTERIOR_EVENT_NAMESPACE,
    LocalAnalyticsProcessor,
    QueueLike,
    _run_loop,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"
TRANSCRIPTION = "hello creator"


def _stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text="Say hello to the creator",
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


def _handoff_payload(
    *,
    stimulus_time: float | None = 100.0,
    response_inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    payload = {
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
        "_stimulus_modality": _stimulus_definition().stimulus_modality,
        "_stimulus_payload": _stimulus_definition().stimulus_payload.model_dump(mode="json"),
        "_expected_stimulus_rule": _stimulus_definition().expected_stimulus_rule,
        "_expected_response_rule": _stimulus_definition().expected_response_rule,
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
            "stimulus_modality": _stimulus_definition().stimulus_modality,
            "stimulus_payload": _stimulus_definition().stimulus_payload.model_dump(mode="json"),
            "expected_stimulus_rule": _stimulus_definition().expected_stimulus_rule,
            "expected_response_rule": _stimulus_definition().expected_response_rule,
            "decision_context_hash": DECISION_CONTEXT_HASH,
            "random_seed": 42,
        },
    }
    if response_inference is not None:
        payload["response_inference"] = response_inference
    return payload


def _acoustic_payload() -> dict[str, Any]:
    return {
        "f0_valid_measure": True,
        "f0_valid_baseline": False,
        "perturbation_valid_measure": True,
        "perturbation_valid_baseline": False,
        "voiced_coverage_measure_s": 1.25,
        "voiced_coverage_baseline_s": 0.0,
        "f0_mean_measure_hz": 215.5,
        "f0_mean_baseline_hz": None,
        "f0_delta_semitones": None,
        "jitter_mean_measure": 0.031,
        "jitter_mean_baseline": None,
        "jitter_delta": None,
        "shimmer_mean_measure": 0.044,
        "shimmer_mean_baseline": None,
        "shimmer_delta": None,
    }


def _analytics_message(
    *,
    handoff: dict[str, Any] | None = None,
    message_id: str = "00000000-0000-4000-8000-000000000010",
    is_match: bool = True,
    confidence_score: float = 0.5,
    transcription: str = TRANSCRIPTION,
    reward: dict[str, Any] | None = None,
    attribution: dict[str, Any] | None = None,
) -> AnalyticsResultMessage:
    payload: dict[str, Any] = {
        "message_id": message_id,
        "handoff": handoff or _handoff_payload(),
        "semantic": {
            "reasoning": "cross_encoder_high_match",
            "is_match": is_match,
            "confidence_score": confidence_score,
            "semantic_method": "cross_encoder",
            "semantic_method_version": "test-v1",
        },
        "transcription": transcription,
        "acoustic": _acoustic_payload(),
    }
    if reward is not None:
        payload["reward"] = reward
    if attribution is not None:
        payload["attribution"] = attribution
    return AnalyticsResultMessage.model_validate(payload)


def _prepare_db(db: Path) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    bootstrap_schema(conn)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, stream_url) VALUES (?, ?)",
        (SESSION_ID, "https://example.com/stream"),
    )
    conn.close()


def _prepare_legacy_replay_db(db: Path) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            stream_url TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attribution_event (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            segment_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_time_utc TEXT NOT NULL,
            stimulus_time_utc TEXT,
            finality TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE outcome_event (
            outcome_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            outcome_type TEXT NOT NULL,
            outcome_value REAL NOT NULL,
            outcome_time_utc TEXT NOT NULL,
            source_system TEXT NOT NULL,
            source_event_ref TEXT,
            confidence REAL NOT NULL,
            finality TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE event_outcome_link (
            link_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES attribution_event(event_id),
            outcome_id TEXT NOT NULL REFERENCES outcome_event(outcome_id),
            lag_s REAL NOT NULL,
            horizon_s REAL NOT NULL,
            link_rule_version TEXT NOT NULL,
            eligibility_flags TEXT NOT NULL DEFAULT '[]',
            finality TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attribution_score (
            score_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES attribution_event(event_id),
            outcome_id TEXT REFERENCES outcome_event(outcome_id),
            attribution_method TEXT NOT NULL,
            method_version TEXT NOT NULL,
            score_raw REAL,
            score_normalized REAL,
            confidence REAL,
            evidence_flags TEXT NOT NULL DEFAULT '[]',
            finality TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    bootstrap_schema(conn)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, stream_url) VALUES (?, ?)",
        (SESSION_ID, "https://example.com/stream"),
    )
    conn.close()


def _visual_state(
    *,
    message_id: str = "00000000-0000-4000-8000-000000000020",
    timestamp_utc: datetime = datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
    status: str = "calibrating",
    face_present: bool = True,
    is_calibrating: bool = True,
    calibration_frames_accumulated: int = 5,
    calibration_frames_required: int = 10,
    latest_au12_intensity: float | None = 0.42,
    latest_au12_timestamp_s: float | None = 12.25,
) -> VisualAnalyticsStateMessage:
    return VisualAnalyticsStateMessage.model_validate(
        {
            "message_id": message_id,
            "session_id": SESSION_ID,
            "timestamp_utc": timestamp_utc,
            "face_present": face_present,
            "is_calibrating": is_calibrating,
            "calibration_frames_accumulated": calibration_frames_accumulated,
            "calibration_frames_required": calibration_frames_required,
            "active_arm": "warm_welcome",
            "stimulus_definition": _stimulus_definition(),
            "latest_au12_intensity": latest_au12_intensity,
            "latest_au12_timestamp_s": latest_au12_timestamp_s,
            "status": status,
        }
    )


class _OneMessageInbox:
    def __init__(self, raw: object, shutdown_event: _ShutdownEvent) -> None:
        self._raw = raw
        self._shutdown_event = shutdown_event

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        self._shutdown_event.set()
        return self._raw


class _ShutdownEvent:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


class _Outbox:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, object]] = []

    def enqueue_inference_handoff(self, handoff: object) -> None:
        self.enqueued.append(("handoff", handoff))

    def enqueue_attribution_event(self, event: object) -> None:
        self.enqueued.append(("attribution_event", event))

    def enqueue_posterior_delta(self, delta: object) -> None:
        self.enqueued.append(("posterior_delta", delta))


class _FailingAttributionOutbox(_Outbox):
    def enqueue_attribution_event(self, event: object) -> None:
        raise RuntimeError("outbox unavailable")


def test_processor_upserts_visual_state(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    processor.process_visual_state(_visual_state())

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT session_id, active_arm, stimulus_definition, is_calibrating,
               calibration_frames_accumulated, calibration_frames_required,
               face_present, latest_au12_intensity, latest_au12_timestamp_s,
               status, updated_at_utc
        FROM live_session_state
        WHERE session_id = ?
        """,
        (SESSION_ID,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["session_id"] == SESSION_ID
    assert row["active_arm"] == "warm_welcome"
    assert json.loads(str(row["stimulus_definition"])) == _stimulus_definition().model_dump(
        mode="json"
    )
    assert row["is_calibrating"] == 1
    assert row["calibration_frames_accumulated"] == 5
    assert row["calibration_frames_required"] == 10
    assert row["face_present"] == 1
    assert row["latest_au12_intensity"] == pytest.approx(0.42, abs=1e-12)
    assert row["latest_au12_timestamp_s"] == pytest.approx(12.25, abs=1e-12)
    assert row["status"] == "calibrating"
    assert row["updated_at_utc"] == "2026-05-02T12:00:00Z"


def test_processor_visual_state_replaces_previous_state(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    processor.process_visual_state(_visual_state())
    processor.process_visual_state(
        _visual_state(
            message_id="00000000-0000-4000-8000-000000000021",
            timestamp_utc=datetime(2026, 5, 2, 12, 1, tzinfo=UTC),
            status="ready",
            face_present=True,
            is_calibrating=False,
            calibration_frames_accumulated=10,
            calibration_frames_required=10,
            latest_au12_intensity=0.88,
            latest_au12_timestamp_s=60.5,
        )
    )

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT is_calibrating, calibration_frames_accumulated, face_present,
               latest_au12_intensity, latest_au12_timestamp_s, status, updated_at_utc
        FROM live_session_state
        WHERE session_id = ?
        """,
        (SESSION_ID,),
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["is_calibrating"] == 0
    assert row["calibration_frames_accumulated"] == 10
    assert row["face_present"] == 1
    assert row["latest_au12_intensity"] == pytest.approx(0.88, abs=1e-12)
    assert row["latest_au12_timestamp_s"] == pytest.approx(60.5, abs=1e-12)
    assert row["status"] == "ready"
    assert row["updated_at_utc"] == "2026-05-02T12:01:00Z"


def test_processor_visual_state_does_not_write_analytics_or_reward_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    processor.process_visual_state(_visual_state())

    conn = sqlite3.connect(str(db), isolation_level=None)
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "analytics_message_ledger",
            "encounter_log",
            "metrics",
            "transcripts",
            "evaluations",
            "pending_uploads",
        )
    }
    experiment = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    conn.close()

    assert counts == {
        "analytics_message_ledger": 0,
        "encounter_log": 0,
        "metrics": 0,
        "transcripts": 0,
        "evaluations": 0,
        "pending_uploads": 0,
    }
    assert experiment == pytest.approx((1.0, 1.0), abs=1e-12)


def test_run_loop_dispatches_visual_state_without_outbox_enqueue(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    shutdown = _ShutdownEvent()
    inbox = _OneMessageInbox(_visual_state().model_dump(mode="json"), shutdown)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    outbox = _Outbox()

    _run_loop(
        cast("Any", shutdown),
        cast("QueueLike", inbox),
        processor,
        cast("CloudOutbox", outbox),
    )

    conn = sqlite3.connect(str(db), isolation_level=None)
    live_count = conn.execute("SELECT COUNT(*) FROM live_session_state").fetchone()[0]
    ledger_count = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()[0]
    conn.close()

    assert live_count == 1
    assert ledger_count == 0
    assert outbox.enqueued == []


def test_processor_updates_local_posterior_and_persists_analytics_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    enqueue_plan = processor.process(_analytics_message())

    assert enqueue_plan is not None
    delta = enqueue_plan.posterior_delta
    assert delta is not None
    assert enqueue_plan.handoff.segment_id == SEGMENT_ID
    assert enqueue_plan.attribution_event is None
    assert delta.experiment_id == 1
    assert delta.arm_id == "warm_welcome"
    assert delta.delta_alpha == pytest.approx(0.82, abs=1e-12)
    assert delta.delta_beta == pytest.approx(0.18, abs=1e-12)
    assert delta.decision_context_hash == DECISION_CONTEXT_HASH
    assert delta.event_id.version == 5
    assert delta.event_id == uuid.uuid5(
        POSTERIOR_EVENT_NAMESPACE,
        f"desktop-a:{SEGMENT_ID}:1:warm_welcome",
    )

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
    transcript = conn.execute(
        "SELECT text FROM transcripts WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    evaluation = conn.execute(
        "SELECT reasoning, is_match, confidence FROM evaluations WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    metrics = conn.execute(
        "SELECT au12_intensity, f0_valid_measure, perturbation_valid_measure, "
        "voiced_coverage_measure_s, f0_mean_measure_hz, jitter_mean_measure, "
        "shimmer_mean_measure FROM metrics WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    ledger = conn.execute(
        "SELECT message_id, segment_id, client_id, arm "
        "FROM analytics_message_ledger WHERE segment_id = ?",
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
    assert transcript is not None
    assert transcript["text"] == TRANSCRIPTION
    assert evaluation is not None
    assert evaluation["reasoning"] == "cross_encoder_high_match"
    assert evaluation["is_match"] == 1
    assert evaluation["confidence"] == pytest.approx(0.5, abs=1e-12)
    assert metrics is not None
    assert metrics["au12_intensity"] is None
    assert metrics["f0_valid_measure"] == 1
    assert metrics["perturbation_valid_measure"] == 1
    assert metrics["voiced_coverage_measure_s"] == pytest.approx(1.25, abs=1e-12)
    assert metrics["f0_mean_measure_hz"] == pytest.approx(215.5, abs=1e-12)
    assert metrics["jitter_mean_measure"] == pytest.approx(0.031, abs=1e-12)
    assert metrics["shimmer_mean_measure"] == pytest.approx(0.044, abs=1e-12)
    assert ledger is not None
    assert ledger["segment_id"] == SEGMENT_ID
    assert ledger["client_id"] == "desktop-a"
    assert ledger["arm"] == "warm_welcome"


def test_processor_ignores_inbound_reward_and_recomputes_from_handoff(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    enqueue_plan = processor.process(
        _analytics_message(
            reward={
                "gated_reward": 0.0,
                "p90_intensity": 0.0,
                "semantic_gate": 0,
                "n_frames_in_window": 0,
                "au12_baseline_pre": None,
                "stimulus_time": 999.0,
            }
        )
    )

    assert enqueue_plan is not None
    delta = enqueue_plan.posterior_delta
    assert delta is not None
    assert delta.delta_alpha == pytest.approx(0.82, abs=1e-12)
    assert delta.delta_beta == pytest.approx(0.18, abs=1e-12)
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    encounter = conn.execute(
        "SELECT gated_reward, p90_intensity, semantic_gate, n_frames_in_window, "
        "au12_baseline_pre, stimulus_time FROM encounter_log WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    experiment = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    conn.close()

    assert encounter is not None
    assert encounter["gated_reward"] == pytest.approx(0.82, abs=1e-12)
    assert encounter["p90_intensity"] == pytest.approx(0.82, abs=1e-12)
    assert encounter["semantic_gate"] == 1
    assert encounter["n_frames_in_window"] == 3
    assert encounter["au12_baseline_pre"] == pytest.approx(0.3, abs=1e-12)
    assert encounter["stimulus_time"] == pytest.approx(100.0, abs=1e-12)
    assert experiment == pytest.approx((1.82, 1.18), abs=1e-12)


def test_processor_duplicate_identity_does_not_reapply_or_duplicate_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    first_message = _analytics_message(message_id="00000000-0000-4000-8000-000000000010")
    second_message = _analytics_message(message_id="00000000-0000-4000-8000-000000000011")

    first = processor.process(first_message)
    second = processor.process(second_message)

    assert first is not None
    assert first.posterior_delta is not None
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
    transcript_count = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    evaluation_count = conn.execute(
        "SELECT COUNT(*) FROM evaluations WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    metrics_count = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    ledger_count = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()[0]
    conn.close()

    assert experiment == pytest.approx((1.82, 1.18), abs=1e-12)
    assert encounter_count == 1
    assert transcript_count == 1
    assert evaluation_count == 1
    assert metrics_count == 1
    assert ledger_count == 1


def test_processor_rolls_back_all_rows_when_local_update_fails(tmp_path: Path) -> None:
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
    transcript_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    evaluation_count = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    metrics_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    conn.close()
    assert ledger_count == 0
    assert encounter_count == 0
    assert transcript_count == 0
    assert evaluation_count == 0
    assert metrics_count == 0


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

    enqueue_plan = processor.process(
        _analytics_message(handoff=payload, is_match=False, confidence_score=0.99)
    )

    assert enqueue_plan is not None
    delta = enqueue_plan.posterior_delta
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


def test_processor_persists_non_reward_rows_without_stimulus(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    enqueue_plan = processor.process(
        _analytics_message(handoff=_handoff_payload(stimulus_time=None))
    )

    assert enqueue_plan is not None
    assert enqueue_plan.posterior_delta is None
    assert enqueue_plan.attribution_event is None
    conn = sqlite3.connect(str(db), isolation_level=None)
    encounter_count = conn.execute("SELECT COUNT(*) FROM encounter_log").fetchone()[0]
    transcript_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    evaluation_count = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    metrics_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    ledger_count = conn.execute("SELECT COUNT(*) FROM analytics_message_ledger").fetchone()[0]
    experiment = conn.execute(
        "SELECT alpha_param, beta_param FROM experiments WHERE id = ? AND arm = ?",
        (1, "warm_welcome"),
    ).fetchone()
    conn.close()
    assert encounter_count == 0
    assert transcript_count == 1
    assert evaluation_count == 1
    assert metrics_count == 1
    assert ledger_count == 1
    assert experiment == pytest.approx((1.0, 1.0), abs=1e-12)


def test_enqueue_plan_persists_handoff_and_delta_after_local_update(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    outbox = CloudOutbox(db)
    try:
        enqueue_plan = processor.process(_analytics_message())
        assert enqueue_plan is not None
        assert enqueue_plan.attribution_event is None
        assert enqueue_plan.posterior_delta is not None
        outbox.enqueue_inference_handoff(enqueue_plan.handoff)
        outbox.enqueue_posterior_delta(enqueue_plan.posterior_delta)
        segment_upload = outbox.fetch_ready_batch("telemetry_segments", limit=1)[0]
        delta_upload = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=1)[0]
    finally:
        outbox.close()

    segment_stored = json.loads(segment_upload.payload_json)
    delta_stored = json.loads(delta_upload.payload_json)
    assert segment_upload.payload_type == "inference_handoff"
    assert segment_stored["segment_id"] == SEGMENT_ID
    assert delta_upload.payload_type == "posterior_delta"
    assert delta_stored["segment_id"] == SEGMENT_ID
    assert delta_stored["delta_alpha"] == pytest.approx(0.82, abs=1e-12)
    assert delta_stored["delta_beta"] == pytest.approx(0.18, abs=1e-12)


def test_processor_preserves_response_inference_in_cloud_and_attribution_paths(
    tmp_path: Path,
) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    response_inference = {
        "is_match": True,
        "confidence_score": 0.91,
        "registration_status": "observable_response",
        "response_reason_code": "response_semantic_ack",
        "matched_response_time": 100.75,
        "evidence_span_ref": "segment:response-window",
    }

    enqueue_plan = processor.process(
        _analytics_message(
            handoff=_handoff_payload(
                stimulus_time=100.0,
                response_inference=response_inference,
            ),
            is_match=True,
            confidence_score=0.8,
            attribution={"_creator_follow": True},
        )
    )

    assert enqueue_plan is not None
    assert enqueue_plan.handoff.response_inference is not None
    assert enqueue_plan.handoff.response_inference.registration_status == "observable_response"
    assert enqueue_plan.handoff.response_inference.response_reason_code == "response_semantic_ack"
    assert enqueue_plan.attribution_event is not None
    assert enqueue_plan.attribution_event.response_registration_status == "observable_response"
    assert enqueue_plan.attribution_event.response_reason_code == "response_semantic_ack"
    assert enqueue_plan.attribution_event.matched_response_time_utc == datetime.fromtimestamp(
        100.75,
        tz=UTC,
    )
    outbox = CloudOutbox(db)
    try:
        outbox.enqueue_inference_handoff(enqueue_plan.handoff)
        upload = outbox.fetch_ready_batch("telemetry_segments", limit=1)[0]
    finally:
        outbox.close()
    segment_stored = json.loads(upload.payload_json)
    assert segment_stored["response_inference"] == response_inference

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    event = conn.execute(
        "SELECT matched_response_time_utc, response_registration_status, response_reason_code "
        "FROM attribution_event WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    conn.close()
    assert event is not None
    assert event["matched_response_time_utc"] == "1970-01-01T00:01:40.750000Z"
    assert event["response_registration_status"] == "observable_response"
    assert event["response_reason_code"] == "response_semantic_ack"


def test_processor_builds_attribution_event_when_inputs_exist(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    enqueue_plan = processor.process(
        _analytics_message(
            handoff=_handoff_payload(stimulus_time=100.0),
            is_match=True,
            confidence_score=0.8,
            attribution={"_creator_follow": True},
        )
    )

    assert enqueue_plan is not None
    event = enqueue_plan.attribution_event
    assert event is not None
    assert event.segment_id == SEGMENT_ID
    assert str(event.session_id) == SESSION_ID
    assert event.selected_arm_id == "warm_welcome"
    assert event.finality == "online_provisional"
    assert event.semantic_p_match == pytest.approx(0.8, abs=1e-12)
    assert event.semantic_reason_code == "cross_encoder_high_match"


def test_processor_persists_online_provisional_attribution_ledger_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")

    enqueue_plan = processor.process(
        _analytics_message(
            handoff=_handoff_payload(stimulus_time=100.0),
            is_match=True,
            confidence_score=0.8,
            attribution={
                "_outcome_event": {
                    "outcome_type": "creator_follow",
                    "outcome_time_utc": "2026-05-02T12:00:03+00:00",
                    "source_system": "desktop_test",
                    "source_event_ref": SEGMENT_ID,
                    "confidence": 1.0,
                },
            },
        )
    )

    assert enqueue_plan is not None
    assert enqueue_plan.attribution_event is not None
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    event = conn.execute(
        "SELECT segment_id, selected_arm_id, finality FROM attribution_event WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    outcome_count = conn.execute("SELECT COUNT(*) FROM outcome_event").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM event_outcome_link").fetchone()[0]
    score_count = conn.execute("SELECT COUNT(*) FROM attribution_score").fetchone()[0]
    conn.close()

    assert event is not None
    assert event["segment_id"] == SEGMENT_ID
    assert event["selected_arm_id"] == "warm_welcome"
    assert event["finality"] == "online_provisional"
    assert outcome_count == 1
    assert link_count == 1
    assert score_count > 0


def test_processor_finalizes_only_closed_horizon_attribution_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    message = _analytics_message(
        handoff=_handoff_payload(stimulus_time=100.0),
        is_match=True,
        confidence_score=0.8,
        attribution={
            "_outcome_events": [
                {
                    "outcome_type": "creator_follow",
                    "outcome_time_utc": "2026-05-02T12:00:03+00:00",
                    "source_system": "desktop_test",
                    "source_event_ref": SEGMENT_ID,
                    "confidence": 1.0,
                },
                {
                    "outcome_type": "creator_follow",
                    "outcome_time_utc": "2026-05-10T12:00:03+00:00",
                    "source_system": "desktop_test",
                    "source_event_ref": SEGMENT_ID,
                    "confidence": 1.0,
                },
            ],
        },
    )
    enqueue_plan = processor.process(message)
    assert enqueue_plan is not None
    assert enqueue_plan.attribution_event is not None
    online_event_id = str(enqueue_plan.attribution_event.event_id)

    younger = processor.finalize_closed_attribution_events(
        now_utc=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )
    finalized = processor.finalize_closed_attribution_events(
        now_utc=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
    )

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    event = conn.execute(
        "SELECT event_id, finality FROM attribution_event WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()
    outcome_finalities = [
        str(row[0]) for row in conn.execute("SELECT finality FROM outcome_event").fetchall()
    ]
    link_finalities = [
        str(row[0]) for row in conn.execute("SELECT finality FROM event_outcome_link").fetchall()
    ]
    score_rows = conn.execute(
        "SELECT score_id, finality FROM attribution_score ORDER BY score_id"
    ).fetchall()
    score_ids = [str(row["score_id"]) for row in score_rows]
    conn.close()

    assert younger == []
    assert len(finalized) == 1
    assert str(finalized[0].event_id) == online_event_id
    assert finalized[0].finality == "offline_final"
    assert event is not None
    assert event["event_id"] == online_event_id
    assert event["finality"] == "offline_final"
    assert outcome_finalities == ["offline_final", "offline_final"]
    assert link_finalities == ["offline_final"]
    assert [str(row["finality"]) for row in score_rows] == ["offline_final"] * len(score_rows)
    assert len(score_ids) == len(set(score_ids))


def test_processor_finalizes_closed_horizon_rows_after_bootstrap_migrates_legacy_attribution_event(
    tmp_path: Path,
) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_legacy_replay_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    message = _analytics_message(
        handoff=_handoff_payload(stimulus_time=100.0),
        is_match=True,
        confidence_score=0.8,
        attribution={"_creator_follow": True},
    )
    enqueue_plan = processor.process(message)
    assert enqueue_plan is not None
    assert enqueue_plan.attribution_event is not None

    finalized = processor.finalize_closed_attribution_events(
        now_utc=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
    )

    assert len(finalized) == 1
    assert finalized[0].stimulus_modality == "spoken_greeting"
    assert finalized[0].expected_response_rule_text_hash is not None
    assert finalized[0].finality == "offline_final"

    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT stimulus_id, stimulus_modality, selected_arm_id,
               expected_rule_text_hash, expected_response_rule_text_hash,
               semantic_method, semantic_method_version, semantic_p_match,
               semantic_reason_code, matched_response_time_utc,
               response_registration_status, response_reason_code,
               reward_path_version, bandit_decision_snapshot,
               evidence_flags, schema_version, created_at
        FROM attribution_event
        WHERE segment_id = ?
        """,
        (SEGMENT_ID,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["stimulus_modality"] == "spoken_greeting"
    assert row["selected_arm_id"] == "warm_welcome"
    assert row["expected_rule_text_hash"] is not None
    assert row["expected_response_rule_text_hash"] is not None
    assert row["semantic_method"] == "cross_encoder"
    assert row["semantic_method_version"] == "test-v1"
    assert row["reward_path_version"] is not None
    assert row["bandit_decision_snapshot"] is not None
    assert row["evidence_flags"] is not None
    assert row["schema_version"] is not None
    assert row["created_at"] is not None


def test_run_loop_enqueues_finalized_event_through_existing_outbox_path(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    old_handoff = _handoff_payload(stimulus_time=100.0)
    old_handoff["timestamp_utc"] = "2026-04-01T12:00:00+00:00"
    processor.process(
        _analytics_message(
            handoff=old_handoff,
            attribution={"_creator_follow": True},
        )
    )
    shutdown = _ShutdownEvent()
    inbox = _OneMessageInbox(_visual_state().model_dump(mode="json", by_alias=True), shutdown)
    outbox = _Outbox()

    _run_loop(
        cast("Any", shutdown),
        cast("QueueLike", inbox),
        processor,
        cast("CloudOutbox", outbox),
    )

    assert [kind for kind, _ in outbox.enqueued] == ["attribution_event"]
    finalized_event = outbox.enqueued[0][1]
    assert isinstance(finalized_event, AttributionEvent)
    assert finalized_event.finality == "offline_final"

    conn = sqlite3.connect(str(db), isolation_level=None)
    finality = conn.execute(
        "SELECT finality FROM attribution_event WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert finality == "offline_final"


def test_run_loop_keeps_replay_rows_provisional_when_outbox_enqueue_fails(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    old_handoff = _handoff_payload(stimulus_time=100.0)
    old_handoff["timestamp_utc"] = "2026-04-01T12:00:00+00:00"
    processor.process(
        _analytics_message(
            handoff=old_handoff,
            attribution={"_creator_follow": True},
        )
    )
    shutdown = _ShutdownEvent()
    inbox = _OneMessageInbox(_visual_state().model_dump(mode="json", by_alias=True), shutdown)

    _run_loop(
        cast("Any", shutdown),
        cast("QueueLike", inbox),
        processor,
        cast("CloudOutbox", _FailingAttributionOutbox()),
    )

    conn = sqlite3.connect(str(db), isolation_level=None)
    finality = conn.execute(
        "SELECT finality FROM attribution_event WHERE segment_id = ?",
        (SEGMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert finality == "online_provisional"


def test_run_loop_enqueues_handoff_before_posterior_delta(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    shutdown = _ShutdownEvent()
    inbox = _OneMessageInbox(_analytics_message().model_dump(mode="json", by_alias=True), shutdown)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    outbox = _Outbox()

    _run_loop(
        cast("Any", shutdown),
        cast("QueueLike", inbox),
        processor,
        cast("CloudOutbox", outbox),
    )

    assert [kind for kind, _ in outbox.enqueued] == ["handoff", "posterior_delta"]


def test_run_loop_enqueues_attribution_event_before_posterior_delta(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _prepare_db(db)
    shutdown = _ShutdownEvent()
    message = _analytics_message(attribution={"_creator_follow": True}).model_dump(
        mode="json",
        by_alias=True,
    )
    inbox = _OneMessageInbox(message, shutdown)
    processor = LocalAnalyticsProcessor(db, client_id="desktop-a")
    outbox = _Outbox()

    _run_loop(
        cast("Any", shutdown),
        cast("QueueLike", inbox),
        processor,
        cast("CloudOutbox", outbox),
    )

    assert [kind for kind, _ in outbox.enqueued] == [
        "handoff",
        "attribution_event",
        "posterior_delta",
    ]
