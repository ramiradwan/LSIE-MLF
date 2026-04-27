"""
Service-level tests for `OperatorReadService`.

Focus:
  * §7C co-modulation null-reason propagation (null is valid).
  * §12 recovery_mode + operator_action_hint populated for degraded /
    recovering states only.
  * §7B reward-explanation fields (`p90_intensity`, `semantic_gate`,
    `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`)
    flow cleanly onto `EncounterSummary` / `LatestEncounterSummary`.
  * §4.C encounter state derivation (`REJECTED_GATE_CLOSED` when
    `semantic_gate = 0`, `REJECTED_NO_FRAMES` when n_frames = 0).
  * Alert synthesis from physio staleness / recently ended sessions.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from packages.schemas.operator_console import (
    AlertKind,
    EncounterState,
    HealthProbeState,
    HealthState,
)
from services.api.services.operator_read_service import OperatorReadService
from services.api.services.subsystem_probes import ProbeResult


def _cursor(results: list[Any]) -> MagicMock:
    """Build a mock cursor that returns the next element of `results` on each execute().

    The repo layer calls `cursor.description` then `fetchone()` / `fetchall()`.
    We stage (columns, rows) pairs so each repo call consumes one entry.
    """
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    state = {"i": 0}

    def _execute(*_args: Any, **_kwargs: Any) -> None:
        i = state["i"]
        if i >= len(results):
            cursor.description = None
            return
        columns, rows = results[i]
        cursor.description = [(c,) for c in columns]
        cursor.fetchall.return_value = list(rows)
        cursor.fetchone.return_value = rows[0] if rows else None
        state["i"] = i + 1

    cursor.execute.side_effect = _execute
    return cursor


def _service(
    cursor: MagicMock,
    now: datetime | None = None,
    *,
    redis_factory: Any | None = None,
    subsystem_probe_runner: Any | None = None,
) -> OperatorReadService:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    kwargs: dict[str, Any] = {}
    if subsystem_probe_runner is not None:
        kwargs["subsystem_probe_runner"] = subsystem_probe_runner
    return OperatorReadService(
        get_conn=lambda: conn,
        put_conn=lambda _c: None,
        redis_factory=redis_factory,
        clock=lambda: now or datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        **kwargs,
    )


class _FakeRedis:
    def __init__(self, payloads: dict[str, str]) -> None:
        self._payloads = payloads
        self.closed = False

    def get(self, name: str) -> str | None:
        return self._payloads.get(name)

    def close(self) -> None:
        self.closed = True


# ----------------------------------------------------------------------
# Live-session Redis overlay
# ----------------------------------------------------------------------


class TestLiveSessionOverlay:
    def test_get_session_merges_orchestrator_calibration_state(self) -> None:
        session_id = "33333333-3333-3333-3333-333333333333"
        cursor = _cursor(
            [
                (
                    [
                        "session_id",
                        "started_at",
                        "ended_at",
                        "duration_s",
                        "experiment_id",
                        "active_arm",
                        "last_segment_completed_at_utc",
                        "latest_reward",
                        "latest_semantic_gate",
                        "is_calibrating",
                        "calibration_frames_accumulated",
                        "calibration_frames_required",
                    ],
                    [
                        (
                            session_id,
                            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
                            None,
                            300.0,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    ],
                )
            ]
        )
        redis = _FakeRedis(
            {
                f"operator:live_session:{session_id}": (
                    '{"active_arm":"warm_welcome","expected_greeting":"hei rakas",'
                    '"is_calibrating":true,"calibration_frames_accumulated":12,'
                    '"calibration_frames_required":45}'
                )
            }
        )
        svc = _service(cursor, redis_factory=lambda: redis)

        import uuid

        summary = svc.get_session(uuid.UUID(session_id))
        assert summary is not None
        assert summary.active_arm == "warm_welcome"
        assert summary.expected_greeting == "hei rakas"
        assert summary.is_calibrating is True
        assert summary.calibration_frames_accumulated == 12
        assert summary.calibration_frames_required == 45
        assert redis.closed is True

    def test_get_session_missing_live_state_leaves_calibration_fields_none(self) -> None:
        session_id = "44444444-4444-4444-4444-444444444444"
        cursor = _cursor(
            [
                (
                    [
                        "session_id",
                        "started_at",
                        "ended_at",
                        "duration_s",
                        "experiment_id",
                        "active_arm",
                        "last_segment_completed_at_utc",
                        "latest_reward",
                        "latest_semantic_gate",
                        "is_calibrating",
                        "calibration_frames_accumulated",
                        "calibration_frames_required",
                    ],
                    [
                        (
                            session_id,
                            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
                            None,
                            60.0,
                            None,
                            "warm_welcome",
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    ],
                )
            ]
        )
        redis = _FakeRedis({})
        svc = _service(cursor, redis_factory=lambda: redis)

        import uuid

        summary = svc.get_session(uuid.UUID(session_id))
        assert summary is not None
        assert summary.active_arm == "warm_welcome"
        assert summary.is_calibrating is None
        assert summary.calibration_frames_accumulated is None
        assert summary.calibration_frames_required is None


# ----------------------------------------------------------------------
# §7C null-reason propagation
# ----------------------------------------------------------------------


class TestCoModulationNullReason:
    def test_null_index_surfaces_null_reason(self) -> None:
        session_id = "11111111-1111-1111-1111-111111111111"
        cursor = _cursor(
            [
                # fetch_session_by_id -> present, active session
                (
                    [
                        "session_id",
                        "started_at",
                        "ended_at",
                        "duration_s",
                        "experiment_id",
                        "active_arm",
                        "last_segment_completed_at_utc",
                        "latest_reward",
                        "latest_semantic_gate",
                    ],
                    [
                        (
                            session_id,
                            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
                            None,
                            300.0,
                            "greeting_line_v1",
                            "warm_welcome",
                            None,
                            None,
                            None,
                        )
                    ],
                ),
                # fetch_latest_physiology_rows -> empty
                (["subject_role"], []),
                # fetch_latest_comodulation_row -> null index + 0 pairs
                (
                    [
                        "session_id",
                        "window_end_utc",
                        "window_minutes",
                        "co_modulation_index",
                        "n_paired_observations",
                        "coverage_ratio",
                        "streamer_rmssd_mean",
                        "operator_rmssd_mean",
                        "created_at",
                    ],
                    [
                        (
                            session_id,
                            datetime(2026, 4, 17, 11, 55, tzinfo=UTC),
                            5,
                            None,
                            0,
                            0.0,
                            None,
                            None,
                            datetime(2026, 4, 17, 11, 56, tzinfo=UTC),
                        )
                    ],
                ),
            ]
        )
        svc = _service(cursor)
        import uuid

        snap = svc.get_session_physiology(uuid.UUID(session_id))
        assert snap is not None
        assert snap.comodulation is not None
        assert snap.comodulation.co_modulation_index is None
        assert snap.comodulation.null_reason is not None
        assert "insufficient" in snap.comodulation.null_reason.lower()

    def test_no_comod_row_still_yields_null_reason(self) -> None:
        session_id = "22222222-2222-2222-2222-222222222222"
        cursor = _cursor(
            [
                # session exists
                (
                    [
                        "session_id",
                        "started_at",
                        "ended_at",
                        "duration_s",
                        "experiment_id",
                        "active_arm",
                        "last_segment_completed_at_utc",
                        "latest_reward",
                        "latest_semantic_gate",
                    ],
                    [
                        (
                            session_id,
                            datetime(2026, 4, 17, 11, 0, tzinfo=UTC),
                            None,
                            60.0,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    ],
                ),
                # physio rows empty
                (["subject_role"], []),
                # no comod row at all
                (
                    [
                        "session_id",
                        "window_end_utc",
                        "window_minutes",
                        "co_modulation_index",
                        "n_paired_observations",
                        "coverage_ratio",
                        "streamer_rmssd_mean",
                        "operator_rmssd_mean",
                        "created_at",
                    ],
                    [],
                ),
            ]
        )
        svc = _service(cursor)
        import uuid

        snap = svc.get_session_physiology(uuid.UUID(session_id))
        assert snap is not None
        assert snap.comodulation is not None
        assert snap.comodulation.null_reason is not None


# ----------------------------------------------------------------------
# §12 recovery_mode / operator_action_hint
# ----------------------------------------------------------------------


class TestHealthRecoveryFields:
    def test_recovery_fields_populated_for_recovering_state(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        # All subsystems wrote within the recovering window except one
        # that wrote 45s ago -> degraded-but-recovering range (30-120s).
        recovering = now - timedelta(seconds=45)
        ok = now - timedelta(seconds=5)
        cursor = _cursor(
            [
                (
                    [
                        "last_metric_at",
                        "last_physio_at",
                        "last_comod_at",
                        "last_encounter_at",
                    ],
                    [(ok, recovering, ok, ok)],
                ),
            ]
        )
        svc = _service(cursor, now=now)
        snapshot = asyncio.run(svc.get_health())
        physio_row = next(row for row in snapshot.subsystems if row.subsystem_key == "physiology")
        assert physio_row.state is HealthState.RECOVERING
        assert physio_row.recovery_mode == "physio_adapter_retry"
        assert physio_row.operator_action_hint is not None

    def test_health_snapshot_emits_keyed_probe_dict_without_changing_rollup(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ok = now - timedelta(seconds=5)
        cursor = _cursor(
            [
                (
                    [
                        "last_metric_at",
                        "last_physio_at",
                        "last_comod_at",
                        "last_encounter_at",
                    ],
                    [(ok, ok, ok, ok)],
                ),
            ]
        )

        async def _probe_runner(**_kwargs: Any) -> list[ProbeResult]:
            return [
                ProbeResult(
                    subsystem_key="redis",
                    label="Redis Broker",
                    state=HealthProbeState.OK,
                    latency_ms=2.0,
                    detail="PING succeeded",
                )
            ]

        svc = _service(cursor, now=now, subsystem_probe_runner=_probe_runner)
        snapshot = asyncio.run(svc.get_health())

        assert snapshot.overall_state is HealthState.OK
        assert snapshot.degraded_count == 0
        assert snapshot.recovering_count == 0
        assert snapshot.error_count == 0
        assert list(snapshot.subsystem_probes) == ["redis"]
        assert snapshot.subsystem_probes["redis"].state is HealthProbeState.OK

    def test_recovery_fields_absent_for_ok_state(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ok = now - timedelta(seconds=5)
        cursor = _cursor(
            [
                (
                    [
                        "last_metric_at",
                        "last_physio_at",
                        "last_comod_at",
                        "last_encounter_at",
                    ],
                    [(ok, ok, ok, ok)],
                ),
            ]
        )
        svc = _service(cursor, now=now)
        snapshot = asyncio.run(svc.get_health())
        for row in snapshot.subsystems:
            assert row.state is HealthState.OK
            assert row.recovery_mode is None
            assert row.operator_action_hint is None

    def test_error_state_when_subsystem_silent(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        long_ago = now - timedelta(hours=2)
        cursor = _cursor(
            [
                (
                    [
                        "last_metric_at",
                        "last_physio_at",
                        "last_comod_at",
                        "last_encounter_at",
                    ],
                    [(long_ago, long_ago, long_ago, long_ago)],
                ),
            ]
        )
        svc = _service(cursor, now=now)
        snapshot = asyncio.run(svc.get_health())
        assert snapshot.overall_state is HealthState.ERROR
        for row in snapshot.subsystems:
            assert row.state is HealthState.ERROR
            assert row.recovery_mode is None

    def test_unknown_when_no_writes_yet(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        cursor = _cursor(
            [
                (
                    [
                        "last_metric_at",
                        "last_physio_at",
                        "last_comod_at",
                        "last_encounter_at",
                    ],
                    [(None, None, None, None)],
                ),
            ]
        )
        svc = _service(cursor, now=now)
        snapshot = asyncio.run(svc.get_health())
        assert snapshot.overall_state is HealthState.UNKNOWN


# ----------------------------------------------------------------------
# §7B reward explanation + encounter state derivation
# ----------------------------------------------------------------------


_ENC_COLS = [
    "id",
    "session_id",
    "segment_id",
    "experiment_id",
    "arm",
    "timestamp_utc",
    "gated_reward",
    "p90_intensity",
    "semantic_gate",
    "is_valid",
    "n_frames",
    "baseline_neutral",
    "stimulus_time",
    "created_at",
]

_ACOUSTIC_COLS = [
    "metrics_row_id",
    "pitch_f0",
    "jitter",
    "shimmer",
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
]

_ENC_WITH_ACOUSTIC_COLS = [*_ENC_COLS, *_ACOUSTIC_COLS]
_ALL_NULL_ACOUSTIC_VALUES = (None,) * len(_ACOUSTIC_COLS)


class TestEncounterExplanation:
    def test_completed_encounter_carries_all_reward_fields(self) -> None:
        session_id = "33333333-3333-3333-3333-333333333333"
        ts = datetime(2026, 4, 17, 11, 30, tzinfo=UTC)
        stim_epoch = ts.timestamp()
        cursor = _cursor(
            [
                (
                    _ENC_WITH_ACOUSTIC_COLS,
                    [
                        (
                            42,
                            session_id,
                            "seg-1",
                            "greeting_line_v1",
                            "warm_welcome",
                            ts,
                            0.88,
                            0.88,
                            1,
                            True,
                            120,
                            0.1,
                            stim_epoch,
                            ts,
                            *_ALL_NULL_ACOUSTIC_VALUES,
                        )
                    ],
                )
            ]
        )
        svc = _service(cursor)
        import uuid

        encounters = svc.list_encounters(uuid.UUID(session_id), limit=10)
        assert len(encounters) == 1
        enc = encounters[0]
        assert enc.state is EncounterState.COMPLETED
        assert enc.semantic_gate == 1
        assert enc.p90_intensity == 0.88
        assert enc.gated_reward == 0.88
        assert enc.n_frames_in_window == 120
        assert enc.baseline_b_neutral == 0.1
        assert enc.active_arm == "warm_welcome"
        assert enc.stimulus_time_utc is not None
        assert enc.stimulus_time_utc.tzinfo is UTC
        assert enc.acoustic is None
        assert enc.observational_acoustic is None
        assert enc.notes == []

    def test_completed_encounter_attaches_canonical_acoustic_metrics(self) -> None:
        session_id = "34343434-3434-3434-3434-343434343434"
        ts = datetime(2026, 4, 17, 11, 45, tzinfo=UTC)
        stim_epoch = ts.timestamp()
        cursor = _cursor(
            [
                (
                    _ENC_WITH_ACOUSTIC_COLS,
                    [
                        (
                            45,
                            session_id,
                            "seg-acoustic",
                            "greeting_line_v1",
                            "warm_welcome",
                            ts,
                            0.91,
                            0.91,
                            1,
                            True,
                            150,
                            0.05,
                            stim_epoch,
                            ts,
                            101,
                            215.5,
                            0.018,
                            0.027,
                            True,
                            False,
                            True,
                            False,
                            2.4,
                            0.0,
                            215.5,
                            None,
                            None,
                            0.018,
                            None,
                            None,
                            0.027,
                            None,
                            None,
                        )
                    ],
                )
            ]
        )
        svc = _service(cursor)
        import uuid

        [enc] = svc.list_encounters(uuid.UUID(session_id), limit=10)
        assert enc.acoustic is not None
        acoustic = enc.acoustic
        assert acoustic.pitch_f0 == 215.5
        assert acoustic.jitter == 0.018
        assert acoustic.shimmer == 0.027
        assert acoustic.f0_valid_measure is True
        assert isinstance(acoustic.f0_valid_measure, bool)
        assert acoustic.f0_valid_baseline is False
        assert isinstance(acoustic.f0_valid_baseline, bool)
        assert acoustic.perturbation_valid_measure is True
        assert acoustic.perturbation_valid_baseline is False
        assert acoustic.voiced_coverage_measure_s == 2.4
        assert acoustic.voiced_coverage_baseline_s == 0.0
        assert acoustic.f0_mean_measure_hz == 215.5
        assert acoustic.f0_mean_baseline_hz is None
        assert acoustic.f0_delta_semitones is None
        assert acoustic.jitter_mean_measure == 0.018
        assert acoustic.jitter_mean_baseline is None
        assert acoustic.jitter_delta is None
        assert acoustic.shimmer_mean_measure == 0.027
        assert acoustic.shimmer_mean_baseline is None
        assert acoustic.shimmer_delta is None
        assert enc.observational_acoustic is not None
        assert enc.observational_acoustic.f0_valid_measure is True
        assert enc.observational_acoustic.f0_valid_baseline is False
        assert enc.observational_acoustic.perturbation_valid_measure is True
        assert enc.observational_acoustic.perturbation_valid_baseline is False
        assert enc.observational_acoustic.voiced_coverage_measure_s == 2.4
        assert enc.observational_acoustic.voiced_coverage_baseline_s == 0.0
        assert enc.observational_acoustic.f0_mean_measure_hz == 215.5
        assert enc.observational_acoustic.f0_mean_baseline_hz is None
        assert enc.observational_acoustic.f0_delta_semitones is None
        assert enc.observational_acoustic.jitter_mean_measure == 0.018
        assert enc.observational_acoustic.jitter_mean_baseline is None
        assert enc.observational_acoustic.jitter_delta is None
        assert enc.observational_acoustic.shimmer_mean_measure == 0.027
        assert enc.observational_acoustic.shimmer_mean_baseline is None
        assert enc.observational_acoustic.shimmer_delta is None

    def test_latest_encounter_projection_preserves_acoustic_bool_and_null_defaults(self) -> None:
        svc = _service(_cursor([]))
        ts = datetime(2026, 4, 17, 11, 50, tzinfo=UTC)
        latest = svc._build_latest_encounter_summary(
            {
                "id": 46,
                "session_id": "35353535-3535-3535-3535-353535353535",
                "segment_id": "seg-null-acoustic",
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": ts,
                "gated_reward": 0.0,
                "p90_intensity": 0.0,
                "semantic_gate": 0,
                "is_valid": False,
                "n_frames": 0,
                "baseline_neutral": None,
                "stimulus_time": None,
                "created_at": ts,
                "metrics_row_id": 202,
                "pitch_f0": None,
                "jitter": None,
                "shimmer": None,
                "f0_valid_measure": None,
                "f0_valid_baseline": None,
                "perturbation_valid_measure": None,
                "perturbation_valid_baseline": None,
                "voiced_coverage_measure_s": None,
                "voiced_coverage_baseline_s": None,
                "f0_mean_measure_hz": None,
                "f0_mean_baseline_hz": None,
                "f0_delta_semitones": None,
                "jitter_mean_measure": None,
                "jitter_mean_baseline": None,
                "jitter_delta": None,
                "shimmer_mean_measure": None,
                "shimmer_mean_baseline": None,
                "shimmer_delta": None,
            }
        )
        assert latest is not None
        assert latest.acoustic is not None
        acoustic = latest.acoustic
        assert acoustic.pitch_f0 is None
        assert acoustic.jitter is None
        assert acoustic.shimmer is None
        assert acoustic.f0_valid_measure is False
        assert isinstance(acoustic.f0_valid_measure, bool)
        assert acoustic.f0_valid_baseline is False
        assert isinstance(acoustic.f0_valid_baseline, bool)
        assert acoustic.perturbation_valid_measure is False
        assert acoustic.perturbation_valid_baseline is False
        assert acoustic.voiced_coverage_measure_s == 0.0
        assert acoustic.voiced_coverage_baseline_s == 0.0
        assert acoustic.f0_mean_measure_hz is None
        assert acoustic.f0_mean_baseline_hz is None
        assert acoustic.f0_delta_semitones is None
        assert acoustic.jitter_mean_measure is None
        assert acoustic.jitter_mean_baseline is None
        assert acoustic.jitter_delta is None
        assert acoustic.shimmer_mean_measure is None
        assert acoustic.shimmer_mean_baseline is None
        assert acoustic.shimmer_delta is None
        assert latest.observational_acoustic is None

    def test_gate_closed_encounter_rejection(self) -> None:
        session_id = "44444444-4444-4444-4444-444444444444"
        ts = datetime(2026, 4, 17, 11, 30, tzinfo=UTC)
        cursor = _cursor(
            [
                (
                    _ENC_WITH_ACOUSTIC_COLS,
                    [
                        (
                            43,
                            session_id,
                            "seg-2",
                            "greeting_line_v1",
                            "warm_welcome",
                            ts,
                            0.0,
                            0.75,
                            0,
                            False,
                            130,
                            0.1,
                            None,
                            ts,
                            *_ALL_NULL_ACOUSTIC_VALUES,
                        )
                    ],
                )
            ]
        )
        svc = _service(cursor)
        import uuid

        [enc] = svc.list_encounters(uuid.UUID(session_id))
        assert enc.state is EncounterState.REJECTED_GATE_CLOSED
        assert enc.gated_reward == 0.0
        assert enc.semantic_gate == 0
        assert any("gate-closed" in n for n in enc.notes)

    def test_no_frames_encounter_rejection(self) -> None:
        session_id = "55555555-5555-5555-5555-555555555555"
        ts = datetime(2026, 4, 17, 11, 30, tzinfo=UTC)
        cursor = _cursor(
            [
                (
                    _ENC_WITH_ACOUSTIC_COLS,
                    [
                        (
                            44,
                            session_id,
                            "seg-3",
                            "greeting_line_v1",
                            "warm_welcome",
                            ts,
                            0.0,
                            0.0,
                            1,
                            False,
                            0,
                            None,
                            None,
                            ts,
                            *_ALL_NULL_ACOUSTIC_VALUES,
                        )
                    ],
                )
            ]
        )
        svc = _service(cursor)
        import uuid

        [enc] = svc.list_encounters(uuid.UUID(session_id))
        assert enc.state is EncounterState.REJECTED_NO_FRAMES
        assert enc.n_frames_in_window == 0
        assert any("no valid frames" in n for n in enc.notes)

    def test_latest_encounter_projection_keeps_inline_false_zero_acoustic_without_metrics_row(
        self,
    ) -> None:
        svc = _service(_cursor([]))
        ts = datetime(2026, 4, 17, 11, 55, tzinfo=UTC)
        latest = svc._build_latest_encounter_summary(
            {
                "id": 47,
                "session_id": "36363636-3636-3636-3636-363636363636",
                "segment_id": "seg-inline-acoustic",
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": ts,
                "gated_reward": 0.0,
                "p90_intensity": 0.0,
                "semantic_gate": 0,
                "is_valid": False,
                "n_frames": 0,
                "baseline_neutral": None,
                "stimulus_time": None,
                "created_at": ts,
                "pitch_f0": None,
                "jitter": None,
                "shimmer": None,
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
            }
        )

        assert latest is not None
        assert latest.acoustic is not None
        acoustic = latest.acoustic
        assert acoustic.f0_valid_measure is False
        assert isinstance(acoustic.f0_valid_measure, bool)
        assert acoustic.f0_valid_baseline is False
        assert isinstance(acoustic.f0_valid_baseline, bool)
        assert acoustic.perturbation_valid_measure is False
        assert isinstance(acoustic.perturbation_valid_measure, bool)
        assert acoustic.perturbation_valid_baseline is False
        assert isinstance(acoustic.perturbation_valid_baseline, bool)
        assert acoustic.voiced_coverage_measure_s == 0.0
        assert acoustic.voiced_coverage_baseline_s == 0.0
        assert acoustic.f0_mean_measure_hz is None
        assert acoustic.f0_mean_baseline_hz is None
        assert acoustic.f0_delta_semitones is None
        assert acoustic.jitter_mean_measure is None
        assert acoustic.shimmer_delta is None
        assert latest.observational_acoustic is not None
        assert latest.observational_acoustic.f0_valid_measure is False
        assert latest.observational_acoustic.f0_valid_baseline is False
        assert latest.observational_acoustic.perturbation_valid_measure is False
        assert latest.observational_acoustic.perturbation_valid_baseline is False
        assert latest.observational_acoustic.voiced_coverage_measure_s == 0.0
        assert latest.observational_acoustic.voiced_coverage_baseline_s == 0.0
        assert latest.observational_acoustic.f0_mean_measure_hz is None
        assert latest.observational_acoustic.shimmer_delta is None


# ----------------------------------------------------------------------
# Alert synthesis
# ----------------------------------------------------------------------


class TestAlertSynthesis:
    def test_physio_stale_rows_become_warning_alerts(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        session_id = "66666666-6666-6666-6666-666666666666"
        stale_at = now - timedelta(minutes=3)
        cursor = _cursor(
            [
                # fetch_recent_stale_physiology
                (
                    ["session_id", "subject_role", "created_at", "freshness_s"],
                    [(session_id, "streamer", stale_at, 42.5)],
                ),
                # fetch_recently_ended_sessions
                (["session_id", "ended_at"], []),
            ]
        )
        svc = _service(cursor, now=now)
        alerts = svc.list_alerts(limit=10)
        assert len(alerts) == 1
        assert alerts[0].kind is AlertKind.PHYSIOLOGY_STALE
        assert "streamer" in alerts[0].message
        assert alerts[0].subsystem_key == "physiology"

    def test_recently_ended_sessions_become_info_alerts(self) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        session_id = "77777777-7777-7777-7777-777777777777"
        ended_at = now - timedelta(minutes=2)
        cursor = _cursor(
            [
                # fetch_recent_stale_physiology
                (["session_id"], []),
                # fetch_recently_ended_sessions
                (
                    ["session_id", "ended_at"],
                    [(session_id, ended_at)],
                ),
            ]
        )
        svc = _service(cursor, now=now)
        alerts = svc.list_alerts(limit=10)
        assert len(alerts) == 1
        assert alerts[0].kind is AlertKind.SESSION_ENDED


# ----------------------------------------------------------------------
# Experiment detail active-arm readback
# ----------------------------------------------------------------------


class TestExperimentDetail:
    def test_arms_carry_posterior_and_variance(self) -> None:
        updated = datetime(2026, 4, 17, 11, 50, tzinfo=UTC)
        cursor = _cursor(
            [
                # fetch_experiment_arms -> management-column discovery
                (
                    ["column_name"],
                    [("label",), ("greeting_text",), ("enabled",), ("end_dated_at",)],
                ),
                # fetch_experiment_arms -> arm rows
                (
                    [
                        "experiment_id",
                        "label",
                        "arm",
                        "greeting_text",
                        "alpha_param",
                        "beta_param",
                        "enabled",
                        "end_dated_at",
                        "updated_at",
                        "selection_count",
                        "recent_reward_mean",
                        "recent_semantic_pass_rate",
                    ],
                    [
                        (
                            "greeting_line_v1",
                            "Greeting v1",
                            "warm_welcome",
                            "Hei lämmin",
                            4.0,
                            2.0,
                            True,
                            None,
                            updated,
                            15,
                            0.7,
                            0.8,
                        ),
                        (
                            "greeting_line_v1",
                            "Greeting v1",
                            "simple_hello",
                            "Hei yksinkertainen",
                            3.0,
                            3.0,
                            False,
                            updated,
                            updated,
                            10,
                            0.5,
                            0.6,
                        ),
                    ],
                ),
                # fetch_active_arm_for_experiment
                (
                    ["arm", "timestamp_utc"],
                    [("warm_welcome", updated)],
                ),
            ]
        )
        svc = _service(cursor)
        detail = svc.get_experiment_detail("greeting_line_v1")
        assert detail is not None
        assert detail.active_arm_id == "warm_welcome"
        assert len(detail.arms) == 2
        warm = next(a for a in detail.arms if a.arm_id == "warm_welcome")
        # Beta variance = 4*2 / ((6)^2 * 7) = 8/252
        assert warm.evaluation_variance is not None
        assert abs(warm.evaluation_variance - (8.0 / (36.0 * 7.0))) < 1e-9
        assert warm.posterior_alpha == 4.0
        assert warm.posterior_beta == 2.0
        assert warm.greeting_text == "Hei lämmin"
        assert warm.enabled is True
        assert detail.label == "Greeting v1"
        disabled = next(a for a in detail.arms if a.arm_id == "simple_hello")
        assert disabled.enabled is False
        assert disabled.end_dated_at == updated

    def test_empty_experiment_returns_none(self) -> None:
        cursor = _cursor(
            [
                # fetch_experiment_arms -> management-column discovery
                (
                    ["column_name"],
                    [("label",), ("greeting_text",), ("enabled",), ("end_dated_at",)],
                ),
                (
                    ["experiment_id"],
                    [],
                ),
            ]
        )
        svc = _service(cursor)
        assert svc.get_experiment_detail("missing") is None

    def test_legacy_schema_without_management_columns_still_reads(self) -> None:
        updated = datetime(2026, 4, 17, 11, 50, tzinfo=UTC)
        cursor = _cursor(
            [
                # fetch_experiment_arms -> management-column discovery sees legacy schema
                (["column_name"], []),
                # fetch_experiment_arms -> fallback projection provides defaulted fields
                (
                    [
                        "experiment_id",
                        "label",
                        "arm",
                        "greeting_text",
                        "alpha_param",
                        "beta_param",
                        "enabled",
                        "end_dated_at",
                        "updated_at",
                        "selection_count",
                        "recent_reward_mean",
                        "recent_semantic_pass_rate",
                    ],
                    [
                        (
                            "legacy_exp",
                            "legacy_exp",
                            "legacy_arm",
                            "legacy_arm",
                            1.0,
                            1.0,
                            True,
                            None,
                            updated,
                            0,
                            None,
                            None,
                        )
                    ],
                ),
                # fetch_active_arm_for_experiment
                (["arm", "timestamp_utc"], []),
            ]
        )
        svc = _service(cursor)

        detail = svc.get_experiment_detail("legacy_exp")

        assert detail is not None
        assert detail.label == "legacy_exp"
        assert detail.arms[0].arm_id == "legacy_arm"
        assert detail.arms[0].greeting_text == "legacy_arm"
        assert detail.arms[0].enabled is True
        arm_sql = cursor.execute.call_args_list[1].args[0]
        assert "ex.label" not in arm_sql
        assert "ex.greeting_text" not in arm_sql
        assert "ex.enabled" not in arm_sql
        assert "ex.end_dated_at" not in arm_sql
