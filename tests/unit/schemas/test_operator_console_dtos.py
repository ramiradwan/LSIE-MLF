"""Unit tests for operator-console shared DTOs.

Focus (per Phase 1 done-when):
  * UTC-aware timestamps are required; naive datetimes are rejected.
  * `CoModulationSummary.null_reason` surfaces when
    `co_modulation_index is None` (§7C null-valid semantics).
  * `StimulusRequest.client_action_id` is the API Server dedup key and
    must be present (§4.C authoritative stimulus bookkeeping).
  * Enum values round-trip through JSON.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    AttributionSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    ObservationalAcousticSummary,
    OperatorEventEnvelope,
    OperatorStateBootstrap,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
    StimulusRequest,
    UiStatusKind,
)


def _utc(year: int = 2026, month: int = 1, day: int = 1, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, 0, tzinfo=UTC)


# ----------------------------------------------------------------------
# UTC enforcement
# ----------------------------------------------------------------------


class TestUtcTimestampEnforcement:
    def test_session_summary_rejects_naive_started_at(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)  # no tzinfo
        with pytest.raises(ValidationError) as info:
            SessionSummary(
                session_id=uuid.uuid4(),
                status="active",
                started_at_utc=naive,
            )
        assert "UTC-aware" in str(info.value)

    def test_session_summary_accepts_utc_aware(self) -> None:
        s = SessionSummary(
            session_id=uuid.uuid4(),
            status="active",
            started_at_utc=_utc(),
        )
        assert s.started_at_utc.tzinfo is UTC

    def test_session_summary_accepts_non_utc_offset(self) -> None:
        # Any tz-aware datetime is acceptable — rejection is only for naive.
        aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
        s = SessionSummary(
            session_id=uuid.uuid4(),
            status="active",
            started_at_utc=aware,
        )
        assert s.started_at_utc.utcoffset() is not None

    def test_encounter_summary_rejects_naive_stimulus_time(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            EncounterSummary(
                encounter_id="e-1",
                session_id=uuid.uuid4(),
                segment_timestamp_utc=_utc(),
                state=EncounterState.COMPLETED,
                stimulus_time_utc=naive,
            )

    def test_physiology_snapshot_rejects_naive_source_timestamp(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=42.5,
                heart_rate_bpm=68,
                provider="oura",
                source_timestamp_utc=naive,
                freshness_s=5.0,
                is_stale=False,
            )

    def test_health_snapshot_requires_utc_generated_at(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            HealthSnapshot(
                generated_at_utc=naive,
                overall_state=HealthState.OK,
            )

    def test_stimulus_accepted_requires_utc_received_at(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            StimulusAccepted(
                session_id=uuid.uuid4(),
                client_action_id=uuid.uuid4(),
                accepted=True,
                received_at_utc=naive,
            )

    def test_stimulus_accepted_requires_utc_stimulus_time(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValidationError):
            StimulusAccepted(
                session_id=uuid.uuid4(),
                client_action_id=uuid.uuid4(),
                accepted=True,
                received_at_utc=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                stimulus_time_utc=naive,
            )


# ----------------------------------------------------------------------
# SSE state DTOs
# ----------------------------------------------------------------------


class TestOperatorStateEvents:
    def test_event_envelope_accepts_existing_payload_dto(self) -> None:
        payload = OverviewSnapshot(generated_at_utc=_utc())
        envelope = OperatorEventEnvelope(
            event_id="overview:1",
            event_type="overview",
            cursor="overview:1",
            generated_at_utc=_utc(),
            payload=payload,
        )
        assert envelope.payload == payload

    def test_event_envelope_rejects_unknown_event_type(self) -> None:
        with pytest.raises(ValidationError):
            OperatorEventEnvelope.model_validate(
                {
                    "event_id": "bad:1",
                    "event_type": "unknown",
                    "cursor": "bad:1",
                    "generated_at_utc": _utc(),
                    "payload": [],
                }
            )

    def test_bootstrap_reuses_existing_payload_types(self) -> None:
        health = HealthSnapshot(generated_at_utc=_utc(), overall_state=HealthState.OK)
        overview = OverviewSnapshot(generated_at_utc=_utc(), health=health)
        bootstrap = OperatorStateBootstrap(
            generated_at_utc=_utc(),
            overview=overview,
            health=health,
        )
        assert bootstrap.overview is overview
        assert bootstrap.health is health
        assert bootstrap.sessions == []


# ----------------------------------------------------------------------
# §7C null-valid co-modulation
# ----------------------------------------------------------------------


class TestCoModulationNullValid:
    def test_null_index_with_reason_is_accepted(self) -> None:
        summary = CoModulationSummary(
            session_id=uuid.uuid4(),
            co_modulation_index=None,
            n_paired_observations=0,
            coverage_ratio=0.0,
            null_reason="insufficient aligned non-stale pairs",
        )
        assert summary.co_modulation_index is None
        assert summary.null_reason == "insufficient aligned non-stale pairs"

    def test_populated_index_survives_roundtrip(self) -> None:
        summary = CoModulationSummary(
            session_id=uuid.uuid4(),
            co_modulation_index=0.42,
            n_paired_observations=12,
            coverage_ratio=0.8,
            window_start_utc=_utc(),
            window_end_utc=_utc(hour=13),
        )
        restored = CoModulationSummary.model_validate_json(summary.model_dump_json())
        assert restored == summary

    @pytest.mark.parametrize("value", [-1.5, 1.5])
    def test_index_is_bounded_to_pearson_range(self, value: float) -> None:
        with pytest.raises(ValidationError):
            CoModulationSummary(
                session_id=uuid.uuid4(),
                co_modulation_index=value,
                n_paired_observations=5,
                coverage_ratio=0.5,
            )

    def test_coverage_ratio_bounded_0_to_1(self) -> None:
        with pytest.raises(ValidationError):
            CoModulationSummary(
                session_id=uuid.uuid4(),
                co_modulation_index=0.1,
                n_paired_observations=5,
                coverage_ratio=1.5,
            )


# ----------------------------------------------------------------------
# §7D observational acoustic payload
# ----------------------------------------------------------------------


class TestObservationalAcousticSummary:
    def test_defaults_follow_optional_null_contract(self) -> None:
        summary = ObservationalAcousticSummary()
        dumped = json.loads(summary.model_dump_json())
        assert dumped == {
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

    def test_field_names_match_canonical_section_7d_schema(self) -> None:
        assert tuple(ObservationalAcousticSummary.model_fields.keys()) == (
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
        )

    def test_non_finite_floats_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObservationalAcousticSummary(f0_mean_measure_hz=float("nan"))

    def test_negative_non_negative_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObservationalAcousticSummary(voiced_coverage_measure_s=-0.1)

    def test_encounter_summaries_default_observational_acoustic_to_none(self) -> None:
        encounter = EncounterSummary(
            encounter_id="e-no-acoustic",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
        )
        latest = LatestEncounterSummary(
            encounter_id="e-latest-no-acoustic",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
        )
        assert encounter.observational_acoustic is None
        assert latest.observational_acoustic is None

    def test_encounter_summary_roundtrips_nested_observational_acoustic(self) -> None:
        summary = EncounterSummary(
            encounter_id="e-observational-acoustic-1",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
            semantic_gate=1,
            p90_intensity=0.91,
            gated_reward=0.91,
            observational_acoustic=ObservationalAcousticSummary(
                f0_valid_measure=True,
                f0_valid_baseline=True,
                perturbation_valid_measure=True,
                perturbation_valid_baseline=True,
                voiced_coverage_measure_s=2.5,
                voiced_coverage_baseline_s=2.0,
                f0_mean_measure_hz=210.0,
                f0_mean_baseline_hz=180.0,
                f0_delta_semitones=2.667,
                jitter_mean_measure=0.012,
                jitter_mean_baseline=0.009,
                jitter_delta=0.003,
                shimmer_mean_measure=0.034,
                shimmer_mean_baseline=0.03,
                shimmer_delta=0.004,
            ),
        )
        restored = EncounterSummary.model_validate_json(summary.model_dump_json())
        assert restored.observational_acoustic is not None
        assert restored.observational_acoustic.f0_valid_measure is True
        assert restored.observational_acoustic.f0_mean_measure_hz == pytest.approx(210.0)


# ----------------------------------------------------------------------
# §7E semantic / attribution payloads
# ----------------------------------------------------------------------


class TestSemanticAndAttributionSummaries:
    def test_optional_nested_fields_default_to_none_on_encounter_summaries(self) -> None:
        encounter = EncounterSummary(
            encounter_id="e-no-attribution",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
        )
        latest = LatestEncounterSummary(
            encounter_id="e-latest-no-attribution",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
        )

        assert encounter.semantic_evaluation is None
        assert encounter.attribution is None
        assert latest.semantic_evaluation is None
        assert latest.attribution is None

    def test_field_names_match_section_7e_operator_summary_contract(self) -> None:
        assert tuple(SemanticEvaluationSummary.model_fields.keys()) == (
            "reasoning",
            "is_match",
            "confidence_score",
            "semantic_method",
            "semantic_method_version",
        )
        assert tuple(AttributionSummary.model_fields.keys()) == (
            "finality",
            "soft_reward_candidate",
            "au12_baseline_pre",
            "au12_lift_p90",
            "au12_lift_peak",
            "au12_peak_latency_ms",
            "sync_peak_corr",
            "sync_peak_lag",
            "outcome_link_lag_s",
        )

    def test_encounter_summary_roundtrips_nested_semantic_and_attribution(self) -> None:
        summary = EncounterSummary(
            encounter_id="e-semantic-attribution-1",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
            semantic_gate=0,
            p90_intensity=0.0,
            gated_reward=0.0,
            semantic_evaluation=SemanticEvaluationSummary(
                reasoning="cross_encoder_high_nonmatch",
                is_match=False,
                confidence_score=0.0,
                semantic_method="cross_encoder",
                semantic_method_version="ce-v1.2.3",
            ),
            attribution=AttributionSummary(
                finality="online_provisional",
                soft_reward_candidate=0.0,
                au12_baseline_pre=0.0,
                au12_lift_p90=0.0,
                au12_lift_peak=0.04,
                au12_peak_latency_ms=125.0,
                sync_peak_corr=0.0,
                sync_peak_lag=0,
                outcome_link_lag_s=0.0,
            ),
        )

        restored = EncounterSummary.model_validate_json(summary.model_dump_json())
        assert restored.semantic_evaluation is not None
        assert restored.semantic_evaluation.is_match is False
        assert restored.semantic_evaluation.confidence_score == 0.0
        assert restored.attribution is not None
        assert restored.attribution.soft_reward_candidate == 0.0
        assert restored.attribution.sync_peak_lag == 0
        assert isinstance(restored.attribution.sync_peak_lag, int)
        assert restored.attribution.outcome_link_lag_s == 0.0

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_semantic_confidence_is_probability(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationSummary(confidence_score=confidence)

    @pytest.mark.parametrize("corr", [-1.1, 1.1])
    def test_sync_peak_corr_is_bounded_to_pearson_range(self, corr: float) -> None:
        with pytest.raises(ValidationError):
            AttributionSummary(sync_peak_corr=corr)

    def test_attribution_finality_is_restricted_to_spec_values(self) -> None:
        with pytest.raises(ValidationError):
            AttributionSummary.model_validate({"finality": "draft"})

    @pytest.mark.parametrize("soft_reward", [-0.1, 1.1])
    def test_soft_reward_candidate_is_probability(self, soft_reward: float) -> None:
        with pytest.raises(ValidationError):
            AttributionSummary(soft_reward_candidate=soft_reward)

    def test_sync_peak_lag_is_non_negative_integer(self) -> None:
        with pytest.raises(ValidationError):
            AttributionSummary(sync_peak_lag=-1)
        with pytest.raises(ValidationError):
            AttributionSummary.model_validate({"sync_peak_lag": 1.5})


# ----------------------------------------------------------------------
# StimulusRequest dedup key
# ----------------------------------------------------------------------


class TestStimulusRequestDedupKey:
    def test_client_action_id_is_required(self) -> None:
        with pytest.raises(ValidationError) as info:
            StimulusRequest(operator_note="test")  # type: ignore[call-arg]
        assert "client_action_id" in str(info.value)

    def test_client_action_id_accepted_as_uuid(self) -> None:
        action_id = uuid.uuid4()
        request = StimulusRequest(client_action_id=action_id, operator_note=None)
        assert request.client_action_id == action_id
        assert isinstance(request.client_action_id, UUID)

    def test_client_action_id_roundtrips_through_json(self) -> None:
        action_id = uuid.uuid4()
        request = StimulusRequest(client_action_id=action_id, operator_note="pinned")
        restored = StimulusRequest.model_validate_json(request.model_dump_json())
        assert restored.client_action_id == action_id
        assert restored.operator_note == "pinned"

    def test_operator_note_defaults_to_none(self) -> None:
        request = StimulusRequest(client_action_id=uuid.uuid4())
        assert request.operator_note is None


class TestSessionLifecycleDtos:
    def test_session_create_request_trims_and_roundtrips(self) -> None:
        action_id = uuid.uuid4()
        request = SessionCreateRequest(
            stream_url="  https://example.com/live  ",
            experiment_id="  exp-1  ",
            client_action_id=action_id,
        )
        restored = SessionCreateRequest.model_validate_json(request.model_dump_json())
        assert restored.stream_url == "https://example.com/live"
        assert restored.experiment_id == "exp-1"
        assert restored.client_action_id == action_id

    def test_session_create_request_rejects_blank_values(self) -> None:
        with pytest.raises(ValidationError):
            SessionCreateRequest(
                stream_url="   ",
                experiment_id="exp-1",
                client_action_id=uuid.uuid4(),
            )
        with pytest.raises(ValidationError):
            SessionCreateRequest(
                stream_url="https://example.com/live",
                experiment_id="   ",
                client_action_id=uuid.uuid4(),
            )

    def test_session_create_request_rejects_invalid_stream_url(self) -> None:
        with pytest.raises(ValidationError):
            SessionCreateRequest(
                stream_url="123",
                experiment_id="exp-1",
                client_action_id=uuid.uuid4(),
            )

    def test_session_end_request_requires_client_action_id(self) -> None:
        with pytest.raises(ValidationError):
            SessionEndRequest()  # type: ignore[call-arg]

    def test_session_lifecycle_accepted_requires_utc_received_at(self) -> None:
        with pytest.raises(ValidationError):
            SessionLifecycleAccepted(
                action="start",
                session_id=uuid.uuid4(),
                client_action_id=uuid.uuid4(),
                accepted=True,
                received_at_utc=datetime(2026, 1, 1, 12, 0),
            )

    def test_session_lifecycle_accepted_roundtrips(self) -> None:
        accepted = SessionLifecycleAccepted(
            action="end",
            session_id=uuid.uuid4(),
            client_action_id=uuid.uuid4(),
            accepted=True,
            received_at_utc=_utc(),
            message="duplicate submission suppressed",
        )
        restored = SessionLifecycleAccepted.model_validate_json(accepted.model_dump_json())
        assert restored == accepted


# ----------------------------------------------------------------------
# Enum round-trip
# ----------------------------------------------------------------------


class TestEnumRoundTrip:
    def test_ui_status_kind_values(self) -> None:
        for kind in UiStatusKind:
            assert UiStatusKind(kind.value) is kind

    def test_alert_severity_values(self) -> None:
        for severity in AlertSeverity:
            assert AlertSeverity(severity.value) is severity

    def test_alert_kind_values(self) -> None:
        for kind in AlertKind:
            assert AlertKind(kind.value) is kind

    def test_encounter_state_values(self) -> None:
        for state in EncounterState:
            assert EncounterState(state.value) is state

    def test_stimulus_action_state_values(self) -> None:
        for state in StimulusActionState:
            assert StimulusActionState(state.value) is state

    def test_health_state_values(self) -> None:
        for state in HealthState:
            assert HealthState(state.value) is state

    def test_alert_event_enum_serializes_as_value(self) -> None:
        event = AlertEvent(
            alert_id="a-1",
            severity=AlertSeverity.CRITICAL,
            kind=AlertKind.SUBSYSTEM_ERROR,
            message="capture offline",
            subsystem_key="capture",
            emitted_at_utc=_utc(),
        )
        dumped = json.loads(event.model_dump_json())
        assert dumped["severity"] == "critical"
        assert dumped["kind"] == "subsystem_error"
        restored = AlertEvent.model_validate_json(event.model_dump_json())
        assert restored.severity is AlertSeverity.CRITICAL
        assert restored.kind is AlertKind.SUBSYSTEM_ERROR

    def test_encounter_state_in_encounter_summary_roundtrips(self) -> None:
        summary = EncounterSummary(
            encounter_id="e-1",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.REJECTED_GATE_CLOSED,
            semantic_gate=0,
            p90_intensity=0.3,
            gated_reward=0.0,
        )
        restored = EncounterSummary.model_validate_json(summary.model_dump_json())
        assert restored.state is EncounterState.REJECTED_GATE_CLOSED


# ----------------------------------------------------------------------
# Composite DTOs
# ----------------------------------------------------------------------


class TestCompositeDtos:
    def test_overview_snapshot_accepts_all_null_inner_cards(self) -> None:
        snap = OverviewSnapshot(generated_at_utc=_utc())
        assert snap.active_session is None
        assert snap.latest_encounter is None
        assert snap.experiment_summary is None
        assert snap.physiology is None
        assert snap.health is None
        assert snap.alerts == []

    def test_overview_snapshot_validates_nested_dtos(self) -> None:
        snap = OverviewSnapshot(
            generated_at_utc=_utc(),
            active_session=SessionSummary(
                session_id=uuid.uuid4(),
                status="active",
                started_at_utc=_utc(),
            ),
            latest_encounter=LatestEncounterSummary(
                encounter_id="e-1",
                session_id=uuid.uuid4(),
                segment_timestamp_utc=_utc(),
                state=EncounterState.COMPLETED,
                p90_intensity=0.87,
                gated_reward=0.87,
                semantic_gate=1,
                n_frames_in_window=120,
            ),
        )
        assert snap.latest_encounter is not None
        assert snap.latest_encounter.gated_reward == pytest.approx(0.87)

    def test_session_physiology_snapshot_bundles_roles_and_comodulation(self) -> None:
        session_id = uuid.uuid4()
        snap = SessionPhysiologySnapshot(
            session_id=session_id,
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=40.0,
                heart_rate_bpm=70,
                provider="oura",
                source_timestamp_utc=_utc(),
                freshness_s=10.0,
                is_stale=False,
            ),
            operator=PhysiologyCurrentSnapshot(
                subject_role="operator",
                rmssd_ms=None,
                heart_rate_bpm=None,
                provider=None,
                source_timestamp_utc=None,
                freshness_s=None,
                is_stale=None,
            ),
            comodulation=CoModulationSummary(
                session_id=session_id,
                co_modulation_index=None,
                n_paired_observations=0,
                coverage_ratio=0.0,
                null_reason="insufficient aligned non-stale pairs",
            ),
            generated_at_utc=_utc(),
        )
        assert snap.operator is not None
        assert snap.operator.rmssd_ms is None
        assert snap.comodulation is not None
        assert snap.comodulation.null_reason is not None

    def test_experiment_detail_carries_arm_posteriors(self) -> None:
        detail = ExperimentDetail(
            experiment_id="greeting_line_v1",
            label="greetings-v1",
            active_arm_id="arm-a",
            arms=[
                ArmSummary(
                    arm_id="arm-a",
                    stimulus_definition=StimulusDefinition(
                        stimulus_modality="spoken_greeting",
                        stimulus_payload=StimulusPayload(text="hi"),
                        expected_stimulus_rule="Deliver the spoken greeting to the creator",
                        expected_response_rule="The live streamer acknowledges the greeting",
                    ),
                    posterior_alpha=2.0,
                    posterior_beta=3.0,
                    evaluation_variance=0.04,
                    selection_count=12,
                    recent_reward_mean=0.7,
                    recent_semantic_pass_rate=0.85,
                ),
            ],
            last_updated_utc=_utc(),
        )
        assert detail.arms[0].posterior_alpha == pytest.approx(2.0)
        assert detail.arms[0].posterior_beta == pytest.approx(3.0)

    def test_arm_summary_rejects_non_positive_posterior(self) -> None:
        with pytest.raises(ValidationError):
            ArmSummary(
                arm_id="arm-a",
                stimulus_definition=StimulusDefinition(
                    stimulus_modality="spoken_greeting",
                    stimulus_payload=StimulusPayload(text="hi"),
                    expected_stimulus_rule="Deliver the spoken greeting to the creator",
                    expected_response_rule="The live streamer acknowledges the greeting",
                ),
                posterior_alpha=0.0,
                posterior_beta=1.0,
            )

    def test_health_subsystem_carries_recovery_fields(self) -> None:
        row = HealthSubsystemStatus(
            subsystem_key="capture",
            label="Capture Container",
            state=HealthState.RECOVERING,
            last_success_utc=_utc(),
            detail="FFmpeg restarting after pipe overflow",
            recovery_mode="ffmpeg_restart",
            operator_action_hint="monitor; intervene if still recovering in 60s",
        )
        assert row.recovery_mode == "ffmpeg_restart"
        assert row.operator_action_hint is not None

    def test_health_snapshot_carries_probe_rows_without_changing_rollup_state(self) -> None:
        probe = HealthSubsystemProbe(
            subsystem_key="azure_openai",
            label="Azure OpenAI",
            state=HealthProbeState.NOT_CONFIGURED,
            latency_ms=None,
            detail="missing AZURE_OPENAI_ENDPOINT",
            checked_at_utc=_utc(),
        )
        snap = HealthSnapshot(
            generated_at_utc=_utc(),
            overall_state=HealthState.OK,
            subsystem_probes={probe.subsystem_key: probe},
        )
        assert snap.overall_state is HealthState.OK
        assert snap.subsystem_probes == {probe.subsystem_key: probe}

    def test_health_snapshot_rejects_probe_dict_key_mismatch(self) -> None:
        probe = HealthSubsystemProbe(
            subsystem_key="redis",
            label="Redis Broker",
            state=HealthProbeState.OK,
            checked_at_utc=_utc(),
        )
        with pytest.raises(ValidationError):
            HealthSnapshot(
                generated_at_utc=_utc(),
                overall_state=HealthState.OK,
                subsystem_probes={"postgres": probe},
            )

    def test_experiment_summary_rejects_negative_arm_count(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentSummary(experiment_id="greeting_line_v1", arm_count=-1)


# ----------------------------------------------------------------------
# Reward-explanation field completeness (§7B)
# ----------------------------------------------------------------------


class TestRewardExplanationFields:
    def test_encounter_summary_carries_all_reward_explanation_fields(self) -> None:
        enc = EncounterSummary(
            encounter_id="e-1",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.COMPLETED,
            active_arm="arm-a",
            expected_response_text="hi there",
            stimulus_time_utc=_utc(hour=13),
            semantic_gate=1,
            semantic_confidence=0.92,
            p90_intensity=0.88,
            gated_reward=0.88,
            n_frames_in_window=120,
            au12_baseline_pre=0.1,
            physiology_attached=True,
            physiology_stale=False,
            notes=["gate-open"],
        )
        # r_t = p90_intensity × semantic_gate per §7B
        assert enc.semantic_gate == 1
        assert enc.gated_reward == pytest.approx(enc.p90_intensity or 0.0)
        assert enc.au12_baseline_pre == pytest.approx(0.1)
        assert enc.n_frames_in_window == 120

    def test_gated_reward_is_zero_when_gate_closed(self) -> None:
        enc = EncounterSummary(
            encounter_id="e-2",
            session_id=uuid.uuid4(),
            segment_timestamp_utc=_utc(),
            state=EncounterState.REJECTED_GATE_CLOSED,
            p90_intensity=0.75,
            semantic_gate=0,
            gated_reward=0.0,
            n_frames_in_window=120,
        )
        # The DTO does not enforce the product — that's the pipeline's job —
        # but it must be able to represent a gate-closed encounter cleanly.
        assert enc.gated_reward == 0.0
        assert enc.state is EncounterState.REJECTED_GATE_CLOSED
