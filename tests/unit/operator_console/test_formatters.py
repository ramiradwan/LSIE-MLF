"""Tests for operator-language formatters — Phase 3.

Each test targets one of the operator-trust-critical strings: the §7B
reward explanation (gate-closed + no-frames branches), §4.C.4 freshness
wording, §7C co-modulation null-reason, and the §12 health mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from packages.schemas.operator_console import (
    ArmSummary,
    AttributionSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    HealthState,
    HealthSubsystemStatus,
    ObservationalAcousticSummary,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusActionState,
    UiStatusKind,
)
from services.operator_console.formatters import (
    build_acoustic_detail_display,
    build_acoustic_explanation,
    build_cause_effect_display,
    build_co_modulation_display,
    build_health_detail,
    build_live_telemetry_display,
    build_physiology_explanation,
    build_readiness_display,
    build_reward_explanation,
    build_semantic_attribution_diagnostics_display,
    build_strategy_evidence_display,
    format_acoustic_ratio,
    format_acoustic_seconds,
    format_acoustic_validity,
    format_acoustic_validity_pill,
    format_acoustic_voiced_coverage,
    format_attribution_finality_label,
    format_au12_lift_metrics,
    format_au12_peak_latency,
    format_bounded_reason_code_label,
    format_calibration_status,
    format_comodulation_index,
    format_duration,
    format_f0_hz,
    format_freshness,
    format_health_state,
    format_outcome_link_lag,
    format_percentage,
    format_perturbation_delta,
    format_probability_confidence,
    format_reward,
    format_semantic_confidence,
    format_semantic_gate,
    format_semantic_method_label,
    format_semitone_delta,
    format_session_id_compact,
    format_soft_reward_candidate,
    format_synchrony_metrics,
    format_timestamp,
    operator_ready_for_submit,
    semantic_attribution_diagnostics_for_encounter,
    semantic_confidence_for_encounter,
    truncate_expected_greeting,
    ui_status_for_health,
)

_SESSION_ID = UUID("00000000-0000-0000-0000-000000000005")


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------


class TestPrimitives:
    def test_timestamp_none_returns_em_dash(self) -> None:
        assert format_timestamp(None) == "—"

    def test_timestamp_utc_renders_z_suffix(self) -> None:
        assert format_timestamp(_utc(2026, 4, 17, 11, 30)) == "2026-04-17 11:30:00Z"

    def test_duration_formats_hms(self) -> None:
        assert format_duration(3725.0) == "1h 2m 5s"
        assert format_duration(65.0) == "1m 5s"
        assert format_duration(5.0) == "5s"
        assert format_duration(None) == "—"
        assert format_duration(-1.0) == "—"

    def test_percentage_respects_digits(self) -> None:
        assert format_percentage(0.1234) == "12.3%"
        assert format_percentage(0.5, digits=0) == "50%"
        assert format_percentage(None) == "—"

    def test_reward_defaults_to_three_decimals(self) -> None:
        assert format_reward(0.12345) == "0.123"
        assert format_reward(None) == "—"

    def test_session_id_compact_keeps_edges(self) -> None:
        session_id = UUID("12345678-1234-5678-9abc-123456789abc")
        assert format_session_id_compact(session_id) == "12345678…9abc"
        assert format_session_id_compact(None) == "—"


# ----------------------------------------------------------------------
# §7B semantic gate
# ----------------------------------------------------------------------


class TestSemanticGate:
    def test_gate_open_text(self) -> None:
        assert format_semantic_gate(1) == "yes — reward can count"

    def test_gate_closed_text_calls_out_reward_being_held_back(self) -> None:
        assert "held back" in format_semantic_gate(0)

    def test_gate_none_renders_em_dash(self) -> None:
        assert format_semantic_gate(None) == "—"

    def test_semantic_confidence_is_percentage(self) -> None:
        assert format_semantic_confidence(0.87) == "87%"
        assert format_semantic_confidence(None) == "—"


# ----------------------------------------------------------------------
# §7E / §8 semantic-attribution diagnostics
# ----------------------------------------------------------------------


class TestSemanticAttributionDiagnostics:
    def test_semantic_method_labels_cover_cross_encoder_and_gray_band(self) -> None:
        assert format_semantic_method_label("cross_encoder", "ce-v1") == (
            "local stimulus confirmation checker · ce-v1"
        )
        assert format_semantic_method_label("llm_gray_band", "llm-v2") == (
            "backup stimulus confirmation checker · llm-v2"
        )
        assert format_semantic_method_label(None) == "—"

    def test_bounded_reason_code_labels_are_operator_friendly(self) -> None:
        assert format_bounded_reason_code_label("cross_encoder_high_match") == (
            "Stimulus was clearly confirmed"
        )
        assert format_bounded_reason_code_label("gray_band_llm_nonmatch") == (
            "Backup checker did not confirm the stimulus"
        )
        assert "{" not in format_bounded_reason_code_label("semantic_timeout")
        assert format_bounded_reason_code_label(None) == "—"

    def test_probability_finality_soft_reward_and_lag_labels(self) -> None:
        assert format_probability_confidence(0.834) == "confirmation confidence 83%"
        assert format_probability_confidence(None) == "—"
        assert format_attribution_finality_label("online_provisional") == "online provisional"
        assert format_attribution_finality_label("offline_final") == "offline final"
        assert format_soft_reward_candidate(0.4264) == "possible follow-up reward 0.426"
        assert format_outcome_link_lag(42.25) == "after 42.2s"

    def test_attribution_metric_formatters_render_au12_and_synchrony(self) -> None:
        attribution = AttributionSummary(
            finality="online_provisional",
            soft_reward_candidate=0.426,
            au12_baseline_pre=0.12,
            au12_lift_p90=0.42,
            au12_lift_peak=0.68,
            au12_peak_latency_ms=1250.0,
            sync_peak_corr=-0.314,
            sync_peak_lag=3,
            outcome_link_lag_s=42.0,
        )
        assert format_au12_lift_metrics(attribution) == (
            "before stimulus 0.120 · strong response lift 0.420 · peak response lift 0.680"
        )
        assert format_au12_peak_latency(attribution) == "1.25s"
        assert format_synchrony_metrics(attribution) == "movement together -0.314 · lag 3"
        assert format_au12_lift_metrics(None) == "—"
        assert format_au12_peak_latency(None) == "—"
        assert format_synchrony_metrics(None) == "—"

    def test_display_combines_direct_cross_encoder_diagnostics(self) -> None:
        display = build_semantic_attribution_diagnostics_display(
            SemanticEvaluationSummary(
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=0.91,
                semantic_method="cross_encoder",
                semantic_method_version="ce-v1",
            ),
            AttributionSummary(
                finality="offline_final",
                soft_reward_candidate=0.77,
                au12_baseline_pre=0.10,
                au12_lift_p90=0.55,
                sync_peak_corr=0.401,
                outcome_link_lag_s=15.0,
            ),
        )
        assert display.has_diagnostics is True
        assert display.semantic_method == "local stimulus confirmation checker · ce-v1"
        assert display.bounded_reason_code == "Stimulus was clearly confirmed"
        assert display.probability_confidence == "confirmation confidence 91%"
        assert display.match_result == "match"
        assert display.attribution_finality == "offline final"
        assert "do not change the reward" in display.observational_note

    def test_display_combines_gray_band_fallback_diagnostics_from_encounter(self) -> None:
        encounter = _encounter(
            semantic_gate=0,
            gated_reward=0.0,
            semantic_confidence=0.63,
        ).model_copy(
            update={
                "semantic_evaluation": SemanticEvaluationSummary(
                    reasoning="gray_band_llm_nonmatch",
                    is_match=False,
                    confidence_score=0.63,
                    semantic_method="llm_gray_band",
                    semantic_method_version="gray-v1",
                ),
                "attribution": AttributionSummary(finality="online_provisional"),
            }
        )
        display = semantic_attribution_diagnostics_for_encounter(encounter)
        assert display.semantic_method == "backup stimulus confirmation checker · gray-v1"
        assert display.bounded_reason_code == "Backup checker did not confirm the stimulus"
        assert display.match_result == "non-match"
        assert display.attribution_finality == "online provisional"


# ----------------------------------------------------------------------
# Freshness (§4.C.4)
# ----------------------------------------------------------------------


class TestFreshness:
    def test_fresh_snapshot_labels_fresh(self) -> None:
        # below the stale threshold
        assert "fresh" in format_freshness(10.0, is_stale=False)

    def test_stale_snapshot_labels_stale(self) -> None:
        assert "stale" in format_freshness(120.0, is_stale=True)

    def test_stale_derived_from_threshold_when_flag_absent(self) -> None:
        # is_stale not provided — the formatter falls back to the
        # UI-side threshold so a never-updated row still reads as stale.
        assert "stale" in format_freshness(120.0)

    def test_absent_freshness_em_dash(self) -> None:
        assert format_freshness(None) == "—"


# ----------------------------------------------------------------------
# Calibration readiness
# ----------------------------------------------------------------------


class TestCalibrationStatus:
    def test_missing_live_state_defaults_to_ready(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
        )
        assert operator_ready_for_submit(summary) is True
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Healthy")

    def test_explicit_false_calibrating_state_is_ready(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=False,
        )
        assert operator_ready_for_submit(summary) is True
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Healthy")

    def test_progress_includes_frame_counts_below_threshold(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=True,
            calibration_frames_accumulated=12,
            calibration_frames_required=45,
        )
        assert operator_ready_for_submit(summary) is False
        assert format_calibration_status(summary) == (
            UiStatusKind.PROGRESS,
            "Preparing response baseline · 12/45 face frames",
        )

    def test_pre_stimulus_threshold_renders_ready_while_lifecycle_calibrating(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=True,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
        )
        assert operator_ready_for_submit(summary) is True
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Healthy")

    def test_missing_calibration_counts_render_ready(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=True,
        )
        assert operator_ready_for_submit(summary) is True
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Healthy")


# ----------------------------------------------------------------------
# Co-Modulation Index (§7C)
# ----------------------------------------------------------------------


class TestCoModulationRendering:
    def test_null_summary_renders_null_reason_text(self) -> None:
        summary = CoModulationSummary(
            session_id=_SESSION_ID,
            co_modulation_index=None,
            n_paired_observations=1,
            coverage_ratio=0.1,
            null_reason="insufficient aligned non-stale pairs",
        )
        # §7C: null is a legitimate outcome, not an error
        assert format_comodulation_index(summary) == "insufficient aligned non-stale pairs"

    def test_value_summary_renders_signed_three_decimals(self) -> None:
        summary = CoModulationSummary(
            session_id=_SESSION_ID,
            co_modulation_index=0.342,
            n_paired_observations=5,
            coverage_ratio=1.0,
        )
        assert format_comodulation_index(summary) == "+0.342"

    def test_negative_value_keeps_sign(self) -> None:
        summary = CoModulationSummary(
            session_id=_SESSION_ID,
            co_modulation_index=-0.121,
            n_paired_observations=5,
            coverage_ratio=1.0,
        )
        assert format_comodulation_index(summary) == "-0.121"

    def test_none_summary_em_dash(self) -> None:
        assert format_comodulation_index(None) == "—"

    def test_null_without_reason_falls_back_to_friendly_text(self) -> None:
        summary = CoModulationSummary(
            session_id=_SESSION_ID,
            co_modulation_index=None,
            n_paired_observations=0,
            coverage_ratio=0.0,
        )
        assert format_comodulation_index(summary) == "insufficient data"


# ----------------------------------------------------------------------
# §7D observational acoustic rendering
# ----------------------------------------------------------------------


def _acoustic_summary(**overrides: Any) -> ObservationalAcousticSummary:
    values: dict[str, Any] = {
        "f0_valid_measure": True,
        "f0_valid_baseline": True,
        "perturbation_valid_measure": True,
        "perturbation_valid_baseline": True,
        "voiced_coverage_measure_s": 2.2,
        "voiced_coverage_baseline_s": 2.1,
        "f0_mean_measure_hz": 220.0,
        "f0_mean_baseline_hz": 200.0,
        "f0_delta_semitones": 1.65,
        "jitter_mean_measure": 0.0112,
        "jitter_mean_baseline": 0.0091,
        "jitter_delta": 0.0021,
        "shimmer_mean_measure": 0.031,
        "shimmer_mean_baseline": 0.034,
        "shimmer_delta": -0.003,
    }
    values.update(overrides)
    return ObservationalAcousticSummary(**values)


class TestAcousticRendering:
    def test_validity_labels_distinguish_true_false_and_none(self) -> None:
        assert format_acoustic_validity(True) == "valid"
        assert format_acoustic_validity(False) == "invalid"
        assert format_acoustic_validity(None) == "—"

    def test_numeric_formatters_return_em_dash_for_absent_values(self) -> None:
        assert format_f0_hz(None) == "—"
        assert format_f0_hz(float("nan")) == "—"
        assert format_semitone_delta(None) == "—"
        assert format_semitone_delta(float("inf")) == "—"
        assert format_perturbation_delta(None) == "—"
        assert format_perturbation_delta(float("-inf")) == "—"

    def test_numeric_formatters_render_zero_values(self) -> None:
        assert format_f0_hz(0.0) == "0.0 Hz"
        assert format_semitone_delta(0.0) == "+0.00 st"
        assert format_perturbation_delta(0.0) == "+0.0000"

    def test_numeric_formatters_render_measured_values(self) -> None:
        assert format_f0_hz(203.456) == "203.5 Hz"
        assert format_semitone_delta(1.234) == "+1.23 st"
        assert format_semitone_delta(-0.456) == "-0.46 st"
        assert format_perturbation_delta(0.003456) == "+0.0035"
        assert format_perturbation_delta(-0.00234) == "-0.0023"

    def test_detail_display_formats_cards_validity_and_coverage(self) -> None:
        summary = _acoustic_summary()
        display = build_acoustic_detail_display(summary)
        assert display.has_summary is True
        assert display.f0_validity.status is UiStatusKind.OK
        assert display.f0_validity.text == "F0 valid"
        assert display.perturbation_validity.status is UiStatusKind.OK
        assert display.f0_mean.primary == "measure 220.0 Hz"
        assert display.f0_mean.secondary == "baseline 200.0 Hz · Δ +1.65 st"
        assert display.voiced_coverage_text == format_acoustic_voiced_coverage(2.2, 2.1)
        assert display.voiced_coverage_text == (
            "Voiced speech needed: at least 1.00s in each window · measure 2.20s · baseline 2.10s"
        )

    def test_validity_pill_display_distinguishes_invalid_and_absent(self) -> None:
        invalid = format_acoustic_validity_pill("F0", False, True)
        absent = format_acoustic_validity_pill("Perturbation", None, None)
        assert invalid.status is UiStatusKind.INFO
        assert invalid.text == "F0 not measured: measure F0 window invalid"
        assert absent.status is UiStatusKind.NEUTRAL
        assert "absent" in absent.text

    def test_voiced_coverage_and_ratio_helpers_are_formatter_owned(self) -> None:
        assert format_acoustic_seconds(1.234) == "1.23s"
        assert format_acoustic_seconds(None) == "—"
        assert format_acoustic_ratio(0.01234) == "0.0123"
        assert format_acoustic_ratio(0.0) == "0.0000"
        assert format_acoustic_voiced_coverage(None, 2.0) == (
            "Voiced speech needed: at least 1.00s in each window · measure — · baseline 2.00s"
        )

    def test_absent_acoustic_summary_message(self) -> None:
        assert build_acoustic_explanation(None) == "No acoustic data for this encounter."

    def test_explanation_renders_zero_acoustic_values_as_measured(self) -> None:
        summary = _acoustic_summary(
            f0_mean_measure_hz=0.0,
            f0_mean_baseline_hz=0.0,
            f0_delta_semitones=0.0,
            jitter_mean_measure=0.0,
            jitter_mean_baseline=0.0,
            jitter_delta=0.0,
            shimmer_mean_measure=0.0,
            shimmer_mean_baseline=0.0,
            shimmer_delta=0.0,
        )
        text = build_acoustic_explanation(summary)
        display = build_acoustic_detail_display(summary)
        assert "F0 means: measure 0.0 Hz, baseline 0.0 Hz" in text
        assert "F0 Δ +0.00 st" in text
        assert "Jitter Δ +0.0000" in text
        assert "Shimmer Δ +0.0000" in text
        assert "not measured" not in text
        assert display.f0_mean.primary == "measure 0.0 Hz"
        assert display.f0_mean.secondary == "baseline 0.0 Hz · Δ +0.00 st"

    def test_explanation_keeps_invalid_f0_separate_from_valid_perturbation(self) -> None:
        summary = _acoustic_summary(
            f0_valid_measure=False,
            f0_mean_measure_hz=None,
            f0_delta_semitones=None,
            jitter_delta=0.00234,
            shimmer_delta=-0.00123,
        )
        text = build_acoustic_explanation(summary)
        assert "F0 windows: measure invalid, baseline valid" in text
        assert "F0 Δ not measured (measure F0 window invalid)" in text
        assert "Perturbation windows: measure valid, baseline valid" in text
        assert "Jitter Δ +0.0023" in text
        assert "Shimmer Δ -0.0012" in text

    def test_explanation_keeps_invalid_perturbation_separate_from_f0_delta(self) -> None:
        summary = _acoustic_summary(
            f0_delta_semitones=2.5,
            perturbation_valid_baseline=False,
            jitter_mean_baseline=None,
            jitter_delta=None,
            shimmer_mean_baseline=None,
            shimmer_delta=None,
        )
        text = build_acoustic_explanation(summary)
        assert "F0 windows: measure valid, baseline valid" in text
        assert "F0 Δ +2.50 st" in text
        assert "Perturbation windows: measure valid, baseline invalid" in text
        assert "Jitter Δ not measured (baseline perturbation window invalid)" in text
        assert "Shimmer Δ not measured (baseline perturbation window invalid)" in text

    def test_explanation_distinguishes_missing_baseline_mean_from_invalid_window(self) -> None:
        summary = _acoustic_summary(
            f0_mean_baseline_hz=None,
            f0_delta_semitones=None,
        )
        text = build_acoustic_explanation(summary)
        assert "F0 means: measure 220.0 Hz, baseline —" in text
        assert "F0 Δ not measured (baseline F0 mean absent)" in text
        assert "F0 window invalid" not in text


# ----------------------------------------------------------------------
# §7B reward explanation
# ----------------------------------------------------------------------


def _encounter(
    *,
    semantic_gate: int | None = 1,
    semantic_confidence: float | None = 0.9,
    p90_intensity: float | None = 0.5,
    gated_reward: float | None = 0.5,
    n_frames_in_window: int | None = 60,
    au12_baseline_pre: float | None = 0.1,
    physiology_attached: bool = False,
    physiology_stale: bool | None = None,
    observational_acoustic: ObservationalAcousticSummary | None = None,
    semantic_evaluation: SemanticEvaluationSummary | None = None,
    attribution: AttributionSummary | None = None,
) -> EncounterSummary:
    return EncounterSummary(
        encounter_id="e1",
        session_id=_SESSION_ID,
        segment_timestamp_utc=_utc(2026, 4, 17, 12, 0),
        state=EncounterState.COMPLETED,
        active_arm="arm_a",
        expected_greeting="hi there",
        stimulus_time_utc=_utc(2026, 4, 17, 12, 0),
        semantic_gate=semantic_gate,
        semantic_confidence=semantic_confidence,
        p90_intensity=p90_intensity,
        gated_reward=gated_reward,
        n_frames_in_window=n_frames_in_window,
        au12_baseline_pre=au12_baseline_pre,
        physiology_attached=physiology_attached,
        physiology_stale=physiology_stale,
        observational_acoustic=observational_acoustic,
        semantic_evaluation=semantic_evaluation,
        attribution=attribution,
    )


class TestCauseEffectDisplay:
    def test_waiting_state_is_operator_friendly(self) -> None:
        display = build_cause_effect_display(None)
        assert display.headline == "Waiting for the first result"
        assert display.status is UiStatusKind.NEUTRAL

    def test_semantic_rejection_uses_required_plain_language(self) -> None:
        display = build_cause_effect_display(
            _encounter(
                semantic_gate=0,
                gated_reward=0.0,
                p90_intensity=0.42,
                semantic_evaluation=SemanticEvaluationSummary(
                    is_match=False,
                    confidence_score=0.72,
                ),
            )
        )
        assert display.headline == "Stimulus missed or ignored by host."
        assert display.status is UiStatusKind.INFO
        assert "42%" in display.response_summary
        assert (
            semantic_confidence_for_encounter(
                _encounter(
                    semantic_confidence=0.1,
                    semantic_evaluation=SemanticEvaluationSummary(confidence_score=0.72),
                )
            )
            == 0.72
        )

    def test_success_prefers_peak_latency_when_attribution_available(self) -> None:
        display = build_cause_effect_display(
            _encounter(
                attribution=AttributionSummary(
                    au12_lift_p90=0.31,
                    au12_peak_latency_ms=1200.0,
                ),
                observational_acoustic=ObservationalAcousticSummary(
                    f0_delta_semitones=2.0,
                    voiced_coverage_measure_s=1.25,
                ),
            )
        )
        assert display.status is UiStatusKind.OK
        assert "1.20s" in display.headline
        assert "response lift 0.310" in display.response_summary
        assert "pitch +2.00 st" in display.voice_summary

    def test_success_falls_back_to_p90_and_baseline_without_attribution(self) -> None:
        display = build_cause_effect_display(_encounter(p90_intensity=0.7, au12_baseline_pre=0.2))
        assert "70%" in display.headline
        assert "response lift 0.500" in display.response_summary

    def test_no_frames_reads_as_not_measured(self) -> None:
        display = build_cause_effect_display(_encounter(n_frames_in_window=0, gated_reward=None))
        assert display.status is UiStatusKind.WARN
        assert display.headline == "Response not measured"


class TestWorkflowDisplayContracts:
    def test_readiness_display_uses_capture_detail(self) -> None:
        display = build_readiness_display(
            ready_for_submit=True,
            calibration_status=(UiStatusKind.OK, "Healthy"),
            capture_detail="Phone healthy · Audio capture healthy",
            progress_message="Send one stimulus.",
        )
        assert display.status is UiStatusKind.OK
        assert display.title == "Ready for a stimulus"
        assert "Phone healthy" in display.detail

    def test_live_telemetry_display_shows_measuring_state(self) -> None:
        display = build_live_telemetry_display(
            stimulus_state=StimulusActionState.MEASURING,
            progress_message="Measuring the response window now.",
            response_signal_percent=64,
            live_status=(UiStatusKind.OK, "Live analysis healthy"),
        )
        assert display.status is UiStatusKind.PROGRESS
        assert display.headline == "Measuring response…"
        assert display.response_signal == "64%"


class TestStrategyEvidenceDisplay:
    def test_strategy_rows_rank_active_enabled_arm_first(self) -> None:
        detail = ExperimentDetail(
            experiment_id="exp",
            active_arm_id="a2",
            arms=[
                ArmSummary(
                    arm_id="a1",
                    greeting_text="Hei a1",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                    selection_count=0,
                ),
                ArmSummary(
                    arm_id="a2",
                    greeting_text="Hei a2",
                    posterior_alpha=5.0,
                    posterior_beta=2.0,
                    evaluation_variance=0.01,
                    selection_count=8,
                    recent_reward_mean=0.62,
                    recent_semantic_pass_rate=0.8,
                ),
            ],
        )
        rows = build_strategy_evidence_display(detail)
        assert rows[0].arm_id == "a2"
        assert rows[0].status is UiStatusKind.OK
        assert rows[0].label == "Active · Lower uncertainty so far"
        assert "recent observed reward 0.620" in rows[0].outcome
        assert "stimulus confirmed 80%" in rows[0].outcome

    def test_strategy_rows_mark_disabled_and_sparse_data(self) -> None:
        detail = ExperimentDetail(
            experiment_id="exp",
            arms=[
                ArmSummary(
                    arm_id="disabled",
                    greeting_text="Hei disabled",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                    selection_count=2,
                    enabled=False,
                ),
                ArmSummary(
                    arm_id="new",
                    greeting_text="Hei new",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                    selection_count=0,
                ),
            ],
        )
        labels = {row.arm_id: row.label for row in build_strategy_evidence_display(detail)}
        assert labels["disabled"] == "Disabled"
        assert labels["new"] == "Needs first try"


class TestRewardExplanation:
    def test_gate_closed_message_calls_out_gate_not_frames(self) -> None:
        # §7B: semantic_gate=0 zeros the reward even when frames exist
        text = build_reward_explanation(
            _encounter(semantic_gate=0, gated_reward=0.0, p90_intensity=0.42)
        )
        assert "stimulus was not confirmed" in text.lower()
        assert "held back" in text.lower()
        assert "0.420" in text

    def test_gate_closed_uses_nested_semantic_confidence_when_present(self) -> None:
        text = build_reward_explanation(
            _encounter(
                semantic_gate=0,
                semantic_confidence=0.11,
                semantic_evaluation=SemanticEvaluationSummary(confidence_score=0.72),
                gated_reward=0.0,
            )
        )
        assert "72% confidence" in text
        assert "11% confidence" not in text

    def test_no_frames_message_calls_out_frames_not_gate(self) -> None:
        # §7B: n_frames_in_window=0 is a distinct outcome from gate=0
        text = build_reward_explanation(
            _encounter(semantic_gate=1, n_frames_in_window=0, gated_reward=None)
        )
        lower = text.lower()
        assert "frames" in lower
        assert "not computed" in lower
        # Must not say "gate closed" — that's a different outcome
        assert "gate closed" not in lower

    def test_happy_path_includes_product_and_baseline(self) -> None:
        text = build_reward_explanation(_encounter(gated_reward=0.42))
        # §7B formula reads as a product
        assert "0.500" in text  # p90_intensity
        assert "0.420" in text  # gated_reward
        assert "before-stimulus" in text.lower()

    def test_physiology_attached_stale_surfaces_stale(self) -> None:
        text = build_reward_explanation(_encounter(physiology_attached=True, physiology_stale=True))
        assert "stale" in text.lower()

    def test_physiology_attached_fresh_surfaces_fresh(self) -> None:
        text = build_reward_explanation(
            _encounter(physiology_attached=True, physiology_stale=False)
        )
        assert "fresh" in text.lower()


# ----------------------------------------------------------------------
# Physiology explanation + health rendering
# ----------------------------------------------------------------------


class TestCoModulationDisplay:
    def test_absent_co_modulation_waits_for_window(self) -> None:
        display = build_co_modulation_display(None)
        assert display.primary == "—"
        assert display.secondary == "No sync window yet"
        assert display.status is UiStatusKind.NEUTRAL

    def test_null_valid_co_modulation_accumulates_data(self) -> None:
        display = build_co_modulation_display(
            CoModulationSummary(
                session_id=_SESSION_ID,
                co_modulation_index=None,
                n_paired_observations=1,
                coverage_ratio=0.1,
                null_reason="insufficient aligned non-stale pairs",
            )
        )
        assert display.primary == "—"
        assert display.secondary == "Sync data accumulating"
        assert display.status is UiStatusKind.INFO
        assert "insufficient aligned non-stale pairs" in display.detail

    def test_numeric_co_modulation_formats_bounded_score(self) -> None:
        display = build_co_modulation_display(
            CoModulationSummary(
                session_id=_SESSION_ID,
                co_modulation_index=0.82,
                n_paired_observations=6,
                coverage_ratio=0.75,
            )
        )
        assert display.primary == "+0.82"
        assert "moving together" in display.secondary
        assert "75%" in display.detail


class TestPhysiologyExplanation:
    def test_absent_snapshot_message(self) -> None:
        assert build_physiology_explanation(None) == "No physiology data for this session."

    def test_stale_streamer_surfaces_stale_wording(self) -> None:
        snapshot = SessionPhysiologySnapshot(
            session_id=_SESSION_ID,
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=34.0,
                heart_rate_bpm=82,
                freshness_s=180.0,
                is_stale=True,
            ),
            operator=None,
            comodulation=None,
            generated_at_utc=_utc(2026, 4, 17, 12, 0),
        )
        text = build_physiology_explanation(snapshot)
        assert "stale" in text
        assert "operator: absent" in text

    def test_co_modulation_null_reason_appears_in_line(self) -> None:
        snapshot = SessionPhysiologySnapshot(
            session_id=_SESSION_ID,
            streamer=None,
            operator=None,
            comodulation=CoModulationSummary(
                session_id=_SESSION_ID,
                co_modulation_index=None,
                n_paired_observations=0,
                coverage_ratio=0.0,
                null_reason="insufficient aligned pairs",
            ),
            generated_at_utc=_utc(2026, 4, 17, 12, 0),
        )
        text = build_physiology_explanation(snapshot)
        assert "insufficient aligned pairs" in text


class TestHealthRendering:
    def test_recovering_maps_to_progress_status(self) -> None:
        row = HealthSubsystemStatus(
            subsystem_key="ffmpeg",
            label="FFmpeg",
            state=HealthState.RECOVERING,
            recovery_mode="restart in progress",
        )
        # §12: recovering is distinct from warn/error
        assert ui_status_for_health(row) is UiStatusKind.PROGRESS

    def test_error_maps_to_error_status(self) -> None:
        row = HealthSubsystemStatus(
            subsystem_key="db",
            label="PostgreSQL",
            state=HealthState.ERROR,
        )
        assert ui_status_for_health(row) is UiStatusKind.ERROR

    def test_state_label_words(self) -> None:
        assert format_health_state(HealthState.OK) == "ok"
        assert format_health_state(HealthState.DEGRADED) == "degraded"
        assert format_health_state(HealthState.RECOVERING) == "recovering"
        assert format_health_state(HealthState.ERROR) == "error"

    def test_detail_line_includes_recovery_and_hint(self) -> None:
        row = HealthSubsystemStatus(
            subsystem_key="azure",
            label="Azure TTS",
            state=HealthState.DEGRADED,
            detail="retry-then-null on 429",
            recovery_mode="backoff 5s",
            operator_action_hint="monitor quota",
            last_success_utc=_utc(2026, 4, 17, 11, 30),
        )
        text = build_health_detail(row)
        assert "degraded" in text
        assert "recovery: backoff 5s" in text
        assert "monitor quota" in text
        assert "2026-04-17" in text


# ----------------------------------------------------------------------
# Stimulus-confirmation text truncation
# ----------------------------------------------------------------------


class TestStimulusConfirmationTruncation:
    def test_short_expected_greeting_passes_through(self) -> None:
        assert truncate_expected_greeting("hi there") == "hi there"

    def test_long_expected_greeting_gets_ellipsis(self) -> None:
        text = "x" * 100
        out = truncate_expected_greeting(text, limit=20)
        assert len(out) == 20
        assert out.endswith("…")

    def test_none_returns_em_dash(self) -> None:
        assert truncate_expected_greeting(None) == "—"
