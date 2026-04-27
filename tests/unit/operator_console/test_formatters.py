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
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    HealthState,
    HealthSubsystemStatus,
    ObservationalAcousticSummary,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    UiStatusKind,
)
from services.operator_console.formatters import (
    build_acoustic_detail_display,
    build_acoustic_explanation,
    build_health_detail,
    build_physiology_explanation,
    build_reward_explanation,
    format_acoustic_ratio,
    format_acoustic_seconds,
    format_acoustic_validity,
    format_acoustic_validity_pill,
    format_acoustic_voiced_coverage,
    format_calibration_status,
    format_comodulation_index,
    format_duration,
    format_f0_hz,
    format_freshness,
    format_health_state,
    format_percentage,
    format_perturbation_delta,
    format_reward,
    format_semantic_confidence,
    format_semantic_gate,
    format_semitone_delta,
    format_timestamp,
    operator_ready_for_submit,
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


# ----------------------------------------------------------------------
# §7B semantic gate
# ----------------------------------------------------------------------


class TestSemanticGate:
    def test_gate_open_text(self) -> None:
        assert format_semantic_gate(1) == "open (reward admitted)"

    def test_gate_closed_text_calls_out_suppression(self) -> None:
        # Operator must not misread "0" as missing data; the word
        # "suppressed" is load-bearing here.
        assert "suppressed" in format_semantic_gate(0)

    def test_gate_none_renders_em_dash(self) -> None:
        assert format_semantic_gate(None) == "—"

    def test_semantic_confidence_is_percentage(self) -> None:
        assert format_semantic_confidence(0.87) == "87%"
        assert format_semantic_confidence(None) == "—"


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
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Ready")

    def test_explicit_false_calibrating_state_is_ready(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=False,
        )
        assert operator_ready_for_submit(summary) is True
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Ready")

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
            "Calibrating · 12/45 frames",
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
        assert format_calibration_status(summary) == (UiStatusKind.OK, "Ready")

    def test_progress_without_counts_falls_back_to_generic_label(self) -> None:
        summary = SessionSummary(
            session_id=_SESSION_ID,
            status="active",
            started_at_utc=_utc(2026, 4, 17, 12, 0),
            is_calibrating=True,
        )
        assert operator_ready_for_submit(summary) is False
        assert format_calibration_status(summary) == (UiStatusKind.PROGRESS, "Calibrating")


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
        assert display.jitter_mean.primary == "measure 0.0112"
        assert display.jitter_mean.secondary == "baseline 0.0091 · Δ +0.0021"
        assert display.voiced_coverage_text == format_acoustic_voiced_coverage(2.2, 2.1)
        assert display.voiced_coverage_text == "Voiced coverage: measure 2.20s · baseline 2.10s"

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
            "Voiced coverage: measure — · baseline 2.00s"
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
        assert display.jitter_mean.primary == "measure 0.0000"
        assert display.jitter_mean.secondary == "baseline 0.0000 · Δ +0.0000"
        assert display.shimmer_mean.primary == "measure 0.0000"
        assert display.shimmer_mean.secondary == "baseline 0.0000 · Δ +0.0000"

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
    baseline_b_neutral: float | None = 0.1,
    physiology_attached: bool = False,
    physiology_stale: bool | None = None,
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
        baseline_b_neutral=baseline_b_neutral,
        physiology_attached=physiology_attached,
        physiology_stale=physiology_stale,
    )


class TestRewardExplanation:
    def test_gate_closed_message_calls_out_gate_not_frames(self) -> None:
        # §7B: semantic_gate=0 zeros the reward even when frames exist
        text = build_reward_explanation(
            _encounter(semantic_gate=0, gated_reward=0.0, p90_intensity=0.42)
        )
        assert "gate closed" in text.lower()
        assert "suppressed" in text.lower()
        # The intensity still shown — operators need to see that the
        # pipeline measured something, it just wasn't admitted.
        assert "0.420" in text

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
        assert "baseline" in text.lower()

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
# Greeting truncation
# ----------------------------------------------------------------------


class TestGreetingTruncation:
    def test_short_greeting_passes_through(self) -> None:
        assert truncate_expected_greeting("hi there") == "hi there"

    def test_long_greeting_gets_ellipsis(self) -> None:
        text = "x" * 100
        out = truncate_expected_greeting(text, limit=20)
        assert len(out) == 20
        assert out.endswith("…")

    def test_none_returns_em_dash(self) -> None:
        assert truncate_expected_greeting(None) == "—"
