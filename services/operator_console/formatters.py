"""
Operator-language formatters.

Pure string helpers that translate DTO fields into the short, precise
phrases the operator UI shows. Nothing here creates widgets, talks to
the DB, or makes a network call; every function is a pure function of
its arguments so views/viewmodels stay simple and tests stay trivial.

Separating this from the widgets serves three purposes:

  1. One home for the language. When the spec wording shifts (e.g., §7B
     renames "semantic gate" to something else), we grep one file.
  2. View code never builds f-strings inline, so widgets remain layout-
     only and the viewmodel layer remains data-only.
  3. Reward-explanation text is the heart of operator trust; centralising
     it means the same phrasing is used on the Overview card and on the
     Live Session detail pane.

Spec references:
  §4.C.4     — physiology freshness / staleness vocabulary
  §7B        — reward = p90_intensity × semantic_gate; the explainer
               names the exact fields the pipeline used
  §7C        — co-modulation null-valid; we render the `null_reason`
               verbatim so "not enough aligned pairs yet" reads as a
               legitimate outcome, not a bug
  §7D        — observational acoustic validity, means, and deltas render
               as measured values or explicit not-measured outcomes
  §7E / §8   — semantic/attribution diagnostics are observational only;
               reason codes are bounded operator explanations
  §12        — degraded-but-recovering vocabulary for health rows
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from packages.schemas.operator_console import (
    AttributionSummary,
    CoModulationSummary,
    EncounterSummary,
    HealthProbeState,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    ObservationalAcousticSummary,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusActionState,
    UiStatusKind,
)

_EM_DASH = "—"
_ACOUSTIC_SECTION_TITLE = "Voice signal details"
_ACOUSTIC_EMPTY_TEXT = "No voice signal details for this segment"
_SEMANTIC_ATTRIBUTION_SECTION_TITLE = "Greeting match and follow-up signals"
_SEMANTIC_ATTRIBUTION_EMPTY_TEXT = "No greeting-match details for this encounter"
_SEMANTIC_ATTRIBUTION_ATTRIBUTION_EMPTY_TEXT = "No follow-up signal details for this encounter"
_SEMANTIC_ATTRIBUTION_OBSERVATIONAL_NOTE = (
    "Shown for transparency only; these details do not change the reward."
)
_DEGREE_OF_FRESHNESS_STALE_S = 60.0  # §4.C.4 default — UI-side fallback


@dataclass(frozen=True)
class AcousticSectionLabels:
    """Static §7D labels supplied to the Live Session acoustic layout."""

    section_title: str
    empty_text: str
    f0_metric_title: str
    jitter_metric_title: str
    shimmer_metric_title: str


@dataclass(frozen=True)
class AcousticValidityPillDisplay:
    """Text and visual status for an acoustic validity pill."""

    status: UiStatusKind
    text: str


@dataclass(frozen=True)
class AcousticMetricCardDisplay:
    """Preformatted text/status payload for an acoustic MetricCard."""

    primary: str
    secondary: str
    status: UiStatusKind = UiStatusKind.NEUTRAL
    status_text: str | None = None


@dataclass(frozen=True)
class AcousticDetailDisplay:
    """Complete formatter/viewmodel contract for the §7D detail section."""

    section_title: str
    empty_text: str
    has_summary: bool
    f0_validity: AcousticValidityPillDisplay
    perturbation_validity: AcousticValidityPillDisplay
    f0_mean: AcousticMetricCardDisplay
    jitter_mean: AcousticMetricCardDisplay
    shimmer_mean: AcousticMetricCardDisplay
    voiced_coverage_text: str
    explanation: str


@dataclass(frozen=True)
class SemanticAttributionDiagnosticsDisplay:
    """Read-only §7E diagnostics contract for compact operator panes.

    The display keeps semantic/attribution analytics adjacent to an encounter
    without changing the §7B reward explanation structure. Every value is
    preformatted so views do not render raw bounded codes or JSON-shaped DTOs.
    """

    section_title: str
    empty_text: str
    attribution_empty_text: str
    has_diagnostics: bool
    has_semantic: bool
    has_attribution: bool
    semantic_method: str
    bounded_reason_code: str
    probability_confidence: str
    match_result: str
    attribution_finality: str
    soft_reward_candidate: str
    au12_lift_metrics: str
    au12_peak_latency: str
    synchrony_metrics: str
    outcome_link_lag: str
    compact_summary: str
    observational_note: str


@dataclass(frozen=True)
class RewardDetailLabels:
    """Static labels for the §7B reward readback cards."""

    p90_title: str
    gate_title: str
    reward_title: str
    frames_title: str
    baseline_title: str
    physiology_title: str
    reward_formula: str


@dataclass(frozen=True)
class PhysiologyLabels:
    """Static labels for physiology and co-modulation cards."""

    rmssd_title: str
    rmssd_explanation: str
    no_rmssd_summary: str
    no_rmssd_detail: str
    comodulation_title: str
    comodulation_subtitle: str
    comodulation_null_status: str
    observations_detail: str
    coverage_detail: str


# ----------------------------------------------------------------------
# Primitive formatters
# ----------------------------------------------------------------------


def format_timestamp(value: datetime | None) -> str:
    """UTC ISO-like timestamp with `Z` suffix. `—` when absent."""
    if value is None:
        return _EM_DASH
    if value.tzinfo is None or value.utcoffset() is None:
        # Defensive: our DTOs reject naive timestamps at the boundary,
        # but a defensively-coded viewmodel may still hand one in.
        return value.isoformat()
    as_utc = value.astimezone(UTC)
    return as_utc.strftime("%Y-%m-%d %H:%M:%SZ")


def format_duration(seconds: float | None) -> str:
    """`h m s` for a non-negative duration; `—` when absent."""
    if seconds is None or seconds < 0:
        return _EM_DASH
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_percentage(ratio: float | None, *, digits: int = 1) -> str:
    """`12.3%` from a 0..1 ratio; `—` when absent."""
    if ratio is None:
        return _EM_DASH
    return f"{ratio * 100:.{digits}f}%"


def format_reward(value: float | None, *, digits: int = 3) -> str:
    """Reward values are unitless; keep three decimals for readability."""
    if value is None:
        return _EM_DASH
    return f"{value:.{digits}f}"


def _acoustic_number_or_none(value: float | None) -> float | None:
    """Return measured §7D acoustic numerics; null/non-finite stay absent."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _has_acoustic_number(value: float | None) -> bool:
    return _acoustic_number_or_none(value) is not None


def format_acoustic_validity(flag: bool | None) -> str:
    """Operator label for §7D acoustic window validity flags."""
    if flag is None:
        return _EM_DASH
    return "valid" if flag else "invalid"


def format_f0_hz(value: float | None) -> str:
    """Render measured F0 means; absent placeholders stay explicit."""
    measured = _acoustic_number_or_none(value)
    if measured is None:
        return _EM_DASH
    return f"{measured:.1f} Hz"


def format_semitone_delta(value: float | None) -> str:
    """Render §7D F0 deltas with sign and semitone unit."""
    measured = _acoustic_number_or_none(value)
    if measured is None:
        return _EM_DASH
    return f"{measured:+.2f} st"


def format_perturbation_delta(value: float | None) -> str:
    """Render additive jitter/shimmer deltas as signed raw ratios."""
    measured = _acoustic_number_or_none(value)
    if measured is None:
        return _EM_DASH
    return f"{measured:+.4f}"


def acoustic_section_labels() -> AcousticSectionLabels:
    """Operator-facing static labels for the Live Session §7D section."""

    return AcousticSectionLabels(
        section_title=_ACOUSTIC_SECTION_TITLE,
        empty_text=_ACOUSTIC_EMPTY_TEXT,
        f0_metric_title="Pitch level",
        jitter_metric_title="Pitch steadiness",
        shimmer_metric_title="Volume steadiness",
    )


def reward_detail_labels() -> RewardDetailLabels:
    """Operator-facing static labels for the reward detail grid."""

    return RewardDetailLabels(
        p90_title="Strongest smile signal",
        gate_title="Greeting matched?",
        reward_title="Reward used",
        frames_title="Face frames in reward window",
        baseline_title="Before-greeting smile level",
        physiology_title="Physiology data",
        reward_formula="Smile signal × greeting match",
    )


def physiology_labels() -> PhysiologyLabels:
    """Operator-facing static labels for physiology cards."""

    return PhysiologyLabels(
        rmssd_title="Heart-rate variability",
        rmssd_explanation="beat-to-beat variation; higher often means more recovery",
        no_rmssd_summary="No heart-rate variability yet",
        no_rmssd_detail="heart-rate variability not in this snapshot",
        comodulation_title="Shared stress/recovery movement",
        comodulation_subtitle=(
            "Shows whether streamer and operator heart-rate variability moved together."
        ),
        comodulation_null_status="not enough data yet",
        observations_detail="matched fresh data points",
        coverage_detail="share of the time window with usable data",
    )


def format_acoustic_seconds(value: float | None) -> str:
    """Render voiced-coverage seconds; null coverage remains explicit."""

    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"{value:.2f}s"


def format_acoustic_ratio(value: float | None) -> str:
    """Render jitter/shimmer ratios with acoustic null placeholders respected."""

    measured = _acoustic_number_or_none(value)
    if measured is None:
        return _EM_DASH
    return f"{measured:.4f}"


def format_acoustic_voiced_coverage(
    measure_s: float | None,
    baseline_s: float | None,
) -> str:
    """Render the paired §7D voiced-coverage windows for operator detail panes."""

    return (
        "Voiced speech needed: at least 1.00s in each window · "
        f"measure {format_acoustic_seconds(measure_s)} · "
        f"baseline {format_acoustic_seconds(baseline_s)}"
    )


def format_acoustic_validity_pill(
    family_label: str,
    measure_valid: bool | None,
    baseline_valid: bool | None,
) -> AcousticValidityPillDisplay:
    """Render validity/nullness copy and visual status for one acoustic family."""

    if measure_valid is True and baseline_valid is True:
        return AcousticValidityPillDisplay(UiStatusKind.OK, f"{family_label} valid")

    reason_label = "F0" if family_label == "F0" else family_label.lower()
    invalid_windows: list[str] = []
    if measure_valid is False:
        invalid_windows.append("measure")
    if baseline_valid is False:
        invalid_windows.append("baseline")
    if invalid_windows:
        window_text = " and ".join(invalid_windows)
        noun = "windows" if len(invalid_windows) > 1 else "window"
        return AcousticValidityPillDisplay(
            UiStatusKind.INFO,
            f"{family_label} not measured: {window_text} {reason_label} {noun} invalid",
        )

    absent_windows: list[str] = []
    if measure_valid is None:
        absent_windows.append("measure")
    if baseline_valid is None:
        absent_windows.append("baseline")
    if absent_windows:
        window_text = " and ".join(absent_windows)
        noun = "statuses" if len(absent_windows) > 1 else "status"
        return AcousticValidityPillDisplay(
            UiStatusKind.NEUTRAL,
            f"{family_label} absent: {window_text} {reason_label} window {noun} missing",
        )

    return AcousticValidityPillDisplay(UiStatusKind.NEUTRAL, f"{family_label} absent")


def build_acoustic_detail_display(
    summary: ObservationalAcousticSummary | None,
) -> AcousticDetailDisplay:
    """Build the complete §7D detail-pane display payload from one summary."""

    labels = acoustic_section_labels()
    if summary is None:
        return AcousticDetailDisplay(
            section_title=labels.section_title,
            empty_text=labels.empty_text,
            has_summary=False,
            f0_validity=AcousticValidityPillDisplay(UiStatusKind.NEUTRAL, "F0 absent"),
            perturbation_validity=AcousticValidityPillDisplay(
                UiStatusKind.NEUTRAL,
                "Perturbation absent",
            ),
            f0_mean=AcousticMetricCardDisplay(_EM_DASH, ""),
            jitter_mean=AcousticMetricCardDisplay(_EM_DASH, ""),
            shimmer_mean=AcousticMetricCardDisplay(_EM_DASH, ""),
            voiced_coverage_text="",
            explanation=build_acoustic_explanation(None),
        )

    return AcousticDetailDisplay(
        section_title=labels.section_title,
        empty_text=labels.empty_text,
        has_summary=True,
        f0_validity=format_acoustic_validity_pill(
            "F0",
            summary.f0_valid_measure,
            summary.f0_valid_baseline,
        ),
        perturbation_validity=format_acoustic_validity_pill(
            "Perturbation",
            summary.perturbation_valid_measure,
            summary.perturbation_valid_baseline,
        ),
        f0_mean=AcousticMetricCardDisplay(
            primary=f"measure {format_f0_hz(summary.f0_mean_measure_hz)}",
            secondary=(
                f"baseline {format_f0_hz(summary.f0_mean_baseline_hz)} · "
                f"Δ {format_semitone_delta(summary.f0_delta_semitones)}"
            ),
        ),
        jitter_mean=AcousticMetricCardDisplay(
            primary=f"measure {format_acoustic_ratio(summary.jitter_mean_measure)}",
            secondary=(
                f"baseline {format_acoustic_ratio(summary.jitter_mean_baseline)} · "
                f"Δ {format_perturbation_delta(summary.jitter_delta)}"
            ),
        ),
        shimmer_mean=AcousticMetricCardDisplay(
            primary=f"measure {format_acoustic_ratio(summary.shimmer_mean_measure)}",
            secondary=(
                f"baseline {format_acoustic_ratio(summary.shimmer_mean_baseline)} · "
                f"Δ {format_perturbation_delta(summary.shimmer_delta)}"
            ),
        ),
        voiced_coverage_text=format_acoustic_voiced_coverage(
            summary.voiced_coverage_measure_s,
            summary.voiced_coverage_baseline_s,
        ),
        explanation=build_acoustic_explanation(summary),
    )


def format_semantic_gate(gate: int | None) -> str:
    """Map the 0/1 gate to operator language.

    §7B: `semantic_gate=0` zeros the reward. Communicating this in plain
    words prevents an operator from misreading a numeric zero as missing
    data.
    """
    if gate is None:
        return _EM_DASH
    if gate == 1:
        return "yes — reward can count"
    return "no — reward held back"


def format_semantic_confidence(confidence: float | None) -> str:
    """Confidence ∈ [0,1], rendered as a percentage. `—` when absent."""
    return format_percentage(confidence, digits=0)


def format_semantic_method_label(
    semantic_method: str | None,
    semantic_method_version: str | None = None,
) -> str:
    """Operator label for the deterministic §8 semantic method.

    v3.4 routes through the local cross-encoder first and may use an
    LLM only as a gray-band fallback. The label names that path without
    implying the diagnostics affect the §7B reward update.
    """

    if semantic_method is None:
        return _EM_DASH
    mapping = {
        "cross_encoder": "local greeting checker",
        "llm_gray_band": "backup greeting checker",
    }
    label = mapping.get(semantic_method, _clean_code_label(semantic_method))
    if semantic_method_version:
        return f"{label} · {semantic_method_version}"
    return label


def format_bounded_reason_code_label(reason_code: str | None) -> str:
    """Operator label for §8.3 bounded semantic reason codes.

    The persisted `reasoning` field is a bounded code, not narrative
    prose; this formatter prevents raw JSON/code strings from leaking
    into operator panes while preserving the diagnostic meaning.
    """

    if reason_code is None:
        return _EM_DASH
    mapping = {
        "cross_encoder_high_match": "Greeting clearly matched",
        "cross_encoder_high_nonmatch": "Greeting clearly did not match",
        "gray_band_llm_match": "Backup checker found a match",
        "gray_band_llm_nonmatch": "Backup checker did not find a match",
        "semantic_local_failure_fallback": "Greeting checker used its safe fallback",
        "semantic_timeout": "Greeting check timed out",
        "semantic_error": "Greeting check error",
    }
    return mapping.get(reason_code, _clean_code_label(reason_code))


def format_probability_confidence(confidence_score: float | None) -> str:
    """Render the §7E semantic probability estimate `p_match`."""

    if confidence_score is None:
        return _EM_DASH
    return f"match confidence {format_percentage(confidence_score, digits=0)}"


def format_semantic_match_label(is_match: bool | None) -> str:
    """Render the final boolean semantic gate result for diagnostics."""

    if is_match is None:
        return _EM_DASH
    return "match" if is_match else "non-match"


def format_attribution_finality_label(finality: str | None) -> str:
    """Render the §7E attribution lifecycle state."""

    if finality is None:
        return _EM_DASH
    mapping = {
        "online_provisional": "online provisional",
        "offline_final": "offline final",
    }
    return mapping.get(finality, _clean_code_label(finality))


def format_soft_reward_candidate(value: float | None) -> str:
    """Render observational §7E `soft_reward_candidate` / `r_t^soft`."""

    if value is None:
        return _EM_DASH
    return f"possible follow-up reward {format_reward(value)}"


def format_au12_lift_metrics(attribution: AttributionSummary | None) -> str:
    """Render baseline-aware AU12 lift metrics from §7E attribution."""

    if attribution is None:
        return _EM_DASH
    parts: list[str] = []
    if attribution.au12_baseline_pre is not None:
        parts.append(f"before greeting {format_reward(attribution.au12_baseline_pre)}")
    if attribution.au12_lift_p90 is not None:
        parts.append(f"strong smile lift {format_reward(attribution.au12_lift_p90)}")
    if attribution.au12_lift_peak is not None:
        parts.append(f"peak smile lift {format_reward(attribution.au12_lift_peak)}")
    return " · ".join(parts) if parts else _EM_DASH


def format_au12_peak_latency(attribution: AttributionSummary | None) -> str:
    """Render the §7E latency from stimulus to peak AU12 lift."""

    if attribution is None:
        return _EM_DASH
    return format_latency_ms(attribution.au12_peak_latency_ms)


def format_synchrony_metrics(attribution: AttributionSummary | None) -> str:
    """Render lag-aware synchrony diagnostics from §7E attribution."""

    if attribution is None:
        return _EM_DASH
    parts: list[str] = []
    if attribution.sync_peak_corr is not None:
        parts.append(f"movement together {attribution.sync_peak_corr:+.3f}")
    if attribution.sync_peak_lag is not None:
        parts.append(f"lag {attribution.sync_peak_lag}")
    return " · ".join(parts) if parts else _EM_DASH


def format_outcome_link_lag(lag_s: float | None) -> str:
    """Render event→outcome lag when a §7E link exists."""

    if lag_s is None or not math.isfinite(lag_s):
        return _EM_DASH
    return f"after {lag_s:.1f}s"


def format_semantic_attribution_compact_summary(
    semantic: SemanticEvaluationSummary | None,
    attribution: AttributionSummary | None,
) -> str:
    """Compact §8/§7E readback for dense cards such as Overview."""

    parts: list[str] = []
    if semantic is not None:
        parts.append(f"greeting {format_semantic_match_label(semantic.is_match)}")
        if semantic.confidence_score is not None:
            parts.append(format_probability_confidence(semantic.confidence_score))
    else:
        parts.append("greeting check absent")

    if attribution is not None:
        parts.append(f"follow-up signals {format_attribution_finality_label(attribution.finality)}")
    else:
        parts.append("follow-up signals absent")
    return " · ".join(parts)


def build_semantic_attribution_diagnostics_display(
    semantic: SemanticEvaluationSummary | None,
    attribution: AttributionSummary | None,
) -> SemanticAttributionDiagnosticsDisplay:
    """Build preformatted read-only §7E diagnostics from encounter aggregates."""

    return SemanticAttributionDiagnosticsDisplay(
        section_title=_SEMANTIC_ATTRIBUTION_SECTION_TITLE,
        empty_text=_SEMANTIC_ATTRIBUTION_EMPTY_TEXT,
        attribution_empty_text=_SEMANTIC_ATTRIBUTION_ATTRIBUTION_EMPTY_TEXT,
        has_diagnostics=semantic is not None or attribution is not None,
        has_semantic=semantic is not None,
        has_attribution=attribution is not None,
        semantic_method=format_semantic_method_label(
            semantic.semantic_method if semantic is not None else None,
            semantic.semantic_method_version if semantic is not None else None,
        ),
        bounded_reason_code=format_bounded_reason_code_label(
            semantic.reasoning if semantic is not None else None,
        ),
        probability_confidence=format_probability_confidence(
            semantic.confidence_score if semantic is not None else None,
        ),
        match_result=format_semantic_match_label(
            semantic.is_match if semantic is not None else None,
        ),
        attribution_finality=format_attribution_finality_label(
            attribution.finality if attribution is not None else None,
        ),
        soft_reward_candidate=format_soft_reward_candidate(
            attribution.soft_reward_candidate if attribution is not None else None,
        ),
        au12_lift_metrics=format_au12_lift_metrics(attribution),
        au12_peak_latency=format_au12_peak_latency(attribution),
        synchrony_metrics=format_synchrony_metrics(attribution),
        outcome_link_lag=format_outcome_link_lag(
            attribution.outcome_link_lag_s if attribution is not None else None,
        ),
        compact_summary=format_semantic_attribution_compact_summary(semantic, attribution),
        observational_note=_SEMANTIC_ATTRIBUTION_OBSERVATIONAL_NOTE,
    )


def semantic_attribution_diagnostics_for_encounter(
    encounter: EncounterSummary | LatestEncounterSummary | None,
) -> SemanticAttributionDiagnosticsDisplay:
    """Preformatted §7E diagnostics for an encounter aggregate, if present."""

    if encounter is None:
        return build_semantic_attribution_diagnostics_display(None, None)
    return build_semantic_attribution_diagnostics_display(
        encounter.semantic_evaluation,
        encounter.attribution,
    )


def _clean_code_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().capitalize()


def format_health_state(state: HealthState) -> str:
    """Operator-facing state label. `DEGRADED` and `RECOVERING` are kept
    distinct from `ERROR` per §12."""
    mapping = {
        HealthState.OK: "ok",
        HealthState.DEGRADED: "degraded",
        HealthState.RECOVERING: "recovering",
        HealthState.ERROR: "error",
        HealthState.UNKNOWN: "unknown",
    }
    return mapping[state]


def format_health_probe_state(state: HealthProbeState) -> str:
    """Operator-facing label for active read-only probes."""

    mapping = {
        HealthProbeState.OK: "ok",
        HealthProbeState.ERROR: "error",
        HealthProbeState.TIMEOUT: "timeout",
        HealthProbeState.NOT_CONFIGURED: "not configured",
        HealthProbeState.UNKNOWN: "unknown",
    }
    return mapping[state]


def format_latency_ms(value: float | None) -> str:
    """Render bounded probe latency; absent latency stays explicit."""

    if value is None:
        return _EM_DASH
    if value >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.0f}ms"


def build_health_probe_detail(probe: HealthSubsystemProbe) -> str:
    """Compact tooltip/detail text for subsystem probes."""

    return probe.detail or "probe completed without additional detail"


# ----------------------------------------------------------------------
# Freshness / physiology
# ----------------------------------------------------------------------


def format_freshness(
    freshness_s: float | None,
    *,
    is_stale: bool | None = None,
) -> str:
    """`12s (fresh)` / `78s (stale)` / `—`.

    §4.C.4 treats stale snapshots as first-class: the UI must say
    "stale", not hide the number.
    """
    if freshness_s is None:
        return _EM_DASH
    stale = bool(is_stale) if is_stale is not None else freshness_s >= _DEGREE_OF_FRESHNESS_STALE_S
    label = "stale" if stale else "fresh"
    return f"{freshness_s:.0f}s ({label})"


def operator_ready_for_submit(snapshot: SessionSummary | None) -> bool:
    """Return the console's safe-submit readiness for a live session.

    The worker's ``is_calibrating`` flag is lifecycle telemetry: it can
    remain true until the first stimulus has actually been injected. The
    console needs a narrower operator-readiness concept so the first
    stimulus is not deadlocked behind that same lifecycle transition.
    """

    if snapshot is None:
        return False
    if snapshot.is_calibrating is not True:
        # Legacy DTOs publish ``None`` here; keep rendering/submission
        # behavior ready rather than treating absent telemetry as blocked.
        return True
    accumulated = snapshot.calibration_frames_accumulated
    required = snapshot.calibration_frames_required
    if accumulated is None or required is None:
        return True
    return accumulated >= required


def format_calibration_status(snapshot: SessionSummary | None) -> tuple[UiStatusKind, str]:
    """Operator pill text for live-session calibration readiness."""

    if snapshot is None:
        return UiStatusKind.NEUTRAL, "Setup not ready"
    if operator_ready_for_submit(snapshot):
        return UiStatusKind.OK, "Healthy"
    accumulated = snapshot.calibration_frames_accumulated
    required = snapshot.calibration_frames_required
    if accumulated is None or required is None:
        return UiStatusKind.OK, "Healthy"
    current = min(max(accumulated, 0), required)
    return UiStatusKind.PROGRESS, f"Preparing smile baseline · {current}/{required} face frames"


def format_stimulus_progress_message(
    state: StimulusActionState,
    *,
    accepted_message: str | None = None,
    ready_for_submit: bool,
    countdown_seconds: float | None = None,
) -> str:
    """Plain-language next step for the persistent stimulus surfaces."""

    if state is StimulusActionState.SUBMITTING:
        return "Sending the test message now."
    if state is StimulusActionState.ACCEPTED:
        return accepted_message or "Test message accepted. Waiting for the next result."
    if state is StimulusActionState.MEASURING:
        if countdown_seconds is None or countdown_seconds <= 0:
            return "Response window closed. Analyzing the next result now."
        total = int(countdown_seconds)
        minutes, seconds = divmod(total, 60)
        return (
            "Measuring the response window now. "
            f"About {minutes:02d}:{seconds:02d} left before analysis starts."
        )
    if state is StimulusActionState.COMPLETED:
        if ready_for_submit:
            return "First result ready. You can send the next test message when needed."
        return "First result ready. Keep the face visible until the system is ready again."
    if state is StimulusActionState.FAILED:
        return accepted_message or "Test message failed. Please try again."
    if ready_for_submit:
        return "Send one test message and wait for the first result."
    return "Keep a clearly visible face on the phone screen until live analysis is ready."


def truncate_expected_greeting(greeting: str | None, *, limit: int = 60) -> str:
    """Short-form expected greeting for compact table cells.

    A greeting often runs several sentences; the action bar shows the
    full text, but cards/tables truncate with an ellipsis so layout
    does not break.
    """
    if greeting is None:
        return _EM_DASH
    stripped = greeting.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 1)] + "…"


# ----------------------------------------------------------------------
# Co-Modulation Index (§7C) — null-valid rendering
# ----------------------------------------------------------------------


def format_comodulation_index(summary: CoModulationSummary | None) -> str:
    """`0.342` or, when null-valid, the `null_reason` text verbatim.

    §7C defines a null result as a valid outcome when insufficient
    aligned non-stale pairs exist. Rendering "insufficient aligned pairs"
    (the reason) instead of a zero or an error preserves the meaning.
    """
    if summary is None:
        return _EM_DASH
    if summary.co_modulation_index is None:
        return summary.null_reason or "insufficient data"
    return f"{summary.co_modulation_index:+.3f}"


# ----------------------------------------------------------------------
# Compound explainers — used on Overview cards and detail panes
# ----------------------------------------------------------------------


def build_reward_explanation(encounter: EncounterSummary) -> str:
    """Human-readable breakdown of the §7B gated reward.

    The text surfaces every input the pipeline used:
      `r_t = p90_intensity × semantic_gate`
    Calling out `semantic_gate=0` explicitly — "gate closed, reward
    suppressed" — prevents an operator from misreading a zero reward
    as a frame-count issue, and vice-versa for `n_frames_in_window=0`.
    """
    labels = reward_detail_labels()
    if encounter.n_frames_in_window == 0:
        return "No usable face frames in the measurement window — reward was not computed."
    if encounter.semantic_gate == 0:
        confidence_text = format_semantic_confidence(encounter.semantic_confidence)
        measured_signal = format_reward(encounter.p90_intensity)
        return (
            f"Greeting did not match with {confidence_text} confidence, so the reward "
            f"was held back. Smile signal {measured_signal} was measured but not used."
        )
    parts: list[str] = []
    gate_text = "yes" if encounter.semantic_gate == 1 else _EM_DASH
    parts.append(
        f"{labels.reward_formula}: {format_reward(encounter.p90_intensity)} × "
        f"{gate_text} = {format_reward(encounter.gated_reward)}."
    )
    if encounter.au12_baseline_pre is not None:
        parts.append(f"Before-greeting smile level {format_reward(encounter.au12_baseline_pre)}.")
    if encounter.n_frames_in_window is not None:
        parts.append(f"{encounter.n_frames_in_window} usable face frame(s) in window.")
    if encounter.physiology_attached:
        stale = encounter.physiology_stale is True
        parts.append(
            "Physiology attached " + ("(stale snapshot)." if stale else "(fresh snapshot).")
        )
    return " ".join(parts)


def build_acoustic_explanation(summary: ObservationalAcousticSummary | None) -> str:
    """One-line §7D observational-acoustic summary for operator panes.

    Validity is shown separately for F0 and perturbation windows so an
    invalid F0 baseline never hides a valid perturbation measurement (or
    vice versa). Null dependent deltas read as "not measured" with the
    nearest available reason instead of zero or a raw null marker.
    """
    if summary is None:
        return "No acoustic data for this encounter."

    parts: list[str] = []
    parts.append(
        "F0 windows: "
        f"measure {format_acoustic_validity(summary.f0_valid_measure)}, "
        f"baseline {format_acoustic_validity(summary.f0_valid_baseline)}"
    )
    parts.append(
        "F0 means: "
        f"measure {format_f0_hz(summary.f0_mean_measure_hz)}, "
        f"baseline {format_f0_hz(summary.f0_mean_baseline_hz)}"
    )
    parts.append(_f0_delta_line(summary))
    parts.append(
        "Perturbation windows: "
        f"measure {format_acoustic_validity(summary.perturbation_valid_measure)}, "
        f"baseline {format_acoustic_validity(summary.perturbation_valid_baseline)}"
    )
    parts.append(
        _perturbation_delta_line(
            "Jitter",
            summary.jitter_delta,
            summary.perturbation_valid_measure,
            summary.perturbation_valid_baseline,
            summary.jitter_mean_measure,
            summary.jitter_mean_baseline,
        )
    )
    parts.append(
        _perturbation_delta_line(
            "Shimmer",
            summary.shimmer_delta,
            summary.perturbation_valid_measure,
            summary.perturbation_valid_baseline,
            summary.shimmer_mean_measure,
            summary.shimmer_mean_baseline,
        )
    )
    return " • ".join(parts)


def _f0_delta_line(summary: ObservationalAcousticSummary) -> str:
    if _has_acoustic_number(summary.f0_delta_semitones):
        return f"F0 Δ {format_semitone_delta(summary.f0_delta_semitones)}"

    invalid_reason = _window_invalid_reason(
        summary.f0_valid_measure,
        summary.f0_valid_baseline,
        "F0",
    )
    if invalid_reason:
        return f"F0 Δ not measured ({invalid_reason})"

    missing_reasons: list[str] = []
    if not _has_acoustic_number(summary.f0_mean_measure_hz):
        missing_reasons.append("measure F0 mean absent")
    if not _has_acoustic_number(summary.f0_mean_baseline_hz):
        missing_reasons.append("baseline F0 mean absent")
    if missing_reasons:
        return f"F0 Δ not measured ({'; '.join(missing_reasons)})"
    return "F0 Δ not measured (semitone delta unavailable)"


def _perturbation_delta_line(
    label: str,
    value: float | None,
    measure_valid: bool | None,
    baseline_valid: bool | None,
    measure_mean: float | None,
    baseline_mean: float | None,
) -> str:
    if _has_acoustic_number(value):
        return f"{label} Δ {format_perturbation_delta(value)}"

    invalid_reason = _window_invalid_reason(
        measure_valid,
        baseline_valid,
        "perturbation",
    )
    if invalid_reason:
        return f"{label} Δ not measured ({invalid_reason})"

    mean_label = label.lower()
    missing_reasons: list[str] = []
    if not _has_acoustic_number(measure_mean):
        missing_reasons.append(f"measure {mean_label} mean absent")
    if not _has_acoustic_number(baseline_mean):
        missing_reasons.append(f"baseline {mean_label} mean absent")
    if missing_reasons:
        return f"{label} Δ not measured ({'; '.join(missing_reasons)})"
    return f"{label} Δ not measured"


def _window_invalid_reason(
    measure_valid: bool | None,
    baseline_valid: bool | None,
    family_label: str,
) -> str | None:
    invalid_windows: list[str] = []
    if measure_valid is False:
        invalid_windows.append("measure")
    if baseline_valid is False:
        invalid_windows.append("baseline")
    if not invalid_windows:
        return None
    window_text = " and ".join(invalid_windows)
    noun = "windows" if len(invalid_windows) > 1 else "window"
    return f"{window_text} {family_label} {noun} invalid"


def build_physiology_explanation(snapshot: SessionPhysiologySnapshot | None) -> str:
    """One-line physiology summary used on Overview / session header.

    Distinguishes fresh / stale / absent so an operator never has to
    guess whether "no value" means "not wearing the strap" or "snapshot
    is old".
    """
    if snapshot is None:
        return "No physiology data for this session."
    parts: list[str] = []
    parts.append(_role_line("streamer", snapshot.streamer))
    parts.append(_role_line("operator", snapshot.operator))
    parts.append(f"shared movement: {format_comodulation_index(snapshot.comodulation)}")
    return " • ".join(parts)


def _role_line(label: str, snap: PhysiologyCurrentSnapshot | None) -> str:
    labels = physiology_labels()
    if snap is None:
        return f"{label}: absent"
    if snap.rmssd_ms is None:
        return f"{label}: {labels.no_rmssd_summary.lower()}"
    freshness = format_freshness(snap.freshness_s, is_stale=snap.is_stale)
    hr = f"{snap.heart_rate_bpm} bpm" if snap.heart_rate_bpm is not None else _EM_DASH
    return f"{label}: variability {snap.rmssd_ms:.0f}ms, heart rate {hr}, {freshness}"


def build_health_detail(row: HealthSubsystemStatus) -> str:
    """Health-row detail line combining state, recovery hint, and last-success
    timestamp. §12 recovery-mode states read distinct from errors.
    """
    parts: list[str] = [format_health_state(row.state)]
    if row.recovery_mode:
        parts.append(f"recovery: {row.recovery_mode}")
    if row.detail:
        parts.append(row.detail)
    if row.last_success_utc is not None:
        parts.append(f"last ok {format_timestamp(row.last_success_utc)}")
    if row.operator_action_hint:
        parts.append(f"hint: {row.operator_action_hint}")
    return " — ".join(parts)


def ui_status_for_health(row: HealthSubsystemStatus) -> UiStatusKind:
    """Map a subsystem's `HealthState` to a `UiStatusKind` palette bucket.

    Recovering is visually distinct from degraded so the operator sees
    that the subsystem is self-healing rather than "just broken".
    """
    mapping: dict[HealthState, UiStatusKind] = {
        HealthState.OK: UiStatusKind.OK,
        HealthState.DEGRADED: UiStatusKind.WARN,
        HealthState.RECOVERING: UiStatusKind.PROGRESS,
        HealthState.ERROR: UiStatusKind.ERROR,
        HealthState.UNKNOWN: UiStatusKind.NEUTRAL,
    }
    return mapping[row.state]
