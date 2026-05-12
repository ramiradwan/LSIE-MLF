"""
Shared Pydantic contracts for API Server ↔ Operator Console payloads.

This module validates derived read/action DTOs used by operator routes, clients,
stores, viewmodels, and table models. It enforces UTC-aware timestamps, canonical
operator terminology (§0, §13.15), raw-media exclusion under §5 data governance,
§7B reward-explanation fields, §7C null-valid co-modulation, §7E attribution
readbacks, and §12 health/recovery states. It has no backend/ORM imports,
persists no biometric media, and does not compute rewards or mutate pipeline
state.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import AnyUrl, ConfigDict, Field, TypeAdapter, field_validator
from pydantic import BaseModel as PydanticBaseModel

_STREAM_URL_ADAPTER = TypeAdapter(AnyUrl)

# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


class UiStatusKind(StrEnum):
    """Visual status categories for pills, cards, and table cells.

    Maps subsystem/encounter/stimulus state into a small set of palette
    buckets. Widgets read the enum; the palette lookup is theme-side.

    `MUTED` is reserved for explicitly unconfigured surfaces — the
    operator should read it as "the system isn't expected to report
    here", visually distinct from `NEUTRAL` ("nothing to say yet") and
    from `WARN`/`ERROR` ("something is wrong").
    """

    OK = "ok"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    NEUTRAL = "neutral"
    PROGRESS = "progress"
    MUTED = "muted"


class AlertSeverity(StrEnum):
    """Severity for operator alert banner + feed."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertKind(StrEnum):
    """Kind of alert surfaced on the attention queue.

    Values are §12-aligned: subsystem state transitions, physiology
    staleness, stimulus lifecycle failures, and data-gap markers are the
    events the operator must triage.
    """

    SUBSYSTEM_DEGRADED = "subsystem_degraded"
    SUBSYSTEM_RECOVERING = "subsystem_recovering"
    SUBSYSTEM_ERROR = "subsystem_error"
    PHYSIOLOGY_STALE = "physiology_stale"
    STIMULUS_FAILED = "stimulus_failed"
    SESSION_ENDED = "session_ended"
    DATA_GAP = "data_gap"


class EncounterState(StrEnum):
    """§7B encounter lifecycle states.

    `AWAITING_STIMULUS`/`STIMULUS_ISSUED`/`MEASURING`/`COMPLETED` are the
    happy path. The two rejection states mirror §7B's reward guards:
    a zero semantic gate zeros the reward, and a measurement window with
    no valid AU12 frames yields no reward at all.
    """

    AWAITING_STIMULUS = "awaiting_stimulus"
    STIMULUS_ISSUED = "stimulus_issued"
    MEASURING = "measuring"
    COMPLETED = "completed"
    REJECTED_NO_FRAMES = "rejected_no_frames"
    REJECTED_GATE_CLOSED = "rejected_gate_closed"


class StimulusActionState(StrEnum):
    """Action-bar lifecycle states for the stimulus rail (§4.C)."""

    IDLE = "idle"
    SUBMITTING = "submitting"
    ACCEPTED = "accepted"
    MEASURING = "measuring"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthState(StrEnum):
    """§12 subsystem rollup states.

    `DEGRADED` and `RECOVERING` are deliberately distinct from `ERROR`
    so the UI can surface in-progress self-heal paths (FFmpeg restart,
    Azure retry, DB write buffer) instead of flattening them to failure.
    """

    OK = "ok"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    ERROR = "error"
    UNKNOWN = "unknown"


class HealthProbeState(StrEnum):
    """Read-only subsystem probe states.

    Probe diagnostics are intentionally separate from §12 rollup states:
    they must not feed the alert pipeline, and missing deployment
    configuration must render as `NOT_CONFIGURED` instead of `ERROR`.
    """

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_CONFIGURED = "not_configured"
    UNKNOWN = "unknown"


class CloudAuthState(StrEnum):
    SIGNED_OUT = "signed_out"
    SIGNED_IN = "signed_in"
    REFRESH_TOKEN_UNAVAILABLE = "refresh_token_unavailable"
    SECRET_STORE_UNAVAILABLE = "secret_store_unavailable"
    REFRESH_FAILED = "refresh_failed"


class CloudActionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CloudExperimentRefreshStatus(StrEnum):
    APPLIED = "applied"
    FAILED = "failed"


class CloudOperatorErrorCode(StrEnum):
    AUTHORIZATION_FAILED = "authorization_failed"
    BUNDLE_CHANGED = "bundle_changed"
    BUNDLE_EXPIRED = "bundle_expired"
    CLOUD_UNAVAILABLE = "cloud_unavailable"
    INVALID_RESPONSE = "invalid_response"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"
    REFRESH_FAILED = "refresh_failed"
    REFRESH_TOKEN_UNAVAILABLE = "refresh_token_unavailable"
    SECRET_STORE_UNAVAILABLE = "secret_store_unavailable"
    SIGNATURE_FAILED = "signature_failed"
    UNAUTHORIZED = "unauthorized"


# ----------------------------------------------------------------------
# Timestamp validator (shared behavior — UTC-aware only)
# ----------------------------------------------------------------------


def _require_utc(value: datetime | None) -> datetime | None:
    """Reject naive datetimes.

    Authoritative timestamps in LSIE-MLF (§4.C stimulus time, §4.E.2
    physiology source_timestamp_utc, §7C co-modulation window bounds)
    are UTC-aware. A naive datetime at this boundary almost always
    means the UI wall clock leaked through; raise instead.
    """

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be UTC-aware (tzinfo required)")
    return value


# ----------------------------------------------------------------------
# Shared model config / validators
# ----------------------------------------------------------------------


class OperatorConsoleModel(PydanticBaseModel):
    """
    Shared base for operator-console DTOs.

    Accepts fields declared by subclasses and produces Pydantic models that
    reject NaN/Infinity while preserving explicit ``None`` as JSON ``null``. It
    does not import backend services, persist data, or infer missing analytics.
    """

    model_config = ConfigDict(allow_inf_nan=False, ser_json_inf_nan="null")

    @field_validator("*", mode="after")
    @classmethod
    def _finite_floats_only(cls, value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite floats are not permitted; use None for null values")
        return value


# ----------------------------------------------------------------------
# Session-level DTOs
# ----------------------------------------------------------------------


class ObservationalAcousticSummary(OperatorConsoleModel):
    """Canonical §7D observational acoustic summary for operator/API payloads.

    The deterministic null-stimulus contract is represented directly on this
    model: validity flags and nullable summary statistics are preserved as
    JSON `null` when omitted, while coverage and non-negative mean fields are
    range-checked at the schema boundary.
    """

    f0_valid_measure: bool | None = None
    f0_valid_baseline: bool | None = None
    perturbation_valid_measure: bool | None = None
    perturbation_valid_baseline: bool | None = None
    voiced_coverage_measure_s: float | None = Field(default=None, ge=0.0)
    voiced_coverage_baseline_s: float | None = Field(default=None, ge=0.0)
    f0_mean_measure_hz: float | None = Field(default=None, ge=0.0)
    f0_mean_baseline_hz: float | None = Field(default=None, ge=0.0)
    f0_delta_semitones: float | None = None
    jitter_mean_measure: float | None = Field(default=None, ge=0.0)
    jitter_mean_baseline: float | None = Field(default=None, ge=0.0)
    jitter_delta: float | None = None
    shimmer_mean_measure: float | None = Field(default=None, ge=0.0)
    shimmer_mean_baseline: float | None = Field(default=None, ge=0.0)
    shimmer_delta: float | None = None


class SemanticEvaluationSummary(OperatorConsoleModel):
    """Bounded §7E semantic attribution readback for encounter aggregates.

    `reasoning` is the bounded reason code persisted on the AttributionEvent.
    The confidence/probability and method metadata are observational readbacks
    and do not change the §7B reward path.
    """

    reasoning: str | None = None
    is_match: bool | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_method: str | None = None
    semantic_method_version: str | None = None


class AttributionSummary(OperatorConsoleModel):
    """§7E attribution diagnostics summarized for operator aggregates.

    The fields are derived analytics only: soft semantic reward candidate,
    baseline-aware AU12 lift diagnostics, lag-aware synchrony diagnostics, and
    the event→outcome lag when a link exists. Absence of all source values is
    represented by the parent field being ``None`` rather than an empty object.
    """

    finality: Literal["online_provisional", "offline_final"] | None = None
    soft_reward_candidate: float | None = Field(default=None, ge=0.0, le=1.0)
    au12_baseline_pre: float | None = Field(default=None, ge=0.0)
    au12_lift_p90: float | None = Field(default=None, ge=0.0)
    au12_lift_peak: float | None = Field(default=None, ge=0.0)
    au12_peak_latency_ms: float | None = None
    sync_peak_corr: float | None = Field(default=None, ge=-1.0, le=1.0)
    sync_peak_lag: int | None = Field(default=None, ge=0)
    outcome_link_lag_s: float | None = Field(default=None, ge=0.0)


class SessionSummary(OperatorConsoleModel):
    """Lightweight session card for overview/history surfaces.

    Fields mirror §4.E.1 operator concerns: identity, status, duration,
    the §4.C active arm + expected greeting, the operator-readiness
    calibration summary, and the most recent §7B reward outcome so the
    operator sees adaptive progress at a glance.
    """

    session_id: UUID
    status: str
    started_at_utc: datetime
    ended_at_utc: datetime | None = None
    duration_s: float | None = None
    experiment_id: str | None = None
    active_arm: str | None = None
    expected_greeting: str | None = None
    is_calibrating: bool | None = None
    calibration_frames_accumulated: int | None = Field(default=None, ge=0)
    calibration_frames_required: int | None = Field(default=None, ge=0)
    last_segment_completed_at_utc: datetime | None = None
    latest_reward: float | None = None
    latest_semantic_gate: int | None = Field(default=None, ge=0, le=1)

    @field_validator(
        "started_at_utc",
        "ended_at_utc",
        "last_segment_completed_at_utc",
    )
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class EncounterSummary(OperatorConsoleModel):
    """Per-segment encounter row with full §7B reward explanation.

    The reward-explanation fields (`p90_intensity`, `semantic_gate`,
    `gated_reward`, `n_frames_in_window`, `au12_baseline_pre`) are the
    exact inputs the pipeline uses — exposing them is what makes the
    Live Session page operator-trustable.
    """

    encounter_id: str
    session_id: UUID
    segment_timestamp_utc: datetime
    state: EncounterState
    active_arm: str | None = None
    expected_greeting: str | None = None
    stimulus_time_utc: datetime | None = None
    semantic_gate: int | None = Field(default=None, ge=0, le=1)
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    transcription: str | None = None
    p90_intensity: float | None = None
    gated_reward: float | None = None
    n_frames_in_window: int | None = Field(default=None, ge=0)
    au12_baseline_pre: float | None = None
    observational_acoustic: ObservationalAcousticSummary | None = None
    semantic_evaluation: SemanticEvaluationSummary | None = None
    attribution: AttributionSummary | None = None
    physiology_attached: bool = False
    physiology_stale: bool | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("segment_timestamp_utc", "stimulus_time_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class LatestEncounterSummary(OperatorConsoleModel):
    """Trimmed encounter card shown on the Overview page.

    A strict subset of `EncounterSummary` — the overview renders the
    headline reward fields without the full detail pane.
    """

    encounter_id: str
    session_id: UUID
    segment_timestamp_utc: datetime
    state: EncounterState
    active_arm: str | None = None
    expected_greeting: str | None = None
    stimulus_time_utc: datetime | None = None
    semantic_gate: int | None = Field(default=None, ge=0, le=1)
    p90_intensity: float | None = None
    gated_reward: float | None = None
    n_frames_in_window: int | None = Field(default=None, ge=0)
    observational_acoustic: ObservationalAcousticSummary | None = None
    semantic_evaluation: SemanticEvaluationSummary | None = None
    attribution: AttributionSummary | None = None

    @field_validator("segment_timestamp_utc", "stimulus_time_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


# ----------------------------------------------------------------------
# Experiment DTOs (§7B)
# ----------------------------------------------------------------------


class ArmDecisionEvidence(OperatorConsoleModel):
    arm_id: str
    pre_update_alpha: float = Field(..., gt=0.0)
    pre_update_beta: float = Field(..., gt=0.0)
    sampled_theta: float | None = Field(default=None, ge=0.0, le=1.0)


class BanditDecisionEvidence(OperatorConsoleModel):
    selection_time_utc: datetime
    selected_arm_id: str
    policy_version: str
    decision_context_hash: str = Field(..., pattern="^[0-9a-f]{64}$")
    random_seed: int = Field(..., ge=0, le=18446744073709551615)
    arm_evidence: list[ArmDecisionEvidence] = Field(default_factory=list)

    @field_validator("selection_time_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class ArmSummary(OperatorConsoleModel):
    """One arm of a Thompson-sampled experiment.

    Posterior parameters (`posterior_alpha`, `posterior_beta`) and
    evaluation variance are read-only readbacks. Human-owned management
    fields (`greeting_text`, `enabled`, `end_dated_at`) let the operator
    see arm metadata without implying posterior edits are supported.
    """

    arm_id: str
    greeting_text: str
    posterior_alpha: float = Field(..., gt=0.0)
    posterior_beta: float = Field(..., gt=0.0)
    evaluation_variance: float | None = Field(default=None, ge=0.0)
    selection_count: int = Field(default=0, ge=0)
    recent_reward_mean: float | None = None
    recent_semantic_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_evidence: ArmDecisionEvidence | None = None
    enabled: bool = True
    end_dated_at: datetime | None = None

    @field_validator("end_dated_at")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class ExperimentSummary(OperatorConsoleModel):
    """Compact experiment card for Overview / session header."""

    experiment_id: str
    label: str | None = None
    active_arm_id: str | None = None
    arm_count: int = Field(default=0, ge=0)
    last_updated_utc: datetime | None = None
    latest_reward: float | None = None

    @field_validator("last_updated_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class ExperimentDetail(OperatorConsoleModel):
    """Full experiment readback for the Experiments page.

    `last_update_summary` is a pre-formatted human-readable line — the
    formatter lives in `services/operator_console/formatters.py`
    so the route layer is free of string building.
    """

    experiment_id: str
    label: str | None = None
    active_arm_id: str | None = None
    decision_evidence: BanditDecisionEvidence | None = None
    arms: list[ArmSummary] = Field(default_factory=list)
    last_update_summary: str | None = None
    last_updated_utc: datetime | None = None

    @field_validator("last_updated_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


# ----------------------------------------------------------------------
# Physiology DTOs (§4.C.4, §4.E.2, §7C)
# ----------------------------------------------------------------------


class PhysiologyCurrentSnapshot(OperatorConsoleModel):
    """Per-`subject_role` latest physiology readback.

    Mirrors the fields Module E persists in `physiology_log` plus the
    Orchestrator's freshness/staleness computation from §4.C.4. No raw
    HRV trace or raw provider payload — only the derived metrics.
    """

    subject_role: Literal["streamer", "operator"]
    rmssd_ms: float | None = Field(default=None, ge=0.0)
    heart_rate_bpm: int | None = Field(default=None, ge=20, le=300)
    provider: str | None = None
    source_timestamp_utc: datetime | None = None
    freshness_s: float | None = Field(default=None, ge=0.0)
    is_stale: bool | None = None

    @field_validator("source_timestamp_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class CoModulationSummary(OperatorConsoleModel):
    """§7C rolling Co-Modulation Index summary.

    `co_modulation_index=None` + `null_reason` set is the documented
    valid outcome when insufficient aligned non-stale pairs exist for
    the rolling window. The UI must render this state as legitimate,
    not as an error.
    """

    session_id: UUID
    co_modulation_index: float | None = Field(default=None, ge=-1.0, le=1.0)
    n_paired_observations: int = Field(default=0, ge=0)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    streamer_rmssd_mean: float | None = Field(default=None, ge=0.0)
    operator_rmssd_mean: float | None = Field(default=None, ge=0.0)
    null_reason: str | None = None
    window_start_utc: datetime | None = None
    window_end_utc: datetime | None = None

    @field_validator("window_start_utc", "window_end_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class SessionPhysiologySnapshot(OperatorConsoleModel):
    """Composed physiology payload returned from the operator physiology
    route: per-role current snapshots plus the latest co-modulation row.
    """

    session_id: UUID
    streamer: PhysiologyCurrentSnapshot | None = None
    operator: PhysiologyCurrentSnapshot | None = None
    comodulation: CoModulationSummary | None = None
    generated_at_utc: datetime

    @field_validator("generated_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None  # value is not Optional here
        return validated


# ----------------------------------------------------------------------
# Health DTOs (§12)
# ----------------------------------------------------------------------


class HealthSubsystemStatus(OperatorConsoleModel):
    """Per-subsystem health row.

    `recovery_mode` and `operator_action_hint` exist specifically for
    §12's degraded-but-recovering states (ADB drift freeze/reset,
    FFmpeg restart, Azure retry-then-null, DB write buffering). They
    stay null when the subsystem is fully `OK` or fully `ERROR`.
    """

    subsystem_key: str
    label: str
    state: HealthState
    last_success_utc: datetime | None = None
    detail: str | None = None
    recovery_mode: str | None = None
    operator_action_hint: str | None = None

    @field_validator("last_success_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class HealthSubsystemProbe(OperatorConsoleModel):
    """One bounded read-only subsystem connectivity diagnostic.

    These rows augment the operator Health page with active checks such
    as Postgres ``SELECT 1`` and Redis ``PING``. They are deliberately
    not §12 subsystem statuses: probe failures are visible diagnostics
    only and must not synthesize or mutate alert events.
    """

    subsystem_key: str
    label: str
    state: HealthProbeState
    latency_ms: float | None = Field(default=None, ge=0.0)
    detail: str | None = None
    checked_at_utc: datetime

    @field_validator("checked_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class HealthSnapshot(OperatorConsoleModel):
    """Overall health rollup + per-subsystem rows + keyed bounded probes."""

    generated_at_utc: datetime
    overall_state: HealthState
    subsystems: list[HealthSubsystemStatus] = Field(default_factory=list)
    subsystem_probes: dict[str, HealthSubsystemProbe] = Field(default_factory=dict)
    degraded_count: int = Field(default=0, ge=0)
    recovering_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)

    @field_validator("subsystem_probes")
    @classmethod
    def _probes_keyed_by_subsystem(
        cls, value: dict[str, HealthSubsystemProbe]
    ) -> dict[str, HealthSubsystemProbe]:
        for key, probe in value.items():
            if key != probe.subsystem_key:
                raise ValueError("subsystem_probes keys must match probe subsystem_key")
        return value

    @field_validator("generated_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


# ----------------------------------------------------------------------
# Alert feed
# ----------------------------------------------------------------------


class AlertEvent(OperatorConsoleModel):
    """One row on the operator attention queue."""

    alert_id: str
    severity: AlertSeverity
    kind: AlertKind
    message: str
    session_id: UUID | None = None
    subsystem_key: str | None = None
    emitted_at_utc: datetime
    acknowledged: bool = False

    @field_validator("emitted_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


# ----------------------------------------------------------------------
# Overview composite
# ----------------------------------------------------------------------


class OverviewSnapshot(OperatorConsoleModel):
    """Top-level payload for `GET /api/v1/operator/overview`.

    A composed view that the Overview page renders as its six cards.
    Every component is itself a validated DTO; no raw dicts ever cross
    this boundary.
    """

    generated_at_utc: datetime
    active_session: SessionSummary | None = None
    latest_encounter: LatestEncounterSummary | None = None
    experiment_summary: ExperimentSummary | None = None
    physiology: SessionPhysiologySnapshot | None = None
    health: HealthSnapshot | None = None
    alerts: list[AlertEvent] = Field(default_factory=list)

    @field_validator("generated_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


OperatorEventType: TypeAlias = Literal[
    "overview",
    "sessions",
    "live_session",
    "encounters",
    "experiment_summaries",
    "experiment",
    "physiology",
    "health",
    "alerts",
]

OperatorEventPayload: TypeAlias = (
    OverviewSnapshot
    | list[SessionSummary]
    | SessionSummary
    | list[EncounterSummary]
    | list[ExperimentSummary]
    | ExperimentDetail
    | SessionPhysiologySnapshot
    | HealthSnapshot
    | list[AlertEvent]
)


class OperatorStateBootstrap(OperatorConsoleModel):
    generated_at_utc: datetime
    overview: OverviewSnapshot
    sessions: list[SessionSummary] = Field(default_factory=list)
    live_session: SessionSummary | None = None
    encounters: list[EncounterSummary] = Field(default_factory=list)
    experiment_summaries: list[ExperimentSummary] = Field(default_factory=list)
    experiment: ExperimentDetail | None = None
    physiology: SessionPhysiologySnapshot | None = None
    health: HealthSnapshot
    alerts: list[AlertEvent] = Field(default_factory=list)

    @field_validator("generated_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class OperatorEventEnvelope(OperatorConsoleModel):
    event_id: str = Field(..., min_length=1)
    event_type: OperatorEventType
    cursor: str = Field(..., min_length=1)
    generated_at_utc: datetime
    payload: OperatorEventPayload

    @field_validator("generated_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


# ----------------------------------------------------------------------
# Stimulus action DTOs (§4.C)
# ----------------------------------------------------------------------


class StimulusRequest(OperatorConsoleModel):
    """Operator-issued stimulus request.

    `client_action_id` is the idempotency key the API Server uses to
    collapse accidental double-submits. Authoritative `stimulus_time`
    is *not* carried on this request — the orchestrator assigns it, so
    the GUI wall clock never becomes the reward-window anchor.
    """

    operator_note: str | None = None
    client_action_id: UUID


class StimulusAccepted(OperatorConsoleModel):
    """API Server acknowledgment of a stimulus submission."""

    session_id: UUID
    client_action_id: UUID
    accepted: bool
    received_at_utc: datetime
    stimulus_time_utc: datetime | None = None
    message: str | None = None

    @field_validator("received_at_utc", "stimulus_time_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)


class SessionCreateRequest(OperatorConsoleModel):
    """Operator-issued request to begin a live session lifecycle.

    The API Server generates/publishes the session identifier and the
    orchestrator remains the sole owner of authoritative `started_at`.
    `client_action_id` is the API idempotency key for duplicate submits.
    """

    stream_url: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    client_action_id: UUID

    @field_validator("stream_url")
    @classmethod
    def _valid_stream_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        _STREAM_URL_ADAPTER.validate_python(stripped)
        return stripped

    @field_validator("experiment_id")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class SessionEndRequest(OperatorConsoleModel):
    """Operator-issued request to end a live session lifecycle."""

    client_action_id: UUID


class SessionLifecycleAccepted(OperatorConsoleModel):
    """API Server acknowledgment of a session lifecycle intent publish."""

    action: Literal["start", "end"]
    session_id: UUID
    client_action_id: UUID
    accepted: bool
    received_at_utc: datetime
    message: str | None = None

    @field_validator("received_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class CloudAuthStatus(OperatorConsoleModel):
    state: CloudAuthState
    checked_at_utc: datetime
    message: str | None = None
    retryable: bool = False

    @field_validator("checked_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class CloudSignInResult(OperatorConsoleModel):
    status: CloudActionStatus
    auth_state: CloudAuthState
    completed_at_utc: datetime
    message: str
    error_code: CloudOperatorErrorCode | None = None
    retryable: bool = False

    @field_validator("completed_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class ExperimentBundleRefreshResult(OperatorConsoleModel):
    status: CloudExperimentRefreshStatus
    completed_at_utc: datetime
    message: str
    bundle_id: str | None = None
    experiment_count: int = Field(default=0, ge=0)
    error_code: CloudOperatorErrorCode | None = None
    retryable: bool = False

    @field_validator("completed_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class ExperimentBundleRefreshRequest(OperatorConsoleModel):
    preview_token: str


class ExperimentBundleRefreshChange(OperatorConsoleModel):
    action: Literal["add", "update", "disable"]
    experiment_id: str
    arm_id: str
    label: str | None = None
    current_greeting_text: str | None = None
    cloud_greeting_text: str | None = None
    current_enabled: bool | None = None
    cloud_enabled: bool | None = None
    learned_state_preserved: bool = True


class ExperimentBundleRefreshPreview(OperatorConsoleModel):
    status: CloudActionStatus
    checked_at_utc: datetime
    message: str
    preview_token: str | None = None
    bundle_id: str | None = None
    policy_version: str | None = None
    experiment_count: int = Field(default=0, ge=0)
    added_count: int = Field(default=0, ge=0)
    updated_count: int = Field(default=0, ge=0)
    disabled_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    existing_preserved_count: int = Field(default=0, ge=0)
    changes: list[ExperimentBundleRefreshChange] = Field(default_factory=list)
    error_code: CloudOperatorErrorCode | None = None
    retryable: bool = False

    @field_validator("checked_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        validated = _require_utc(value)
        assert validated is not None
        return validated


class CloudOutboxSummary(OperatorConsoleModel):
    generated_at_utc: datetime
    pending_count: int = Field(default=0, ge=0)
    in_flight_count: int = Field(default=0, ge=0)
    dead_letter_count: int = Field(default=0, ge=0)
    retry_scheduled_count: int = Field(default=0, ge=0)
    redacted_count: int = Field(default=0, ge=0)
    earliest_next_attempt_utc: datetime | None = None
    last_error: str | None = None
    latest_experiment_refresh: ExperimentBundleRefreshResult | None = None

    @field_validator("generated_at_utc", "earliest_next_attempt_utc")
    @classmethod
    def _utc_only(cls, value: datetime | None) -> datetime | None:
        return _require_utc(value)
