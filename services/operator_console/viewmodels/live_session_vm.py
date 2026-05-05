"""Live Session page viewmodel.

The Live Session VM is the center of operator trust. It owns five
concerns:

  1. The paired `EncountersTableModel`, updated whenever the store's
     `encounters_changed` fires, without losing operator selection.
  2. The authoritative arm/greeting, read from the store's live-session
     DTO (never from the last row of the table — rows are historical
     and the header can update independently when a new §4.C stimulus
     lands).
  3. The stimulus action-bar lifecycle, driven by a `StimulusUiContext`
     held in the store. The VM is the *only* writer of the context
     aside from the coordinator's success/failure callbacks — the view
     hands operator notes in, the VM composes the context and stamps
     it back.
  4. The §7B reward explanation and the measurement-window countdown.
     The countdown is derived from the *authoritative* `stimulus_time`,
     either from the accepted desktop submit response or from encounter
     read-back, not from the operator's click wall clock.
  5. Session lifecycle controls (`start` / `end`) dispatched through
     the coordinator's one-shot API path. The console stays API-only;
     no direct operator-host Postgres/Redis coupling is introduced.

Spec references:
  §4.C           — `_active_arm`, `_expected_greeting`, authoritative
                   `_stimulus_time`; idempotent writes via client_action_id
  §4.E.1         — Live Session operator surface
  §7B            — r_t = p90_intensity × semantic_gate; reward
                   explanation surfaces the exact inputs the pipeline
                   used (P90, semantic gate, frames, baseline)
  §12            — write/poll failures flow through the page-scoped
                   error banner
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import (
    EncounterState,
    EncounterSummary,
    ExperimentSummary,
    HealthSubsystemStatus,
    ObservationalAcousticSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
    UiStatusKind,
)
from services.operator_console.formatters import (
    AcousticDetailDisplay,
    CauseEffectDisplay,
    LiveTelemetryDisplay,
    ReadinessDisplay,
    SemanticAttributionDiagnosticsDisplay,
    build_acoustic_detail_display,
    build_cause_effect_display,
    build_live_telemetry_display,
    build_readiness_display,
    build_reward_explanation,
    format_calibration_status,
    format_stimulus_progress_message,
    operator_ready_for_submit,
    semantic_attribution_diagnostics_for_encounter,
    ui_status_for_health,
)
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.viewmodels.base import ViewModelBase
from services.operator_console.workers import OneShotSignals

_STIMULUS_RESULT_WINDOW_S: float = 5.0
_STIMULUS_READBACK_TOLERANCE_S: float = 1.0
_STIMULUS_TERMINAL_READBACK_GRACE_S: float = 35.0
_SMILE_TIMELINE_WINDOW_S: float = 60.0
_SESSION_START_SCOPE = "session_start"
_SESSION_END_SCOPE = "session_end"

WAITING_FOR_DEVICE = "WAITING_FOR_DEVICE"
WAITING_FOR_FACE = "WAITING_FOR_FACE"
READY = "READY"

TtvDashboardMode = Literal["gate", "calibrating", "ready"]

_ADB_HEALTH_KEYS = frozenset(
    {
        "adb",
        "android_debug_bridge",
        "phone",
        "device",
    }
)
_ML_BACKEND_HEALTH_KEYS = frozenset(
    {
        "ml_backend",
        "ml_worker",
        "gpu_ml_worker",
        "inference",
        "facial_metrics",
    }
)
_AUDIO_CAPTURE_HEALTH_KEYS = frozenset({"audio_capture", "scrcpy_audio"})
_VIDEO_CAPTURE_HEALTH_KEYS = frozenset({"video_capture", "scrcpy_video"})
_LIVE_ANALYTICS_HEALTH_KEYS = frozenset({"live_analytics_producer"})
_LIVE_ANALYTICS_NOTICE = (
    "Waiting for the first result. Keep the face visible and wait for the "
    "response measurement window to finish before sending another stimulus."
)

SessionStartSubmitter = Callable[[SessionCreateRequest], OneShotSignals]
SessionEndSubmitter = Callable[[UUID, SessionEndRequest], OneShotSignals]


@dataclass(frozen=True)
class SmileTimelinePoint:
    timestamp_utc: datetime
    label: str
    intensity_percent: int | None = None
    marker: str | None = None


@dataclass(frozen=True)
class TtvSetupDisplay:
    state: str
    step_label: str
    title: str
    message: str
    dashboard_mode: TtvDashboardMode
    detail: str | None = None


class LiveSessionViewModel(ViewModelBase):
    """Owns the encounters table, stimulus lifecycle, and reward text."""

    # fmt: off
    selection_changed    = Signal(object)  # str | None (encounter_id)
    action_state_changed = Signal(object)  # StimulusUiContext
    state_changed        = Signal(str)
    # fmt: on

    def __init__(
        self,
        store: OperatorStore,
        encounters_model: EncountersTableModel,
        parent: QObject | None = None,
        *,
        default_experiment_id: str | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._encounters_model = encounters_model
        self._default_experiment_id = default_experiment_id
        self._selected_encounter_id: str | None = None
        self._start_session_submitter: SessionStartSubmitter | None = None
        self._end_session_submitter: SessionEndSubmitter | None = None
        self._inflight_session_controls: dict[str, OneShotSignals] = {}
        self._start_request_inflight = False
        self._end_request_inflight = False
        self._pending_start_session_id: UUID | None = None
        self._pending_end_session_id: UUID | None = None
        self._ttv_state = WAITING_FOR_DEVICE
        self._timeline_markers: list[SmileTimelinePoint] = []
        self._last_marker_action_id: UUID | None = None

        # Subscriptions — the VM does not refresh the model on its own
        # tick; it reacts to store mutations the coordinator drives.
        store.selected_session_changed.connect(self._on_selected_session_changed)
        store.encounters_changed.connect(self._on_encounters_changed)
        store.live_session_changed.connect(self._on_live_session_changed)
        store.stimulus_state_changed.connect(self._on_stimulus_state_changed)
        store.health_changed.connect(self._on_health_changed)
        store.experiment_summaries_changed.connect(self._on_experiment_summaries_changed)
        store.managed_experiment_changed.connect(self._on_managed_experiment_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)

        # Seed from whatever the store already has — useful when the VM
        # is constructed after a first poll has already landed.
        self._sync_encounters_model()
        self._reconcile_session_lifecycle_waiters()
        self._ttv_state = self._derive_ttv_state()

    # ------------------------------------------------------------------
    # Lifecycle binding
    # ------------------------------------------------------------------

    def bind_session_lifecycle_actions(
        self,
        start_submitter: SessionStartSubmitter,
        end_submitter: SessionEndSubmitter,
    ) -> None:
        """Inject coordinator-backed one-shot writers for start/end controls."""
        self._start_session_submitter = start_submitter
        self._end_session_submitter = end_submitter
        self.emit_changed()

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def session(self) -> SessionSummary | None:
        session = self._store.live_session()
        if session is None:
            return None
        selected_session_id = self._store.selected_session_id()
        if selected_session_id is None or session.session_id == selected_session_id:
            return session
        return None

    def encounters_model(self) -> EncountersTableModel:
        return self._encounters_model

    def selected_encounter(self) -> EncounterSummary | None:
        if self._selected_encounter_id is None:
            return None
        return self._encounters_model.encounter_by_id(self._selected_encounter_id)

    def selected_acoustic(self) -> ObservationalAcousticSummary | None:
        """Canonical §7D acoustic summary for the selected encounter, if any."""
        encounter = self.selected_encounter()
        return encounter.observational_acoustic if encounter is not None else None

    def acoustic_detail_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> AcousticDetailDisplay:
        """Preformatted §7D detail payload for exactly one encounter."""
        summary = encounter.observational_acoustic if encounter is not None else None
        return build_acoustic_detail_display(summary)

    def acoustic_detail(self) -> AcousticDetailDisplay:
        """Preformatted §7D detail payload for the VM's default encounter."""
        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._current_session_encounters())
        return self.acoustic_detail_for_encounter(encounter)

    def select_encounter(self, encounter_id: str | None) -> None:
        if encounter_id == self._selected_encounter_id:
            return
        self._selected_encounter_id = encounter_id
        self.selection_changed.emit(encounter_id)
        self.emit_changed()

    def active_arm(self) -> str | None:
        # §4.C: arm comes from the orchestrator-owned live-session DTO,
        # not from a row in the encounter table. The header can change
        # between encounters when the operator swaps arms; historical
        # rows must not override that.
        session = self.session()
        return session.active_arm if session is not None else None

    def expected_greeting(self) -> str | None:
        session = self.session()
        return session.expected_greeting if session is not None else None

    def ttv_state(self) -> str:
        return self._ttv_state

    def ttv_empty_title(self) -> str:
        if self._ttv_state == WAITING_FOR_FACE:
            return "Preparing live analysis"
        if self._ttv_state == READY:
            return "Healthy"
        return "Setup not ready"

    def ttv_empty_message(self) -> str:
        if self._ttv_state == WAITING_FOR_FACE:
            return "Open a stream or video with a clearly visible face on your phone."
        if self._ttv_state == READY:
            return "Live analysis is ready. Send one stimulus."
        return "Connect the Android phone with USB debugging allowed."

    def ttv_setup_display(self) -> TtvSetupDisplay:
        if self._ttv_state == WAITING_FOR_FACE:
            return TtvSetupDisplay(
                state=self._ttv_state,
                step_label="Step 2 of 3",
                title="Preparing live analysis",
                message="Keep a clearly visible face on the phone screen.",
                dashboard_mode="calibrating",
                detail=self._face_tracking_detail(),
            )
        if self._ttv_state == READY:
            return TtvSetupDisplay(
                state=self._ttv_state,
                step_label="Step 3 of 3",
                title="Healthy",
                message="Send one stimulus and wait for the observed response.",
                dashboard_mode="ready",
                detail=self._face_tracking_detail(),
            )
        adb_kind, adb_text = self.adb_status()
        detail = self.capture_status_detail()
        if adb_kind is UiStatusKind.OK:
            return TtvSetupDisplay(
                state=self._ttv_state,
                step_label="Step 1 of 3",
                title="Setup not ready",
                message="Start or select a Live Session to begin live analysis.",
                dashboard_mode="gate",
                detail=detail,
            )
        return TtvSetupDisplay(
            state=self._ttv_state,
            step_label="Step 1 of 3",
            title="Setup not ready",
            message="Connect the Android phone with USB debugging allowed.",
            dashboard_mode="gate",
            detail=detail,
        )

    def current_smile_intensity_percent(self) -> int | None:
        latest = _latest_encounter(self._current_session_encounters())
        if latest is None:
            return None
        return _smile_intensity_percent(latest.p90_intensity)

    def smile_timeline_points(self) -> list[SmileTimelinePoint]:
        encounters = self._current_session_encounters()
        latest = _latest_encounter(encounters)
        anchor = latest.segment_timestamp_utc if latest is not None else None
        if anchor is None and self._timeline_markers:
            anchor = max(marker.timestamp_utc for marker in self._timeline_markers)
        if anchor is None:
            return []
        window_start = anchor - timedelta(seconds=_SMILE_TIMELINE_WINDOW_S)
        points: list[SmileTimelinePoint] = []
        for encounter in encounters:
            if encounter.segment_timestamp_utc < window_start:
                continue
            points.append(
                SmileTimelinePoint(
                    timestamp_utc=encounter.segment_timestamp_utc,
                    label="Response signal (P90 AU12)",
                    intensity_percent=_smile_intensity_percent(encounter.p90_intensity),
                    marker=None,
                )
            )
        for marker in self._timeline_markers:
            if marker.timestamp_utc >= window_start:
                points.append(marker)
        return sorted(points, key=lambda point: point.timestamp_utc)

    def adb_status(self) -> tuple[UiStatusKind, str]:
        row = self._matching_health_row(_ADB_HEALTH_KEYS)
        if row is None:
            return UiStatusKind.NEUTRAL, "Phone status unknown"
        return ui_status_for_health(row), _plain_status_label("Phone", row)

    def ml_backend_status(self) -> tuple[UiStatusKind, str]:
        row = self._matching_health_row(_ML_BACKEND_HEALTH_KEYS)
        if row is None:
            return UiStatusKind.NEUTRAL, "Live analysis status unknown"
        return ui_status_for_health(row), _plain_status_label("Live analysis", row)

    def audio_capture_status(self) -> tuple[UiStatusKind, str]:
        row = self._matching_health_row(_AUDIO_CAPTURE_HEALTH_KEYS)
        if row is None:
            return UiStatusKind.NEUTRAL, "Audio capture status unknown"
        return ui_status_for_health(row), _plain_status_label("Audio capture", row)

    def video_capture_status(self) -> tuple[UiStatusKind, str]:
        row = self._matching_health_row(_VIDEO_CAPTURE_HEALTH_KEYS)
        if row is None:
            return UiStatusKind.NEUTRAL, "Video capture status unknown"
        return ui_status_for_health(row), _plain_status_label("Video capture", row)

    def capture_status_detail(self) -> str:
        statuses = (
            self.adb_status()[1],
            self.audio_capture_status()[1],
            self.video_capture_status()[1],
            self.ml_backend_status()[1],
        )
        return " · ".join(statuses)

    def live_analytics_status(self) -> tuple[UiStatusKind, str]:
        row = self._matching_health_row(_LIVE_ANALYTICS_HEALTH_KEYS)
        if row is None:
            return UiStatusKind.NEUTRAL, "Live analysis results unknown"
        return ui_status_for_health(row), _plain_status_label("Live analysis results", row)

    def has_live_encounter_analytics(self) -> bool:
        return bool(self._current_session_encounters())

    def live_analytics_notice(self) -> str | None:
        if self.session() is None or self.has_live_encounter_analytics():
            return None
        kind, _text = self.live_analytics_status()
        if kind is UiStatusKind.NEUTRAL or kind is UiStatusKind.OK:
            return None
        return _LIVE_ANALYTICS_NOTICE

    def readiness_display(self, now_utc: datetime | None = None) -> ReadinessDisplay:
        """Preformatted operator readiness payload for the primary session surface."""

        return build_readiness_display(
            ready_for_submit=self.operator_ready_for_submit(),
            calibration_status=self.calibration_status(),
            capture_detail=self.capture_status_detail(),
            progress_message=self.stimulus_progress_message(now_utc),
        )

    def live_telemetry_display(self, now_utc: datetime | None = None) -> LiveTelemetryDisplay:
        """Preformatted ticker payload for derived live-session telemetry."""

        ctx = self._store.stimulus_ui_context()
        return build_live_telemetry_display(
            stimulus_state=ctx.state,
            progress_message=self.stimulus_progress_message(now_utc),
            response_signal_percent=self.current_smile_intensity_percent(),
            live_status=self.live_analytics_status(),
        )

    def cause_effect_display_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> CauseEffectDisplay:
        """Operator-first cause/effect display for exactly one encounter."""

        return build_cause_effect_display(encounter)

    def cause_effect_display(self) -> CauseEffectDisplay:
        """Cause/effect display for selected or latest terminal encounter."""

        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._current_session_encounters())
        return self.cause_effect_display_for_encounter(encounter)

    def experiment_summaries(self) -> list[ExperimentSummary]:
        return self._store.experiment_summaries()

    def current_experiment_id(self) -> str | None:
        managed_id = self._store.managed_experiment_id()
        if managed_id:
            return managed_id
        summaries = self._store.experiment_summaries()
        if self._default_experiment_id is not None:
            for summary in summaries:
                if summary.experiment_id == self._default_experiment_id:
                    return summary.experiment_id
        return summaries[0].experiment_id if summaries else None

    def start_session_source_summary(self) -> str:
        return (
            "Capture comes from the connected Android device. Open the TikTok stream "
            "on the phone before you start the session."
        )

    def start_session_disabled_reason(self) -> str | None:
        if self.experiment_summaries():
            return None
        return "Create an experiment on the Experiments page before starting a session."

    def validate_start_session_inputs(self, experiment_id: str) -> str:
        normalized_experiment_id = experiment_id.strip()
        if not normalized_experiment_id:
            raise ValueError("Choose an experiment before starting the session.")
        try:
            request = SessionCreateRequest(
                stream_url=_synthesized_stream_url(),
                experiment_id=normalized_experiment_id,
                client_action_id=uuid4(),
            )
        except ValidationError as exc:
            for error in exc.errors():
                location = error.get("loc", ())
                if location == ("experiment_id",):
                    raise ValueError("Choose an experiment before starting the session.") from exc
            raise ValueError("Session start input is invalid.") from exc
        return request.experiment_id

    def session_start_in_progress(self) -> bool:
        return self._start_request_inflight or self._pending_start_session_id is not None

    def session_end_in_progress(self) -> bool:
        if self._end_request_inflight:
            return True
        if self._pending_end_session_id is None:
            return False
        selected_session_id = self._store.selected_session_id()
        if selected_session_id is not None and selected_session_id != self._pending_end_session_id:
            return False
        session = self.session()
        if session is None:
            return True
        if session.session_id != self._pending_end_session_id:
            return False
        return session.ended_at_utc is None

    def can_start_session(self) -> bool:
        return not self.session_start_in_progress() and not self.session_end_in_progress()

    def can_end_session(self) -> bool:
        session = self.session()
        if session is None or session.ended_at_utc is not None:
            return False
        return not self.session_start_in_progress() and not self.session_end_in_progress()

    # ------------------------------------------------------------------
    # Session lifecycle controls
    # ------------------------------------------------------------------

    def start_new_session(self, experiment_id: str) -> UUID | None:
        """Dispatch a start-session write through the injected coordinator path."""
        if self.session_start_in_progress() or self.session_end_in_progress():
            return None
        try:
            normalized_experiment_id = self.validate_start_session_inputs(experiment_id)
            request = SessionCreateRequest(
                stream_url=_synthesized_stream_url(),
                experiment_id=normalized_experiment_id,
                client_action_id=uuid4(),
            )
        except (ValidationError, ValueError) as exc:
            self.set_error(str(exc))
            return None
        if self._start_session_submitter is None:
            self.set_error("Session controls are unavailable.")
            return None

        self.set_error(None)
        self._start_request_inflight = True
        self.emit_changed()

        signals = self._start_session_submitter(request)
        handle_key = str(request.client_action_id)
        self._inflight_session_controls[handle_key] = signals
        signals.succeeded.connect(self._on_session_start_succeeded)
        signals.failed.connect(self._on_session_start_failed)
        signals.finished.connect(
            lambda _job, key=handle_key: self._on_session_control_finished(key)
        )
        return request.client_action_id

    def end_current_session(self) -> UUID | None:
        """Dispatch an end-session write for the currently displayed session."""
        if self.session_start_in_progress() or self.session_end_in_progress():
            return None
        session = self.session()
        if session is None:
            self.set_error("No active session selected.")
            return None
        if session.ended_at_utc is not None:
            self.set_error("Session has already ended.")
            return None
        if self._end_session_submitter is None:
            self.set_error("Session controls are unavailable.")
            return None

        request = SessionEndRequest(client_action_id=uuid4())
        self.set_error(None)
        self._end_request_inflight = True
        self._pending_end_session_id = session.session_id
        self.emit_changed()

        signals = self._end_session_submitter(session.session_id, request)
        handle_key = str(request.client_action_id)
        self._inflight_session_controls[handle_key] = signals
        signals.succeeded.connect(self._on_session_end_succeeded)
        signals.failed.connect(self._on_session_end_failed)
        signals.finished.connect(
            lambda _job, key=handle_key: self._on_session_control_finished(key)
        )
        return request.client_action_id

    def is_calibrating(self) -> bool:
        session = self.session()
        return bool(session is not None and session.is_calibrating is True)

    def operator_ready_for_submit(self) -> bool:
        """Console-level readiness derived from calibration telemetry.

        This intentionally differs from ``is_calibrating()``: the
        authoritative lifecycle flag may stay true until the first
        stimulus injection, while the operator may submit as soon as the
        published calibration frame threshold is reached.
        """
        return operator_ready_for_submit(self.session())

    def calibration_status(self) -> tuple[UiStatusKind, str]:
        return format_calibration_status(self.session())

    # ------------------------------------------------------------------
    # Stimulus lifecycle
    # ------------------------------------------------------------------

    def stimulus_ui_context(self) -> StimulusUiContext:
        return self._store.stimulus_ui_context()

    def build_stimulus_request_id(self) -> UUID:
        """Mint a fresh idempotency key for the next submission.

        Kept as a method (not inlined at the call site) so the VM owns
        the dedup-key lifetime: the shell's action bar asks the VM for
        a key, the VM remembers it on the context, and the coordinator
        carries the same key to the API.
        """
        return uuid4()

    def set_stimulus_submitting(self, note: str | None) -> UUID:
        """Move the action bar into SUBMITTING state and return the key.

        The context holds `client_action_id` through the full
        submitting → accepted → measuring → completed path so a
        re-send (retry, double-click) can reuse the same key — §4.C's
        idempotency contract is UI-facing here, not just server-side.
        """
        action_id = uuid4()
        ctx = StimulusUiContext(
            state=StimulusActionState.SUBMITTING,
            client_action_id=action_id,
            operator_note=note,
            submitted_at_utc=datetime.now(UTC),
            accepted_at_utc=None,
            authoritative_stimulus_time_utc=None,
        )
        self._store.set_stimulus_ui_context(ctx)
        return action_id

    def apply_stimulus_accepted(self, accepted: StimulusAccepted) -> None:
        """Apply the API Server's ack. `accepted=False` → FAILED."""
        ctx = self._store.stimulus_ui_context()
        if not accepted.accepted:
            new_ctx = StimulusUiContext(
                state=StimulusActionState.FAILED,
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                message=accepted.message,
            )
        else:
            new_ctx = StimulusUiContext(
                state=(
                    StimulusActionState.MEASURING
                    if accepted.stimulus_time_utc is not None
                    else StimulusActionState.ACCEPTED
                ),
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                accepted_at_utc=accepted.received_at_utc,
                authoritative_stimulus_time_utc=accepted.stimulus_time_utc,
                message=accepted.message,
            )
        self._store.set_stimulus_ui_context(new_ctx)

    def reconcile_authoritative_stimulus_time(self) -> None:
        """Reconcile the UI stimulus context against encounter readback."""
        ctx = self._store.stimulus_ui_context()
        if ctx.state not in (
            StimulusActionState.ACCEPTED,
            StimulusActionState.MEASURING,
        ):
            return
        encounters = self._current_session_encounters()
        latest = _latest_encounter_for_stimulus(encounters, ctx)
        if latest is None or latest.stimulus_time_utc is None:
            return
        next_state = (
            StimulusActionState.COMPLETED
            if _encounter_is_terminal_for_stimulus(latest)
            else StimulusActionState.MEASURING
        )
        if (
            next_state == ctx.state
            and ctx.authoritative_stimulus_time_utc == latest.stimulus_time_utc
        ):
            return
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=next_state,
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                accepted_at_utc=ctx.accepted_at_utc,
                authoritative_stimulus_time_utc=latest.stimulus_time_utc,
                message=ctx.message,
            )
        )
        if next_state == StimulusActionState.COMPLETED:
            self.select_encounter(latest.encounter_id)

    def measurement_window_remaining_s(self, now_utc: datetime) -> float | None:
        """Seconds remaining until the first stimulus result window closes."""
        ctx = self._store.stimulus_ui_context()
        if ctx.authoritative_stimulus_time_utc is None:
            return None
        elapsed = (now_utc - ctx.authoritative_stimulus_time_utc).total_seconds()
        remaining = _STIMULUS_RESULT_WINDOW_S - elapsed
        if remaining < 0.0:
            return 0.0
        return remaining

    def stimulus_progress_message(self, now_utc: datetime | None = None) -> str:
        """Single plain-language progress line for the stimulus surfaces."""

        current_time = now_utc or datetime.now(UTC)
        ctx = self._store.stimulus_ui_context()
        return format_stimulus_progress_message(
            ctx.state,
            accepted_message=ctx.message,
            ready_for_submit=self.operator_ready_for_submit(),
            countdown_seconds=self.measurement_window_remaining_s(current_time),
        )

    # ------------------------------------------------------------------
    # Reward explanation
    # ------------------------------------------------------------------

    def reward_explanation_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> str:
        """Formatted §7B reward text for exactly one detail-pane encounter."""
        if encounter is None:
            return "No completed encounter yet."
        return build_reward_explanation(encounter)

    def reward_explanation(self) -> str:
        """Formatted §7B reward text for the selected (or latest) encounter."""
        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._current_session_encounters())
        return self.reward_explanation_for_encounter(encounter)

    def acoustic_explanation(self) -> str:
        """Formatted §7D acoustic text for the selected (or latest) encounter."""
        return self.acoustic_detail().explanation

    def semantic_attribution_diagnostics_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> SemanticAttributionDiagnosticsDisplay:
        """Preformatted read-only §7E diagnostics for one encounter."""

        return semantic_attribution_diagnostics_for_encounter(encounter)

    def semantic_attribution_diagnostics(self) -> SemanticAttributionDiagnosticsDisplay:
        """Preformatted §7E diagnostics for the selected (or latest) encounter."""

        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._current_session_encounters())
        return self.semantic_attribution_diagnostics_for_encounter(encounter)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selected_session_changed(self, _session_id: object) -> None:
        self._sync_encounters_model()
        self._timeline_markers.clear()
        self._last_marker_action_id = None
        if (
            self._selected_encounter_id is not None
            and self._encounters_model.encounter_by_id(self._selected_encounter_id) is None
        ):
            self._selected_encounter_id = None
            self.selection_changed.emit(None)
        if self._store.stimulus_ui_context().state != StimulusActionState.IDLE:
            self._store.set_stimulus_ui_context(StimulusUiContext())
        self._reconcile_session_lifecycle_waiters()
        self._transition_ttv_state()
        self.emit_changed()

    def _on_encounters_changed(self, rows: object) -> None:
        # Defensive: the store always hands a list, but a typed object
        # parameter (for Qt marshalling) forces this shape-check.
        if isinstance(rows, list):
            filtered = self._current_session_encounters(rows)
            self._encounters_model.set_rows(filtered)
        else:
            self._sync_encounters_model()
        # Reconcile the stimulus UI context against the fresh rows —
        # the authoritative `stimulus_time` may have just landed.
        self.reconcile_authoritative_stimulus_time()
        # If the previously selected encounter has been evicted, drop
        # the selection silently so the view can update.
        if (
            self._selected_encounter_id is not None
            and self._encounters_model.encounter_by_id(self._selected_encounter_id) is None
        ):
            self._selected_encounter_id = None
            self.selection_changed.emit(None)
        self._transition_ttv_state()
        self.emit_changed()

    def _on_live_session_changed(self, session_update: object) -> None:
        # Arm / expected greeting / session-end readback may have moved.
        session = self.session()
        selected_session_id = self._store.selected_session_id()
        ended_selected_session = (
            isinstance(session_update, SessionSummary)
            and session_update.ended_at_utc is not None
            and (selected_session_id is None or session_update.session_id == selected_session_id)
        )
        if (
            session is None or session.ended_at_utc is not None or ended_selected_session
        ) and self._store.stimulus_ui_context().state != StimulusActionState.IDLE:
            self._store.set_stimulus_ui_context(StimulusUiContext())
        self._reconcile_session_lifecycle_waiters()
        self._transition_ttv_state()
        self.emit_changed()

    def _on_stimulus_state_changed(self, ctx: object) -> None:
        if isinstance(ctx, StimulusUiContext):
            self._record_timeline_marker(ctx)
            self.action_state_changed.emit(ctx)
        else:
            self.action_state_changed.emit(self._store.stimulus_ui_context())
        self.emit_changed()

    def _on_health_changed(self, _snapshot: object) -> None:
        self._transition_ttv_state()
        self.emit_changed()

    def _on_experiment_summaries_changed(self, _summaries: object) -> None:
        self.emit_changed()

    def _on_managed_experiment_changed(self, _experiment_id: object) -> None:
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope in (
            "live_session",
            "encounters",
            "stimulus",
            _SESSION_START_SCOPE,
            _SESSION_END_SCOPE,
        ):
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope in (
            "live_session",
            "encounters",
            "stimulus",
            _SESSION_START_SCOPE,
            _SESSION_END_SCOPE,
        ):
            self.set_error(None)

    def _on_session_start_succeeded(self, _job: str, payload: object) -> None:
        self._start_request_inflight = False
        if not isinstance(payload, SessionLifecycleAccepted) or not payload.accepted:
            self._pending_start_session_id = None
            message = (
                payload.message
                if isinstance(payload, SessionLifecycleAccepted) and payload.message
                else "Session start was not accepted."
            )
            self.set_error(message)
            self.emit_changed()
            return

        self._pending_start_session_id = payload.session_id
        self.set_error(None)
        self._store.set_selected_session_id(payload.session_id)
        self.emit_changed()

    def _on_session_start_failed(self, _job: str, error: object) -> None:
        self._start_request_inflight = False
        self._pending_start_session_id = None
        self.set_error(_error_message(error, default="Session start failed."))
        self.emit_changed()

    def _on_session_end_succeeded(self, _job: str, payload: object) -> None:
        self._end_request_inflight = False
        if not isinstance(payload, SessionLifecycleAccepted) or not payload.accepted:
            self._pending_end_session_id = None
            message = (
                payload.message
                if isinstance(payload, SessionLifecycleAccepted) and payload.message
                else "Session end was not accepted."
            )
            self.set_error(message)
            self.emit_changed()
            return

        self._pending_end_session_id = payload.session_id
        self.set_error(None)
        self.emit_changed()

    def _on_session_end_failed(self, _job: str, error: object) -> None:
        self._end_request_inflight = False
        self._pending_end_session_id = None
        self.set_error(_error_message(error, default="Session end failed."))
        self.emit_changed()

    def _on_session_control_finished(self, handle_key: str) -> None:
        self._inflight_session_controls.pop(handle_key, None)

    def _sync_encounters_model(self) -> None:
        self._encounters_model.set_rows(self._current_session_encounters())

    def _derive_ttv_state(self) -> str:
        adb_row = self._matching_health_row(_ADB_HEALTH_KEYS)
        if adb_row is not None and ui_status_for_health(adb_row) is not UiStatusKind.OK:
            return WAITING_FOR_DEVICE
        if self.session() is None:
            return WAITING_FOR_DEVICE
        if not self.operator_ready_for_submit():
            return WAITING_FOR_FACE
        return READY

    def _transition_ttv_state(self) -> None:
        next_state = self._derive_ttv_state()
        if next_state == self._ttv_state:
            return
        self._ttv_state = next_state
        self.state_changed.emit(next_state)

    def _record_timeline_marker(self, ctx: StimulusUiContext) -> None:
        if ctx.state != StimulusActionState.SUBMITTING or ctx.client_action_id is None:
            return
        if ctx.client_action_id == self._last_marker_action_id:
            return
        self._last_marker_action_id = ctx.client_action_id
        timestamp = ctx.submitted_at_utc or datetime.now(UTC)
        self._timeline_markers.append(
            SmileTimelinePoint(
                timestamp_utc=timestamp,
                label="Stimulus requested",
                marker="Stimulus requested",
            )
        )

    def _face_tracking_detail(self) -> str | None:
        session = self.session()
        if session is None:
            return None
        status, detail = format_calibration_status(session)
        if status is not UiStatusKind.PROGRESS:
            return None
        return detail

    def _matching_health_row(
        self,
        keys: frozenset[str],
    ) -> HealthSubsystemStatus | None:
        snapshot = self._store.health()
        if snapshot is None:
            return None
        for row in snapshot.subsystems:
            if row.subsystem_key.lower() in keys:
                return row
        return None

    def _current_session_encounters(
        self,
        rows: list[EncounterSummary] | None = None,
    ) -> list[EncounterSummary]:
        selected_session_id = self._store.selected_session_id()
        candidates = self._store.encounters() if rows is None else list(rows)
        if selected_session_id is None:
            return candidates
        return [row for row in candidates if row.session_id == selected_session_id]

    def _reconcile_session_lifecycle_waiters(self) -> bool:
        changed = False
        selected_session_id = self._store.selected_session_id()
        session = self.session()

        start_selection_moved = (
            self._pending_start_session_id is not None
            and selected_session_id is not None
            and selected_session_id != self._pending_start_session_id
        )
        start_readback_arrived = (
            self._pending_start_session_id is not None
            and session is not None
            and session.session_id == self._pending_start_session_id
        )
        if start_selection_moved or start_readback_arrived:
            self._pending_start_session_id = None
            changed = True

        end_selection_moved = (
            self._pending_end_session_id is not None
            and selected_session_id is not None
            and selected_session_id != self._pending_end_session_id
        )
        end_readback_arrived = (
            self._pending_end_session_id is not None
            and session is not None
            and session.session_id == self._pending_end_session_id
            and session.ended_at_utc is not None
        )
        if end_selection_moved or end_readback_arrived:
            self._pending_end_session_id = None
            changed = True

        return changed


# ----------------------------------------------------------------------
# Helpers — kept module-level so the core class stays under one screen
# ----------------------------------------------------------------------


def _plain_status_label(prefix: str, row: HealthSubsystemStatus) -> str:
    if row.state.value == "degraded" and prefix.endswith("results"):
        return f"{prefix} need attention"
    mapping = {
        "ok": "healthy",
        "degraded": "needs attention",
        "recovering": "getting ready",
        "error": "needs operator action",
        "unknown": "status unknown",
    }
    return f"{prefix} {mapping[row.state.value]}"


def _synthesized_stream_url() -> str:
    return "android-device://connected-phone/tiktok-live"


def _latest_encounter_for_stimulus(
    encounters: list[EncounterSummary],
    ctx: StimulusUiContext,
) -> EncounterSummary | None:
    """Return the most recent encounter for the current stimulus context."""
    if not encounters:
        return None
    with_stim = [e for e in encounters if _encounter_matches_stimulus_context(e, ctx)]
    if not with_stim:
        return None
    return max(with_stim, key=lambda e: e.segment_timestamp_utc)


def _encounter_matches_stimulus_context(
    encounter: EncounterSummary,
    ctx: StimulusUiContext,
) -> bool:
    stimulus_time = encounter.stimulus_time_utc
    if stimulus_time is None:
        return False
    if ctx.authoritative_stimulus_time_utc is not None:
        earliest = ctx.authoritative_stimulus_time_utc - timedelta(
            seconds=_STIMULUS_READBACK_TOLERANCE_S
        )
        if stimulus_time >= earliest:
            return True
        if not _encounter_is_terminal_for_stimulus(encounter):
            return False
        latest_expected_result = ctx.authoritative_stimulus_time_utc + timedelta(
            seconds=_STIMULUS_TERMINAL_READBACK_GRACE_S
        )
        return encounter.segment_timestamp_utc >= earliest and (
            encounter.segment_timestamp_utc <= latest_expected_result
        )
    if ctx.submitted_at_utc is not None:
        earliest = ctx.submitted_at_utc - timedelta(seconds=_STIMULUS_READBACK_TOLERANCE_S)
        return stimulus_time >= earliest
    return True


def _encounter_is_terminal_for_stimulus(encounter: EncounterSummary) -> bool:
    return encounter.state in {
        EncounterState.COMPLETED,
        EncounterState.REJECTED_GATE_CLOSED,
        EncounterState.REJECTED_NO_FRAMES,
    }


def _latest_completed_encounter(
    encounters: list[EncounterSummary],
) -> EncounterSummary | None:
    completed = [e for e in encounters if _encounter_is_terminal_for_stimulus(e)]
    if not completed:
        return None
    return max(completed, key=lambda e: e.segment_timestamp_utc)


def _latest_encounter(encounters: list[EncounterSummary]) -> EncounterSummary | None:
    if not encounters:
        return None
    return max(encounters, key=lambda e: e.segment_timestamp_utc)


def _smile_intensity_percent(value: float | None) -> int | None:
    if value is None:
        return None
    scaled = value * 100.0 if 0.0 <= value <= 1.0 else value
    return round(min(max(scaled, 0.0), 100.0))


def _error_message(error: object, *, default: str) -> str:
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message
    text = str(error)
    return text or default
