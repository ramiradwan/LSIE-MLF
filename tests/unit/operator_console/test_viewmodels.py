"""Regression tests for Phase 8 viewmodels.

The VMs are the operator-trust layer: reward-explanation wording,
null-valid co-modulation rendering, the active-arm readback, and the
stimulus-countdown arithmetic all live here. Locking these down in
tests guards against silent drift when the DTOs or formatters evolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from PySide6.QtCore import Qt

from packages.schemas.experiments import ExperimentArmCreateRequest, ExperimentCreateRequest
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
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
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
    UiStatusKind,
)
from services.operator_console.api_client import ApiError
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.viewmodels.health_vm import HealthViewModel
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.viewmodels.overview_vm import OverviewViewModel
from services.operator_console.viewmodels.physiology_vm import PhysiologyViewModel
from services.operator_console.workers import OneShotSignals

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------


def _session(
    session_id: UUID | None = None,
    *,
    active_arm: str | None = "greeting_v1",
    expected: str | None = "hei rakas",
    is_calibrating: bool | None = None,
    calibration_frames_accumulated: int | None = None,
    calibration_frames_required: int | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm=active_arm,
        expected_greeting=expected,
        is_calibrating=is_calibrating,
        calibration_frames_accumulated=calibration_frames_accumulated,
        calibration_frames_required=calibration_frames_required,
    )


def _encounter(
    encounter_id: str,
    *,
    state: EncounterState = EncounterState.COMPLETED,
    semantic_gate: int | None = 1,
    semantic_confidence: float | None = 0.9,
    p90: float | None = 0.42,
    gated_reward: float | None = 0.42,
    frames: int | None = 150,
    stimulus_time: datetime | None = None,
    session_id: UUID | None = None,
) -> EncounterSummary:
    return EncounterSummary(
        encounter_id=encounter_id,
        session_id=session_id or uuid4(),
        segment_timestamp_utc=_NOW,
        state=state,
        active_arm="greeting_v1",
        expected_greeting="hei rakas",
        stimulus_time_utc=stimulus_time,
        semantic_gate=semantic_gate,
        semantic_confidence=semantic_confidence,
        p90_intensity=p90,
        gated_reward=gated_reward,
        n_frames_in_window=frames,
    )


@dataclass
class _FakeSessionStartDispatcher:
    calls: list[SessionCreateRequest] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, request: SessionCreateRequest) -> OneShotSignals:
        self.calls.append(request)
        bus = OneShotSignals()
        self.signals.append(bus)
        return bus


@dataclass
class _FakeSessionEndCall:
    session_id: UUID
    request: SessionEndRequest


@dataclass
class _FakeSessionEndDispatcher:
    calls: list[_FakeSessionEndCall] = field(default_factory=list)
    signals: list[OneShotSignals] = field(default_factory=list)

    def __call__(self, session_id: UUID, request: SessionEndRequest) -> OneShotSignals:
        self.calls.append(_FakeSessionEndCall(session_id=session_id, request=request))
        bus = OneShotSignals()
        self.signals.append(bus)
        return bus


# ---------------------------------------------------------------------
# OverviewViewModel
# ---------------------------------------------------------------------


def test_overview_vm_returns_none_when_store_empty() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    assert vm.snapshot() is None
    assert vm.active_session() is None
    assert vm.latest_encounter() is None
    assert vm.experiment_summary() is None
    assert vm.health_summary() is None
    assert vm.alerts() == []


def test_overview_vm_surfaces_snapshot_fields() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    change_hits = 0

    def on_change() -> None:
        nonlocal change_hits
        change_hits += 1

    vm.changed.connect(on_change)

    session = _session()
    snap = OverviewSnapshot(
        generated_at_utc=_NOW,
        active_session=session,
        latest_encounter=LatestEncounterSummary(
            encounter_id="e1",
            session_id=session.session_id,
            segment_timestamp_utc=_NOW,
            state=EncounterState.COMPLETED,
            p90_intensity=0.5,
            gated_reward=0.5,
            semantic_gate=1,
            n_frames_in_window=120,
        ),
    )
    store.set_overview(snap)
    assert vm.active_session() is session
    assert vm.latest_encounter() is not None
    assert change_hits == 1


def test_overview_vm_prefers_dedicated_health_slice() -> None:
    # The dedicated health poll is canonically fresher than the
    # snapshot-embedded copy.
    store = OperatorStore()
    vm = OverviewViewModel(store)
    fresh = HealthSnapshot(
        generated_at_utc=_NOW,
        overall_state=HealthState.OK,
    )
    store.set_health(fresh)
    assert vm.health_summary() is fresh


def test_overview_vm_formats_latest_encounter_semantic_attribution_diagnostics() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    session = _session()
    store.set_overview(
        OverviewSnapshot(
            generated_at_utc=_NOW,
            latest_encounter=LatestEncounterSummary(
                encounter_id="e-diagnostics",
                session_id=session.session_id,
                segment_timestamp_utc=_NOW,
                state=EncounterState.COMPLETED,
                semantic_evaluation=SemanticEvaluationSummary(
                    reasoning="cross_encoder_high_nonmatch",
                    is_match=False,
                    confidence_score=0.08,
                    semantic_method="cross_encoder",
                    semantic_method_version="ce-v1",
                ),
                attribution=AttributionSummary(finality="offline_final"),
            ),
        )
    )

    display = vm.latest_encounter_semantic_attribution_diagnostics()
    assert display.semantic_method == "local cross-encoder · ce-v1"
    assert display.bounded_reason_code == "Cross-encoder high-confidence non-match"
    assert display.attribution_finality == "offline final"


# ---------------------------------------------------------------------
# LiveSessionViewModel
# ---------------------------------------------------------------------


def test_live_session_vm_surfaces_active_arm_from_live_session() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    # Arm must come from live_session DTO, never from table rows.
    store.set_live_session(_session(active_arm="greeting_v7"))
    assert vm.active_arm() == "greeting_v7"
    assert vm.expected_greeting() == "hei rakas"


def test_live_session_vm_surfaces_calibration_status_from_live_session() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_live_session(
        _session(
            is_calibrating=True,
            calibration_frames_accumulated=12,
            calibration_frames_required=45,
        )
    )
    kind, text = vm.calibration_status()
    assert vm.is_calibrating() is True
    assert vm.operator_ready_for_submit() is False
    assert kind is UiStatusKind.PROGRESS
    assert text == "Calibrating · 12/45 frames"


def test_live_session_vm_ready_at_threshold_preserves_authoritative_calibrating_flag() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_live_session(
        _session(
            is_calibrating=True,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
        )
    )
    kind, text = vm.calibration_status()
    assert vm.is_calibrating() is True
    assert vm.operator_ready_for_submit() is True
    assert kind is UiStatusKind.OK
    assert text == "Ready"


def test_live_session_vm_false_and_none_calibration_are_submit_ready() -> None:
    for is_calibrating in (False, None):
        store = OperatorStore()
        model = EncountersTableModel()
        vm = LiveSessionViewModel(store, model)
        store.set_live_session(_session(is_calibrating=is_calibrating))

        assert vm.is_calibrating() is False
        assert vm.operator_ready_for_submit() is True
        assert vm.calibration_status() == (UiStatusKind.OK, "Ready")


def test_live_session_vm_reward_explanation_for_gate_closed() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_encounters(
        [
            _encounter(
                "e1",
                state=EncounterState.COMPLETED,
                semantic_gate=0,
                semantic_confidence=0.81,
                p90=0.6,
                gated_reward=0.0,
                frames=150,
            )
        ]
    )
    vm.select_encounter("e1")
    text = vm.reward_explanation()
    assert "Semantic gate closed" in text
    assert "reward suppressed" in text


def test_live_session_vm_reward_explanation_for_zero_frames() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_encounters(
        [
            _encounter(
                "e1",
                state=EncounterState.REJECTED_NO_FRAMES,
                semantic_gate=1,
                p90=None,
                gated_reward=None,
                frames=0,
            )
        ]
    )
    vm.select_encounter("e1")
    text = vm.reward_explanation()
    assert "No valid AU12 frames" in text


def test_live_session_vm_reward_explanation_without_encounters() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    assert vm.reward_explanation() == "No completed encounter yet."


def test_live_session_vm_semantic_attribution_diagnostics_ride_encounters_cache() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_encounters(
        [
            _encounter("e1").model_copy(
                update={
                    "semantic_evaluation": SemanticEvaluationSummary(
                        reasoning="gray_band_llm_match",
                        is_match=True,
                        confidence_score=0.68,
                        semantic_method="llm_gray_band",
                        semantic_method_version="gray-v1",
                    ),
                    "attribution": AttributionSummary(
                        finality="online_provisional",
                        soft_reward_candidate=0.22,
                        au12_lift_p90=0.40,
                    ),
                }
            )
        ]
    )
    vm.select_encounter("e1")

    display = vm.semantic_attribution_diagnostics()
    assert display.semantic_method == "LLM gray-band fallback · gray-v1"
    assert display.bounded_reason_code == "Gray-band fallback match"
    assert display.soft_reward_candidate == "r_t^soft 0.220"
    assert display.au12_lift_metrics == "P90 lift 0.400"


def test_live_session_vm_set_stimulus_submitting_emits_key() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    key = vm.set_stimulus_submitting("test note")
    ctx = store.stimulus_ui_context()
    assert isinstance(key, UUID)
    assert ctx.state == StimulusActionState.SUBMITTING
    assert ctx.client_action_id == key
    assert ctx.operator_note == "test note"


def test_live_session_vm_apply_stimulus_accepted_true() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    session_id = uuid4()
    key = vm.set_stimulus_submitting(None)
    ack = StimulusAccepted(
        session_id=session_id,
        client_action_id=key,
        accepted=True,
        received_at_utc=_NOW,
        message=None,
    )
    vm.apply_stimulus_accepted(ack)
    ctx = store.stimulus_ui_context()
    assert ctx.state == StimulusActionState.ACCEPTED
    assert ctx.client_action_id == key
    assert ctx.accepted_at_utc == _NOW


def test_live_session_vm_apply_stimulus_accepted_false_is_failed() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    key = vm.set_stimulus_submitting(None)
    ack = StimulusAccepted(
        session_id=uuid4(),
        client_action_id=key,
        accepted=False,
        received_at_utc=_NOW,
        message="session inactive",
    )
    vm.apply_stimulus_accepted(ack)
    ctx = store.stimulus_ui_context()
    assert ctx.state == StimulusActionState.FAILED
    assert ctx.message == "session inactive"


def test_live_session_vm_reconciles_authoritative_stimulus_time() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    key = vm.set_stimulus_submitting(None)
    # Move to ACCEPTED so reconciliation is allowed.
    vm.apply_stimulus_accepted(
        StimulusAccepted(
            session_id=uuid4(),
            client_action_id=key,
            accepted=True,
            received_at_utc=_NOW,
            message=None,
        )
    )
    # Orchestrator publishes an encounter with a stimulus_time — VM
    # must promote the context from ACCEPTED to MEASURING.
    authoritative = _NOW + timedelta(seconds=1)
    store.set_encounters(
        [
            _encounter(
                "e1",
                state=EncounterState.MEASURING,
                stimulus_time=authoritative,
            )
        ]
    )
    # The `encounters_changed` slot triggers reconciliation; no manual
    # call needed.
    ctx = store.stimulus_ui_context()
    assert ctx.state == StimulusActionState.MEASURING
    assert ctx.authoritative_stimulus_time_utc == authoritative


def test_live_session_vm_measurement_window_countdown() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    # No authoritative stimulus time yet → None.
    assert vm.measurement_window_remaining_s(_NOW) is None

    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=_NOW,
        )
    )
    # §7B window default = 30s. 10s in → 20s remaining.
    remaining = vm.measurement_window_remaining_s(_NOW + timedelta(seconds=10))
    assert remaining == pytest.approx(20.0)
    # Past the window → clamps to 0.
    assert vm.measurement_window_remaining_s(_NOW + timedelta(seconds=60)) == 0.0


def test_live_session_vm_selection_dropped_when_row_evicted() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    store.set_encounters([_encounter("e1"), _encounter("e2")])
    vm.select_encounter("e1")
    assert vm.selected_encounter() is not None

    # Replace with a different row set — e1 is gone.
    store.set_encounters([_encounter("e3")])
    assert vm.selected_encounter() is None


def test_live_session_vm_hides_mismatched_live_session_and_rows() -> None:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    selected_session_id = uuid4()
    store.set_selected_session_id(selected_session_id)
    store.set_live_session(_session(uuid4()))
    store.set_encounters([_encounter("other-row", session_id=uuid4())])

    assert vm.session() is None
    assert model.rowCount() == 0


def test_live_session_vm_start_new_session_validates_modal_fields() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())

    assert vm.start_new_session("   ", "exp-a") is None
    assert vm.error() == "Stream URL is required."

    assert vm.start_new_session("rtmp://example/live", "   ") is None
    assert vm.error() == "Experiment ID is required."


def test_live_session_vm_start_new_session_dispatches_and_switches_selection() -> None:
    store = OperatorStore()
    selected_session_id = uuid4()
    store.set_selected_session_id(selected_session_id)
    store.set_live_session(_session(selected_session_id))
    store.set_encounters([_encounter("old-row", session_id=selected_session_id)])
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    start_dispatcher = _FakeSessionStartDispatcher()
    end_dispatcher = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start_dispatcher, end_dispatcher)

    action_id = vm.start_new_session("  rtmp://example/live  ", "  greeting_line_v1  ")

    assert action_id is not None
    assert len(start_dispatcher.calls) == 1
    request = start_dispatcher.calls[0]
    assert request.client_action_id == action_id
    assert request.stream_url == "rtmp://example/live"
    assert request.experiment_id == "greeting_line_v1"
    assert vm.session_start_in_progress() is True

    new_session_id = uuid4()
    start_dispatcher.signals[0].succeeded.emit(
        "session_start",
        SessionLifecycleAccepted(
            action="start",
            session_id=new_session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=_NOW,
        ),
    )

    assert store.selected_session_id() == new_session_id
    assert vm.session() is None  # stale live-session DTO is filtered away
    assert model.rowCount() == 0  # stale encounter rows are filtered away too
    assert vm.session_start_in_progress() is True

    store.set_live_session(_session(new_session_id))
    assert vm.session() is not None
    assert vm.session_start_in_progress() is False


def test_live_session_vm_end_current_session_waits_for_ended_at_readback() -> None:
    store = OperatorStore()
    session_id = uuid4()
    store.set_selected_session_id(session_id)
    store.set_live_session(_session(session_id))
    vm = LiveSessionViewModel(store, EncountersTableModel())
    start_dispatcher = _FakeSessionStartDispatcher()
    end_dispatcher = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start_dispatcher, end_dispatcher)

    action_id = vm.end_current_session()

    assert action_id is not None
    assert len(end_dispatcher.calls) == 1
    call = end_dispatcher.calls[0]
    assert call.session_id == session_id
    assert call.request.client_action_id == action_id
    assert vm.session_end_in_progress() is True
    assert vm.can_end_session() is False

    end_dispatcher.signals[0].succeeded.emit(
        "session_end",
        SessionLifecycleAccepted(
            action="end",
            session_id=session_id,
            client_action_id=call.request.client_action_id,
            accepted=True,
            received_at_utc=_NOW,
        ),
    )

    assert vm.session_end_in_progress() is True
    store.set_live_session(
        _session(session_id).model_copy(
            update={
                "status": "ended",
                "ended_at_utc": _NOW + timedelta(minutes=1),
            }
        )
    )
    assert vm.session_end_in_progress() is False
    assert vm.can_end_session() is False


def test_live_session_vm_session_control_failure_sets_error_and_clears_pending() -> None:
    store = OperatorStore()
    vm = LiveSessionViewModel(store, EncountersTableModel())
    start_dispatcher = _FakeSessionStartDispatcher()
    end_dispatcher = _FakeSessionEndDispatcher()
    vm.bind_session_lifecycle_actions(start_dispatcher, end_dispatcher)

    vm.start_new_session("rtmp://example/live", "greeting_line_v1")
    start_dispatcher.signals[0].failed.emit(
        "session_start",
        ApiError(message="broker unavailable", retryable=True),
    )

    assert vm.error() == "broker unavailable"
    assert vm.session_start_in_progress() is False


# ---------------------------------------------------------------------
# ExperimentsViewModel
# ---------------------------------------------------------------------


def test_experiments_vm_reflects_active_arm_and_arms() -> None:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model)
    detail = ExperimentDetail(
        experiment_id="exp1",
        active_arm_id="a2",
        arms=[
            ArmSummary(arm_id="a1", greeting_text="hi", posterior_alpha=1.0, posterior_beta=1.0),
            ArmSummary(arm_id="a2", greeting_text="hei", posterior_alpha=2.0, posterior_beta=5.0),
        ],
        last_update_summary="arm a2 updated by reward 0.42",
    )
    store.set_experiment(detail)
    assert vm.active_arm_id() == "a2"
    assert vm.latest_update_summary() == "arm a2 updated by reward 0.42"
    assert model.rowCount() == 2


def test_experiments_vm_latest_update_placeholder_when_absent() -> None:
    store = OperatorStore()
    vm = ExperimentsViewModel(store, ExperimentsTableModel())
    assert vm.latest_update_summary() == "No experiment update yet."


def test_experiments_vm_create_emits_typed_request() -> None:
    store = OperatorStore()
    vm = ExperimentsViewModel(store, ExperimentsTableModel(), default_experiment_id="exp-default")
    emissions: list[ExperimentCreateRequest] = []
    vm.create_experiment_requested.connect(emissions.append)

    assert vm.create_experiment("exp-new", "Greeting v2", "arm-a", "Hei") is True
    assert len(emissions) == 1
    assert emissions[0].experiment_id == "exp-new"
    assert emissions[0].arms[0].arm == "arm-a"
    assert vm.error() == ""


def test_experiments_vm_add_arm_requires_loaded_experiment() -> None:
    store = OperatorStore()
    vm = ExperimentsViewModel(store, ExperimentsTableModel(), default_experiment_id="exp-default")
    emissions: list[tuple[str, ExperimentArmCreateRequest]] = []
    vm.add_arm_requested.connect(lambda *args: emissions.append(tuple(args)))

    assert vm.add_arm("arm-b", "Moi") is False
    assert emissions == []
    assert "Load or create" in vm.error()


def test_experiments_vm_uses_managed_experiment_id_when_present() -> None:
    store = OperatorStore()
    vm = ExperimentsViewModel(store, ExperimentsTableModel(), default_experiment_id="exp-default")
    assert vm.current_experiment_id() == "exp-default"
    store.set_managed_experiment_id("exp-managed")
    assert vm.current_experiment_id() == "exp-managed"


def test_experiments_vm_add_arm_emits_for_current_experiment() -> None:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model)
    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp1",
            arms=[
                ArmSummary(
                    arm_id="a1",
                    greeting_text="hi",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                )
            ],
        )
    )
    emissions: list[tuple[str, ExperimentArmCreateRequest]] = []
    vm.add_arm_requested.connect(lambda *args: emissions.append(tuple(args)))

    assert vm.add_arm("a2", "hei") is True
    assert emissions[0][0] == "exp1"
    assert emissions[0][1].arm == "a2"


def test_experiments_vm_table_rename_and_disable_emit_safe_commands() -> None:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model)
    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp1",
            arms=[
                ArmSummary(
                    arm_id="a1",
                    greeting_text="hi",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                )
            ],
        )
    )
    renames: list[tuple[str, str, str]] = []
    disables: list[tuple[str, str]] = []
    vm.rename_arm_requested.connect(lambda *args: renames.append(tuple(args)))
    vm.disable_arm_requested.connect(lambda *args: disables.append(tuple(args)))

    assert model.setData(model.index(0, 1), "hei ystävä") is True
    assert (
        model.setData(
            model.index(0, 2),
            Qt.CheckState.Unchecked,
            Qt.ItemDataRole.CheckStateRole,
        )
        is True
    )
    assert renames == [("exp1", "a1", "hei ystävä")]
    assert disables == [("exp1", "a1")]


def test_experiments_vm_allows_disabled_arm_greeting_rename() -> None:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model)
    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp1",
            arms=[
                ArmSummary(
                    arm_id="archived",
                    greeting_text="old",
                    posterior_alpha=1.0,
                    posterior_beta=1.0,
                    enabled=False,
                )
            ],
        )
    )
    renames: list[tuple[str, str, str]] = []
    vm.rename_arm_requested.connect(lambda *args: renames.append(tuple(args)))

    assert vm.rename_arm_greeting("archived", "historical label") is True
    assert renames == [("exp1", "archived", "historical label")]


# ---------------------------------------------------------------------
# PhysiologyViewModel
# ---------------------------------------------------------------------


def test_physiology_vm_comodulation_null_reads_as_legitimate() -> None:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    snap = SessionPhysiologySnapshot(
        session_id=uuid4(),
        comodulation=CoModulationSummary(
            session_id=uuid4(),
            co_modulation_index=None,
            null_reason="insufficient aligned pairs",
            coverage_ratio=0.1,
        ),
        generated_at_utc=_NOW,
    )
    store.set_physiology(snap)
    explanation = vm.comodulation_explanation()
    assert "null" in explanation.lower()
    assert "insufficient aligned pairs" in explanation


def test_physiology_vm_distinguishes_stale_from_absent() -> None:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    snap = SessionPhysiologySnapshot(
        session_id=uuid4(),
        streamer=PhysiologyCurrentSnapshot(
            subject_role="streamer",
            rmssd_ms=45.0,
            heart_rate_bpm=80,
            is_stale=True,
            freshness_s=120.0,
        ),
        operator=None,
        generated_at_utc=_NOW,
    )
    store.set_physiology(snap)
    assert vm.streamer_snapshot() is not None
    assert vm.streamer_snapshot().is_stale is True  # type: ignore[union-attr]
    assert vm.operator_snapshot() is None


def test_physiology_vm_comodulation_explanation_without_data() -> None:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    assert vm.comodulation_explanation() == "No co-modulation window yet."


# ---------------------------------------------------------------------
# HealthViewModel
# ---------------------------------------------------------------------


def test_health_vm_surfaces_probe_rows_without_touching_alerts() -> None:
    store = OperatorStore()
    vm = HealthViewModel(store, HealthTableModel(), AlertsTableModel())
    probe = HealthSubsystemProbe(
        subsystem_key="redis",
        label="Redis Broker",
        state=HealthProbeState.OK,
        latency_ms=2.0,
        checked_at_utc=_NOW,
    )
    store.set_health(
        HealthSnapshot(
            generated_at_utc=_NOW,
            overall_state=HealthState.OK,
            subsystem_probes={probe.subsystem_key: probe},
        )
    )
    assert vm.subsystem_probes() == [probe]
    assert vm.alerts_model().rowCount() == 0


def test_health_vm_orders_keyed_probe_rows_locally() -> None:
    store = OperatorStore()
    vm = HealthViewModel(store, HealthTableModel(), AlertsTableModel())
    redis_probe = HealthSubsystemProbe(
        subsystem_key="redis",
        label="Redis Broker",
        state=HealthProbeState.OK,
        latency_ms=2.0,
        checked_at_utc=_NOW,
    )
    postgres_probe = HealthSubsystemProbe(
        subsystem_key="postgres",
        label="Postgres",
        state=HealthProbeState.OK,
        latency_ms=1.0,
        checked_at_utc=_NOW,
    )
    custom_probe = HealthSubsystemProbe(
        subsystem_key="zz_custom",
        label="Custom",
        state=HealthProbeState.UNKNOWN,
        checked_at_utc=_NOW,
    )
    store.set_health(
        HealthSnapshot(
            generated_at_utc=_NOW,
            overall_state=HealthState.OK,
            subsystem_probes={
                redis_probe.subsystem_key: redis_probe,
                custom_probe.subsystem_key: custom_probe,
                postgres_probe.subsystem_key: postgres_probe,
            },
        )
    )

    assert [probe.subsystem_key for probe in vm.subsystem_probes()] == [
        "postgres",
        "redis",
        "zz_custom",
    ]


def test_health_vm_degraded_count_sums_degraded_and_recovering() -> None:
    store = OperatorStore()
    vm = HealthViewModel(store, HealthTableModel(), AlertsTableModel())
    snap = HealthSnapshot(
        generated_at_utc=_NOW,
        overall_state=HealthState.DEGRADED,
        subsystems=[
            HealthSubsystemStatus(subsystem_key="a", label="A", state=HealthState.DEGRADED),
            HealthSubsystemStatus(
                subsystem_key="b",
                label="B",
                state=HealthState.RECOVERING,
                recovery_mode="retrying",
            ),
            HealthSubsystemStatus(subsystem_key="c", label="C", state=HealthState.OK),
        ],
        degraded_count=1,
        recovering_count=1,
        error_count=0,
    )
    store.set_health(snap)
    # Badge = degraded + recovering per §12 (both "needs attention").
    assert vm.degraded_count() == 2


def test_health_vm_sync_alerts_model() -> None:
    store = OperatorStore()
    alerts = AlertsTableModel()
    vm = HealthViewModel(store, HealthTableModel(), alerts)
    assert vm.snapshot() is None  # baseline
    store.set_alerts(
        [
            AlertEvent(
                alert_id="a1",
                severity=AlertSeverity.WARNING,
                kind=AlertKind.SUBSYSTEM_DEGRADED,
                message="msg",
                emitted_at_utc=_NOW,
            )
        ]
    )
    assert alerts.rowCount() == 1
