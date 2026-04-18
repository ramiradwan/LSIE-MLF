"""Regression tests for Phase 8 viewmodels.

The VMs are the operator-trust layer: reward-explanation wording,
null-valid co-modulation rendering, the active-arm readback, and the
stimulus-countdown arithmetic all live here. Locking these down in
tests guards against silent drift when the DTOs or formatters evolve.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
)
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
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm=active_arm,
        expected_greeting=expected,
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
