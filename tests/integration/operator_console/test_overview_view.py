"""Integration: Overview page renders from a seeded store.

Wires the full Phase 8 viewmodel → Phase 9 view chain against a real
`OperatorStore` and asserts that a single `store.set_overview(...)`
lands the active session id, arm, experiment label, and health state in
the six overview cards the operator reads first.

Spec references:
  §4.E.1         — Overview is the operator's at-a-glance surface
  §7B            — the reward-explanation fields travel through the
                   latest-encounter card, not through inline widgets
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import (
    EncounterState,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
)
from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.overview_vm import OverviewViewModel
from services.operator_console.views.overview_view import OverviewView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _seed(store: OperatorStore) -> UUID:
    session_id = uuid4()
    store.set_overview(
        OverviewSnapshot(
            generated_at_utc=_NOW,
            active_session=SessionSummary(
                session_id=session_id,
                status="active",
                started_at_utc=_NOW,
                active_arm="greeting_v2",
                expected_greeting="hei rakas",
                duration_s=180.0,
            ),
            latest_encounter=LatestEncounterSummary(
                encounter_id="e-7",
                session_id=session_id,
                segment_timestamp_utc=_NOW,
                state=EncounterState.COMPLETED,
                semantic_gate=1,
                p90_intensity=0.6,
                gated_reward=0.6,
                n_frames_in_window=150,
            ),
            experiment_summary=ExperimentSummary(
                experiment_id="exp-42",
                label="greeting line v2",
                active_arm_id="greeting_v2",
                arm_count=3,
                latest_reward=0.55,
            ),
            physiology=SessionPhysiologySnapshot(
                session_id=session_id,
                streamer=PhysiologyCurrentSnapshot(
                    subject_role="streamer",
                    rmssd_ms=45.0,
                    heart_rate_bpm=72,
                    is_stale=False,
                    freshness_s=4.0,
                ),
                generated_at_utc=_NOW,
            ),
            health=HealthSnapshot(
                generated_at_utc=_NOW,
                overall_state=HealthState.OK,
                subsystems=[
                    HealthSubsystemStatus(
                        subsystem_key="orchestrator",
                        label="Orchestrator",
                        state=HealthState.OK,
                    ),
                ],
            ),
        )
    )
    return session_id


def test_overview_view_renders_seeded_store_end_to_end() -> None:
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)

    session_id = _seed(store)

    # Active session card shows a compact id while preserving the full id in detail.
    primary = view._active_session_card._primary.text()  # type: ignore[attr-defined]
    secondary = view._active_session_card._secondary.text()  # type: ignore[attr-defined]
    assert str(session_id) not in primary
    assert primary.startswith(str(session_id)[:8])
    assert primary.endswith(str(session_id)[-4:])
    assert str(session_id) in secondary
    assert "greeting_v2" in secondary
    # Experiment card binds to the experiment label.
    assert "greeting line v2" in view._experiment_card._primary.text()  # type: ignore[attr-defined]
    # Health card binds to the overall OK state.
    assert "ok" in view._health_card._primary.text()  # type: ignore[attr-defined]
    # Latest encounter card surfaces the §7B gated reward.
    assert "reward" in view._latest_encounter_card._primary.text()  # type: ignore[attr-defined]
    # Physiology card reads live when an RMSSD value is present.
    assert view._physiology_card._primary.text() == "Live heart data"  # type: ignore[attr-defined]


def test_overview_view_active_session_click_round_trips_session_id() -> None:
    # The shell wires `session_activated(UUID)` to a navigation slot;
    # the integration assertion is that clicking the card emits the
    # exact session id the store was seeded with.
    store = OperatorStore()
    vm = OverviewViewModel(store)
    view = OverviewView(vm)
    session_id = _seed(store)

    received: list[UUID] = []
    view.session_activated.connect(received.append)
    view._on_active_session_clicked()  # type: ignore[attr-defined]
    assert received == [session_id]
