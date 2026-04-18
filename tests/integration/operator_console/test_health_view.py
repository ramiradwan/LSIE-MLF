"""Integration: Health page — degraded vs recovering vs error distinction.

§12 draws three lines the operator must never lose sight of:

  * DEGRADED — impaired but stable (WARN pill)
  * RECOVERING — self-healing in flight (PROGRESS pill)
  * ERROR — hard-down, requires operator action (ERROR pill)

A single integration assertion seeds all three at once and verifies the
summary cards render with three distinct `UiStatusKind` values, not
collapsed into a generic "something's wrong" state.

Spec references:
  §12            — degraded / recovering / error distinction
  §4.E.1         — Health operator surface
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.schemas.operator_console import (
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    UiStatusKind,
)
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.viewmodels.health_vm import HealthViewModel
from services.operator_console.views.health_view import HealthView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def test_health_view_renders_degraded_recovering_error_as_three_states() -> None:
    store = OperatorStore()
    health_model = HealthTableModel()
    alerts_model = AlertsTableModel()
    vm = HealthViewModel(store, health_model, alerts_model)
    view = HealthView(vm)

    store.set_health(
        HealthSnapshot(
            generated_at_utc=_NOW,
            # Overall goes to the worst state of the three seeded rows,
            # which is ERROR. The per-state counters below are what
            # matters for the card-distinction assertion.
            overall_state=HealthState.ERROR,
            degraded_count=1,
            recovering_count=1,
            error_count=1,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="capture",
                    label="Capture",
                    state=HealthState.DEGRADED,
                    detail="ADB drift > 1.2s",
                ),
                HealthSubsystemStatus(
                    subsystem_key="orchestrator",
                    label="Orchestrator",
                    state=HealthState.RECOVERING,
                    detail="FFmpeg restart in flight",
                    recovery_mode="ffmpeg-restart",
                ),
                HealthSubsystemStatus(
                    subsystem_key="worker",
                    label="Worker",
                    state=HealthState.ERROR,
                    detail="GPU not available",
                    operator_action_hint="restart worker container",
                ),
            ],
        )
    )

    # Three distinct pill kinds — the §12 line the operator must see.
    kinds = {
        view._degraded_card._status._kind,  # type: ignore[attr-defined]
        view._recovering_card._status._kind,  # type: ignore[attr-defined]
        view._error_card._status._kind,  # type: ignore[attr-defined]
    }
    assert kinds == {UiStatusKind.WARN, UiStatusKind.PROGRESS, UiStatusKind.ERROR}

    # And the counter cards carry their seeded counts.
    assert view._degraded_card._primary.text() == "1"  # type: ignore[attr-defined]
    assert view._recovering_card._primary.text() == "1"  # type: ignore[attr-defined]
    assert view._error_card._primary.text() == "1"  # type: ignore[attr-defined]

    # The error card's secondary text surfaces the first error row's
    # operator-action hint — the operator shouldn't have to scroll the
    # subsystem table below to find what to do.
    assert (
        "restart" in view._error_card._secondary.text()  # type: ignore[attr-defined]
    )
