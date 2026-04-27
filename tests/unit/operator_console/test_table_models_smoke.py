"""Smoke tests for Experiments / Alerts / Sessions table models — Phase 7.

Encounters and Health have dedicated regression suites; these smoke
tests cover the remaining three models' basic shape, set/append
semantics, and a couple of display-cell spot checks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from PySide6.QtCore import Qt

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    SessionSummary,
)
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.table_models.sessions_table_model import (
    SessionsTableModel,
)

pytestmark = pytest.mark.usefixtures("qt_app")


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------


def _arm(
    arm_id: str,
    *,
    alpha: float = 3.0,
    beta: float = 5.0,
    enabled: bool = True,
) -> ArmSummary:
    return ArmSummary(
        arm_id=arm_id,
        greeting_text=f"greeting {arm_id}",
        posterior_alpha=alpha,
        posterior_beta=beta,
        evaluation_variance=0.01,
        selection_count=7,
        recent_reward_mean=0.33,
        recent_semantic_pass_rate=0.5,
        enabled=enabled,
    )


def test_experiments_model_shape_and_lookup() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a"), _arm("b")])
    assert model.rowCount() == 2
    assert model.columnCount() == len(ExperimentsTableModel.COLUMNS)
    assert model.arm_by_id("a") is not None
    assert model.arm_by_id("missing") is None


def test_experiments_model_surfaces_alpha_beta() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a", alpha=3.0, beta=5.0)])
    # Column 3 = posterior α, Column 4 = posterior β (column 2 is the disable toggle).
    alpha_text = model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole)
    beta_text = model.data(model.index(0, 4), Qt.ItemDataRole.DisplayRole)
    assert "3" in alpha_text
    assert "5" in beta_text


def test_experiments_model_emits_greeting_edit_without_mutating_row() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a")])
    emissions: list[tuple[str, str]] = []
    model.greeting_edit_requested.connect(lambda *args: emissions.append(tuple(args)))

    assert model.setData(model.index(0, 1), "new greeting", Qt.ItemDataRole.EditRole) is True
    assert emissions == [("a", "new greeting")]
    # The row waits for store/coordinator readback before changing.
    assert model.row_at(0).greeting_text == "greeting a"  # type: ignore[union-attr]


def test_experiments_model_allows_disabled_arm_greeting_rename_intent() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a", enabled=False)])
    emissions: list[tuple[str, str]] = []
    model.greeting_edit_requested.connect(lambda *args: emissions.append(tuple(args)))

    assert bool(model.flags(model.index(0, 1)) & Qt.ItemFlag.ItemIsEditable)
    assert model.setData(model.index(0, 1), "archived greeting", Qt.ItemDataRole.EditRole)
    assert emissions == [("a", "archived greeting")]


def test_experiments_model_disable_is_one_way_only() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a")])
    disabled: list[str] = []
    model.disable_requested.connect(disabled.append)

    assert (
        model.setData(model.index(0, 2), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        is True
    )
    assert disabled == ["a"]
    assert (
        model.setData(model.index(0, 2), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
        is False
    )
    assert disabled == ["a"]


def test_experiments_model_posterior_columns_are_not_editable() -> None:
    model = ExperimentsTableModel()
    model.set_rows([_arm("a")])
    flags = model.flags(model.index(0, 3))
    assert not bool(flags & Qt.ItemFlag.ItemIsEditable)


# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------


def _alert(alert_id: str, *, severity: AlertSeverity = AlertSeverity.WARNING) -> AlertEvent:
    return AlertEvent(
        alert_id=alert_id,
        severity=severity,
        kind=AlertKind.SUBSYSTEM_DEGRADED,
        message=f"message {alert_id}",
        subsystem_key="ffmpeg",
        emitted_at_utc=datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC),
    )


def test_alerts_model_set_rows_replaces_list() -> None:
    model = AlertsTableModel()
    model.set_rows([_alert("a"), _alert("b")])
    assert model.rowCount() == 2
    model.set_rows([_alert("c")])
    assert model.rowCount() == 1


def test_alerts_model_append_rows_dedups_by_alert_id() -> None:
    model = AlertsTableModel()
    model.set_rows([_alert("a")])
    model.append_rows([_alert("a"), _alert("b")])
    # "a" was already present — append_rows must skip it.
    assert model.rowCount() == 2
    # Newest row rendered first so the attention queue reads top-down.
    assert model.row_at(0) is not None
    first = model.row_at(0)
    assert first is not None
    assert first.alert_id == "b"


def test_alerts_model_renders_severity_label() -> None:
    model = AlertsTableModel()
    model.set_rows([_alert("a", severity=AlertSeverity.CRITICAL)])
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "critical"


# ---------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------


def _session(
    *,
    status: str = "active",
    reward: float | None = 0.42,
    duration: float | None = 600.0,
) -> SessionSummary:
    return SessionSummary(
        session_id=uuid4(),
        status=status,
        started_at_utc=datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC),
        duration_s=duration,
        active_arm="greeting_v1",
        experiment_id="greeting_line_v1",
        latest_reward=reward,
    )


def test_sessions_model_shape() -> None:
    model = SessionsTableModel()
    model.set_rows([_session(), _session()])
    assert model.rowCount() == 2
    assert model.columnCount() == len(SessionsTableModel.COLUMNS)


def test_sessions_model_renders_status_and_duration() -> None:
    model = SessionsTableModel()
    model.set_rows([_session(status="completed", duration=125.0)])
    status = model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole)
    duration = model.data(model.index(0, 5), Qt.ItemDataRole.DisplayRole)
    assert status == "completed"
    assert "2m" in duration
