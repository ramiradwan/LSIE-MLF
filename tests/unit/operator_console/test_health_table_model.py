"""Regression tests for `HealthTableModel` — Phase 7.

§12 distinguishes degraded-but-recovering states from errors. The
Health page reads those states off this model, so we lock in that the
`recovery_mode` column exists and renders the provided mode verbatim
for RECOVERING/DEGRADED rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PySide6.QtCore import Qt

from packages.schemas.operator_console import HealthState, HealthSubsystemStatus
from services.operator_console.table_models.health_table_model import HealthTableModel

pytestmark = pytest.mark.usefixtures("qt_app")


def _row(
    *,
    key: str = "adb_capture",
    label: str = "ADB capture",
    state: HealthState = HealthState.OK,
    recovery_mode: str | None = None,
    detail: str | None = None,
    hint: str | None = None,
) -> HealthSubsystemStatus:
    return HealthSubsystemStatus(
        subsystem_key=key,
        label=label,
        state=state,
        last_success_utc=datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC),
        detail=detail,
        recovery_mode=recovery_mode,
        operator_action_hint=hint,
    )


# ---------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------


def test_column_count_covers_recovery_mode_column() -> None:
    model = HealthTableModel()
    assert model.columnCount() == len(HealthTableModel.COLUMNS)
    # Recovery mode column is mandatory — the spec requires degraded /
    # recovering to read distinct from error.
    assert "Recovery mode" in HealthTableModel.COLUMNS


def test_rows_after_set() -> None:
    model = HealthTableModel()
    model.set_rows([_row(key="a"), _row(key="b")])
    assert model.rowCount() == 2


# ---------------------------------------------------------------------
# §12 recovery semantics
# ---------------------------------------------------------------------


def test_recovering_row_renders_mode_distinct_from_error() -> None:
    model = HealthTableModel()
    model.set_rows(
        [
            _row(
                state=HealthState.RECOVERING,
                recovery_mode="ffmpeg_restarting",
                detail="resample pipe restarted",
                hint="wait 10s",
            )
        ]
    )
    state_idx = model.index(0, 1)
    recovery_idx = model.index(0, 2)
    assert model.data(state_idx, Qt.ItemDataRole.DisplayRole) == "recovering"
    assert model.data(recovery_idx, Qt.ItemDataRole.DisplayRole) == "ffmpeg_restarting"


def test_degraded_row_shows_recovery_mode() -> None:
    model = HealthTableModel()
    model.set_rows(
        [
            _row(
                state=HealthState.DEGRADED,
                recovery_mode="azure_retrying",
                detail="retry 2/3",
            )
        ]
    )
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "degraded"
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == "azure_retrying"


def test_ok_row_renders_em_dash_for_recovery_mode() -> None:
    model = HealthTableModel()
    model.set_rows([_row(state=HealthState.OK)])
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == "—"


def test_operator_hint_is_surfaced_in_hint_column() -> None:
    model = HealthTableModel()
    model.set_rows(
        [
            _row(
                state=HealthState.ERROR,
                hint="restart Oura webhook",
            )
        ]
    )
    # Operator hint column
    assert model.data(model.index(0, 5), Qt.ItemDataRole.DisplayRole) == "restart Oura webhook"


# ---------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------


def test_subsystem_by_key_finds_row() -> None:
    model = HealthTableModel()
    model.set_rows([_row(key="a"), _row(key="b")])
    assert model.subsystem_by_key("b") is not None
    assert model.subsystem_by_key("missing") is None


def test_header_data_labels_columns() -> None:
    model = HealthTableModel()
    for idx, expected in enumerate(HealthTableModel.COLUMNS):
        assert (
            model.headerData(idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            == expected
        )
