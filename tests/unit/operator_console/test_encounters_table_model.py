"""Regression tests for `EncountersTableModel` — Phase 7.

The Live Session page keys operator trust off this model: the columns
are the §7B reward-explanation fields, and `set_rows` must preserve
selection when row identities are stable so the operator's selected
encounter does not blink away on every poll tick.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from PySide6.QtCore import QModelIndex, Qt

from packages.schemas.operator_console import EncounterState, EncounterSummary
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)

pytestmark = pytest.mark.usefixtures("qt_app")


def _make_encounter(
    encounter_id: str,
    *,
    state: EncounterState = EncounterState.COMPLETED,
    semantic_gate: int | None = 1,
    p90: float | None = 0.42,
    gated_reward: float | None = 0.42,
    frames: int | None = 150,
    physiology_attached: bool = True,
    physiology_stale: bool | None = False,
    active_arm: str | None = "greeting_v1",
) -> EncounterSummary:
    return EncounterSummary(
        encounter_id=encounter_id,
        session_id=uuid4(),
        segment_timestamp_utc=datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC),
        state=state,
        active_arm=active_arm,
        expected_greeting="hei rakas",
        semantic_gate=semantic_gate,
        p90_intensity=p90,
        gated_reward=gated_reward,
        n_frames_in_window=frames,
        physiology_attached=physiology_attached,
        physiology_stale=physiology_stale,
    )


# ---------------------------------------------------------------------
# Row / column count
# ---------------------------------------------------------------------


def test_column_count_matches_column_tuple() -> None:
    model = EncountersTableModel()
    assert model.columnCount() == len(EncountersTableModel.COLUMNS)


def test_row_count_reflects_set_rows() -> None:
    model = EncountersTableModel()
    assert model.rowCount() == 0
    model.set_rows([_make_encounter("e1"), _make_encounter("e2")])
    assert model.rowCount() == 2


def test_row_count_ignores_valid_parent() -> None:
    # Qt contract: table models return 0 children for any valid parent
    # index so tree-style views do not try to recurse.
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1")])
    parent = model.index(0, 0)
    assert model.rowCount(parent) == 0
    assert model.columnCount(parent) == 0


# ---------------------------------------------------------------------
# Display values — §7B reward-explanation fields
# ---------------------------------------------------------------------


def test_display_semantic_gate_reads_closed_when_zero() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1", semantic_gate=0)])
    idx = model.index(0, 3)  # Semantic gate
    assert "held back" in model.data(idx, Qt.ItemDataRole.DisplayRole)


def test_display_frames_in_window_zero_is_rendered_as_zero() -> None:
    # "0 frames" is a legitimate §7B outcome and must render numerically —
    # an em-dash here would look like missing data.
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1", frames=0)])
    idx = model.index(0, 6)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "0"


def test_display_physiology_stale_is_visible() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1", physiology_stale=True)])
    idx = model.index(0, 7)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "attached (stale)"


def test_display_physiology_absent_reads_absent() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1", physiology_attached=False)])
    idx = model.index(0, 7)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "absent"


def test_header_data_returns_column_label() -> None:
    model = EncountersTableModel()
    label = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
    assert "Timestamp" in label


def test_header_data_vertical_returns_row_number() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1")])
    value = model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole)
    assert value == 1


# ---------------------------------------------------------------------
# Selection preservation — same-identity update must emit dataChanged
# ---------------------------------------------------------------------


def test_same_identity_update_emits_data_changed_not_reset() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1"), _make_encounter("e2")])
    reset_hits = 0
    change_hits: list[tuple[int, int]] = []

    def on_reset() -> None:
        nonlocal reset_hits
        reset_hits += 1

    def on_change(top_left: QModelIndex, bottom_right: QModelIndex) -> None:
        change_hits.append((top_left.row(), bottom_right.row()))

    model.modelReset.connect(on_reset)
    model.dataChanged.connect(on_change)
    model.set_rows(
        [
            _make_encounter("e1", gated_reward=0.55),
            _make_encounter("e2", gated_reward=0.12),
        ]
    )
    assert reset_hits == 0
    assert change_hits == [(0, 1)]


def test_different_identity_update_resets_model() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1")])
    reset_hits = 0

    def on_reset() -> None:
        nonlocal reset_hits
        reset_hits += 1

    model.modelReset.connect(on_reset)
    model.set_rows([_make_encounter("e1"), _make_encounter("e2")])
    assert reset_hits == 1


# ---------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------


def test_row_at_returns_none_for_out_of_range() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1")])
    assert model.row_at(0) is not None
    assert model.row_at(-1) is None
    assert model.row_at(5) is None


def test_encounter_by_id_finds_row() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1"), _make_encounter("e2")])
    assert model.encounter_by_id("e2") is not None
    assert model.encounter_by_id("nope") is None


def test_index_of_encounter_returns_position() -> None:
    model = EncountersTableModel()
    model.set_rows([_make_encounter("e1"), _make_encounter("e2"), _make_encounter("e3")])
    assert model.index_of_encounter("e2") == 1
    assert model.index_of_encounter("missing") is None
