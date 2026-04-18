"""Tests for `EventTimelineWidget` — Phase 5.

Uses a lightweight in-memory `QAbstractTableModel` stub so the widget
can be driven without Phase-7's production table models.
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
)

from services.operator_console.widgets.event_timeline import EventTimelineWidget

pytestmark = pytest.mark.usefixtures("qt_app")

_ROOT_INDEX: QModelIndex = QModelIndex()


class _ListModel(QAbstractTableModel):
    def __init__(self, rows: list[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows = rows

    def rowCount(  # noqa: N802
        self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(  # noqa: N802
        self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX
    ) -> int:
        return 1

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        return self._rows[index.row()]


def test_set_model_attaches_rows() -> None:
    widget = EventTimelineWidget()
    model = _ListModel(["a", "b", "c"])
    widget.set_model(model)
    attached = widget.model()
    assert attached is model
    assert attached is not None
    assert attached.rowCount() == 3


def test_scroll_to_latest_noop_without_model() -> None:
    widget = EventTimelineWidget()
    # Should not raise when no model is attached.
    widget.scroll_to_latest()


def test_scroll_to_latest_noop_on_empty_model() -> None:
    widget = EventTimelineWidget()
    widget.set_model(_ListModel([]))
    widget.scroll_to_latest()  # should not raise


def test_scroll_to_latest_with_rows() -> None:
    widget = EventTimelineWidget()
    widget.set_model(_ListModel(["first", "second", "latest"]))
    # Simply confirms it runs end-to-end without raising; real scroll
    # position is an interaction concern tested by Phase 11 integration.
    widget.scroll_to_latest()
