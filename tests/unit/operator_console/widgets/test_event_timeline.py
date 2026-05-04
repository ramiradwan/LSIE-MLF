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
from PySide6.QtWidgets import QHeaderView

from services.operator_console.widgets.event_timeline import EventTimelineWidget
from services.operator_console.widgets.responsive_layout import (
    ResponsiveBreakpoints,
    ResponsiveWidthBand,
    TableColumnPolicy,
)

pytestmark = pytest.mark.usefixtures("qt_app")

_ROOT_INDEX: QModelIndex = QModelIndex()


class _ListModel(QAbstractTableModel):
    def __init__(
        self,
        rows: list[str] | list[tuple[str, ...]],
        parent: QObject | None = None,
    ) -> None:
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
        if parent.isValid() or not self._rows:
            return 0
        first = self._rows[0]
        return len(first) if isinstance(first, tuple) else 1

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        if isinstance(row, tuple):
            return row[index.column()]
        return row


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


def test_column_policies_follow_width_band() -> None:
    widget = EventTimelineWidget()
    widget.set_model(_ListModel([("ts", "kind", "detail")]))
    widget.set_column_policies(
        [
            TableColumnPolicy(
                column=2,
                visible_in=frozenset({ResponsiveWidthBand.WIDE}),
            ),
            TableColumnPolicy(
                column=0,
                resize_mode=QHeaderView.ResizeMode.ResizeToContents,
            ),
        ],
        breakpoints=ResponsiveBreakpoints(medium_min_width=400, wide_min_width=700),
    )

    assert widget.apply_responsive_width(320) is ResponsiveWidthBand.NARROW
    assert widget.current_width_band() is ResponsiveWidthBand.NARROW
    assert widget._table.isColumnHidden(2) is True  # type: ignore[attr-defined]
    header = widget._table.horizontalHeader()  # type: ignore[attr-defined]
    assert header.sectionResizeMode(0) is QHeaderView.ResizeMode.ResizeToContents

    assert widget.apply_responsive_width(900) is ResponsiveWidthBand.WIDE
    assert widget.current_width_band() is ResponsiveWidthBand.WIDE
    assert widget._table.isColumnHidden(2) is False  # type: ignore[attr-defined]
