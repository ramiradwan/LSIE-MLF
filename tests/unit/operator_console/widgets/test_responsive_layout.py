from __future__ import annotations

import pytest
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTableView

from services.operator_console.widgets.responsive_layout import (
    MetricGridColumns,
    ResponsiveBreakpoints,
    ResponsiveMetricGrid,
    ResponsiveWidthBand,
    TableColumnPolicy,
    apply_table_column_policies,
)

pytestmark = pytest.mark.usefixtures("qt_app")

_ROOT_INDEX: QModelIndex = QModelIndex()


class _TableModel(QAbstractTableModel):
    def __init__(self, rows: list[tuple[str, ...]], parent: QObject | None = None) -> None:
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
        return len(self._rows[0])

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        return self._rows[index.row()][index.column()]


def test_metric_grid_reflows_for_each_width_band() -> None:
    grid = ResponsiveMetricGrid(
        breakpoints=ResponsiveBreakpoints(medium_min_width=400, wide_min_width=700),
        columns=MetricGridColumns(wide=3, medium=2, narrow=1),
    )
    cards = [QLabel(f"card {index}") for index in range(5)]
    grid.set_widgets(cards)

    assert grid.apply_width(320) is ResponsiveWidthBand.NARROW
    assert grid.current_band() is ResponsiveWidthBand.NARROW
    assert grid.column_count() == 1
    assert grid.layout().itemAtPosition(4, 0).widget() is cards[4]  # type: ignore[union-attr]

    assert grid.apply_width(500) is ResponsiveWidthBand.MEDIUM
    assert grid.current_band() is ResponsiveWidthBand.MEDIUM
    assert grid.column_count() == 2
    assert grid.layout().itemAtPosition(1, 1).widget() is cards[3]  # type: ignore[union-attr]

    assert grid.apply_width(900) is ResponsiveWidthBand.WIDE
    assert grid.current_band() is ResponsiveWidthBand.WIDE
    assert grid.column_count() == 3
    assert grid.layout().itemAtPosition(1, 1).widget() is cards[4]  # type: ignore[union-attr]


def test_metric_grid_add_widget_preserves_order() -> None:
    grid = ResponsiveMetricGrid(columns=MetricGridColumns(wide=2, medium=2, narrow=1))
    first = QLabel("first")
    second = QLabel("second")
    third = QLabel("third")

    grid.set_widgets([first, second])
    grid.apply_width(1200)
    grid.add_widget(third)

    assert grid.widgets() == (first, second, third)
    assert grid.layout().itemAtPosition(1, 0).widget() is third  # type: ignore[union-attr]


def test_apply_table_column_policies_hides_and_sizes_columns_by_band() -> None:
    table = QTableView()
    table.setModel(_TableModel([("a", "b", "c")]))
    policies = [
        TableColumnPolicy(
            column=0,
            resize_mode=QHeaderView.ResizeMode.ResizeToContents,
            widths={ResponsiveWidthBand.WIDE: 180},
        ),
        TableColumnPolicy(
            column=2,
            visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        ),
    ]

    narrow_band = apply_table_column_policies(
        table,
        policies,
        width=320,
        breakpoints=ResponsiveBreakpoints(medium_min_width=400, wide_min_width=700),
        default_resize_mode=QHeaderView.ResizeMode.Stretch,
    )
    assert narrow_band is ResponsiveWidthBand.NARROW
    assert table.isColumnHidden(2) is True
    assert table.horizontalHeader().sectionResizeMode(0) is QHeaderView.ResizeMode.ResizeToContents

    wide_band = apply_table_column_policies(
        table,
        policies,
        width=900,
        breakpoints=ResponsiveBreakpoints(medium_min_width=400, wide_min_width=700),
        default_resize_mode=QHeaderView.ResizeMode.Stretch,
    )
    assert wide_band is ResponsiveWidthBand.WIDE
    assert table.isColumnHidden(2) is False
    assert table.horizontalHeader().sectionResizeMode(0) is QHeaderView.ResizeMode.ResizeToContents
