"""Responsive operator-console layout primitives.

Provides two reusable presentation-only helpers:

* ``ResponsiveMetricGrid`` — reflows child widgets across wide / medium /
  narrow widths without coupling cards to any specific page.
* ``TableColumnPolicy`` + ``apply_table_column_policies`` — width-band-aware
  column visibility and resize behavior for table and timeline surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QGridLayout, QHeaderView, QTableView, QWidget


class ResponsiveWidthBand(StrEnum):
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"


_ALL_WIDTH_BANDS: frozenset[ResponsiveWidthBand] = frozenset(ResponsiveWidthBand)


@dataclass(frozen=True)
class ResponsiveBreakpoints:
    medium_min_width: int = 640
    wide_min_width: int = 980

    def band_for_width(self, width: int) -> ResponsiveWidthBand:
        if width >= self.wide_min_width:
            return ResponsiveWidthBand.WIDE
        if width >= self.medium_min_width:
            return ResponsiveWidthBand.MEDIUM
        return ResponsiveWidthBand.NARROW


@dataclass(frozen=True)
class MetricGridColumns:
    wide: int = 3
    medium: int = 2
    narrow: int = 1

    def for_band(self, band: ResponsiveWidthBand) -> int:
        if band is ResponsiveWidthBand.WIDE:
            return max(1, self.wide)
        if band is ResponsiveWidthBand.MEDIUM:
            return max(1, self.medium)
        return max(1, self.narrow)

    def max_columns(self) -> int:
        return max(self.wide, self.medium, self.narrow, 1)


class ResponsiveMetricGrid(QWidget):
    def __init__(
        self,
        *,
        breakpoints: ResponsiveBreakpoints | None = None,
        columns: MetricGridColumns | None = None,
        horizontal_spacing: int = 14,
        vertical_spacing: int = 14,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._breakpoints = breakpoints or ResponsiveBreakpoints()
        self._columns = columns or MetricGridColumns()
        self._widgets: list[QWidget] = []
        self._current_band = ResponsiveWidthBand.NARROW

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(horizontal_spacing)
        self._layout.setVerticalSpacing(vertical_spacing)

    def set_widgets(self, widgets: Sequence[QWidget]) -> None:
        self._widgets = list(widgets)
        for widget in self._widgets:
            if widget.parentWidget() is not self:
                widget.setParent(self)
        self._reflow(width_override=max(0, self.width()))

    def add_widget(self, widget: QWidget) -> None:
        if widget.parentWidget() is not self:
            widget.setParent(self)
        self._widgets.append(widget)
        self._reflow(width_override=max(0, self.width()))

    def widgets(self) -> tuple[QWidget, ...]:
        return tuple(self._widgets)

    def current_band(self) -> ResponsiveWidthBand:
        return self._current_band

    def column_count(self) -> int:
        return self._columns.for_band(self._current_band)

    def apply_width(self, width: int) -> ResponsiveWidthBand:
        self._reflow(width_override=width)
        return self._current_band

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow(width_override=event.size().width())

    def _reflow(self, *, width_override: int) -> None:
        width = width_override if width_override > 0 else self._breakpoints.wide_min_width
        band = self._breakpoints.band_for_width(width)
        columns = self._columns.for_band(band)
        self._current_band = band

        while self._layout.count() > 0:
            self._layout.takeAt(0)

        for index, widget in enumerate(self._widgets):
            row, column = divmod(index, columns)
            self._layout.addWidget(widget, row, column)

        for column in range(self._columns.max_columns()):
            self._layout.setColumnStretch(column, 1 if column < columns else 0)


@dataclass(frozen=True)
class TableColumnPolicy:
    column: int
    visible_in: frozenset[ResponsiveWidthBand] = _ALL_WIDTH_BANDS
    resize_mode: QHeaderView.ResizeMode | None = None
    resize_modes: dict[ResponsiveWidthBand, QHeaderView.ResizeMode] = field(default_factory=dict)
    widths: dict[ResponsiveWidthBand, int] = field(default_factory=dict)

    def is_visible(self, band: ResponsiveWidthBand) -> bool:
        return band in self.visible_in

    def resize_mode_for_band(self, band: ResponsiveWidthBand) -> QHeaderView.ResizeMode | None:
        return self.resize_modes.get(band, self.resize_mode)

    def width_for_band(self, band: ResponsiveWidthBand) -> int | None:
        return self.widths.get(band)


def apply_table_column_policies(
    table: QTableView,
    policies: Sequence[TableColumnPolicy],
    *,
    width: int | None = None,
    breakpoints: ResponsiveBreakpoints | None = None,
    default_resize_mode: QHeaderView.ResizeMode | None = None,
) -> ResponsiveWidthBand:
    resolved_breakpoints = breakpoints or ResponsiveBreakpoints()
    effective_width = width if width is not None else max(table.viewport().width(), table.width())
    band = resolved_breakpoints.band_for_width(effective_width)

    model = table.model()
    if model is None:
        return band
    header = table.horizontalHeader()
    if header is None:
        return band

    column_count = model.columnCount()
    if default_resize_mode is not None:
        for column in range(column_count):
            header.setSectionResizeMode(column, default_resize_mode)
            table.setColumnHidden(column, False)

    for policy in policies:
        if policy.column < 0 or policy.column >= column_count:
            continue
        visible = policy.is_visible(band)
        table.setColumnHidden(policy.column, not visible)
        resize_mode = policy.resize_mode_for_band(band)
        if resize_mode is not None:
            header.setSectionResizeMode(policy.column, resize_mode)
        width_override = policy.width_for_band(band)
        if visible and width_override is not None:
            table.setColumnWidth(policy.column, width_override)

    return band
