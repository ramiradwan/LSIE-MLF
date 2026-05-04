"""Health page — §12 subsystem rollup + alert timeline.

The operator-trust surface for subsystem state. §12 draws three lines
the UI must not collapse:
  * **ok** — green, self-evident
  * **degraded** — subsystem is impaired but the system knows how to
    keep moving (e.g., Azure retry-then-null, DB write buffer)
  * **recovering** — subsystem is actively self-healing (ADB drift
    reset, FFmpeg restart)
  * **error** — subsystem is hard-down and the operator has to act

The page presents these as a subsystem table (recovery-mode column
distinct from the detail column) and a full alert timeline below. Every
string passes through `formatters.py`; no inline formatting.

Spec references:
  §4.E.1         — Health operator surface
  §12            — degraded vs recovering vs error; operator-action hints
"""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, Slot
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    UiStatusKind,
)
from services.operator_console.formatters import (
    build_health_detail,
    build_health_probe_detail,
    format_health_probe_state,
    format_health_state,
    format_latency_ms,
    format_timestamp,
)
from services.operator_console.viewmodels.health_vm import HealthViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.event_timeline import EventTimelineWidget
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.widgets.responsive_layout import (
    MetricGridColumns,
    ResponsiveBreakpoints,
    ResponsiveMetricGrid,
    ResponsiveWidthBand,
    TableColumnPolicy,
    apply_table_column_policies,
)
from services.operator_console.widgets.section_header import SectionHeader
from services.operator_console.widgets.status_pill import StatusPill

# §12 subsystem state → overview card pill kind. Recovering renders as
# `PROGRESS` (self-healing in flight) so it reads distinct from both
# degraded (impaired but stable) and error (down).
_HEALTH_STATUS: dict[HealthState, UiStatusKind] = {
    HealthState.OK: UiStatusKind.OK,
    HealthState.DEGRADED: UiStatusKind.WARN,
    HealthState.RECOVERING: UiStatusKind.PROGRESS,
    HealthState.ERROR: UiStatusKind.ERROR,
    HealthState.UNKNOWN: UiStatusKind.NEUTRAL,
}

_PROBE_STATUS: dict[HealthProbeState, UiStatusKind] = {
    HealthProbeState.OK: UiStatusKind.OK,
    HealthProbeState.ERROR: UiStatusKind.ERROR,
    HealthProbeState.TIMEOUT: UiStatusKind.ERROR,
    HealthProbeState.NOT_CONFIGURED: UiStatusKind.NEUTRAL,
    HealthProbeState.UNKNOWN: UiStatusKind.NEUTRAL,
}

_HEALTH_BREAKPOINTS = ResponsiveBreakpoints(medium_min_width=760, wide_min_width=1120)

_SUBSYSTEM_TABLE_POLICIES: tuple[TableColumnPolicy, ...] = (
    TableColumnPolicy(
        column=0,
        resize_modes={
            ResponsiveWidthBand.NARROW: QHeaderView.ResizeMode.ResizeToContents,
            ResponsiveWidthBand.MEDIUM: QHeaderView.ResizeMode.ResizeToContents,
        },
        widths={ResponsiveWidthBand.WIDE: 180},
    ),
    TableColumnPolicy(
        column=1,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 120},
    ),
    TableColumnPolicy(
        column=2,
        visible_in=frozenset({ResponsiveWidthBand.MEDIUM, ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 150},
    ),
    TableColumnPolicy(
        column=3,
        resize_mode=QHeaderView.ResizeMode.Stretch,
    ),
    TableColumnPolicy(
        column=4,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 170},
    ),
    TableColumnPolicy(
        column=5,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.Stretch,
    ),
)

_ALERT_TIMELINE_POLICIES: tuple[TableColumnPolicy, ...] = (
    TableColumnPolicy(
        column=0,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 165},
    ),
    TableColumnPolicy(
        column=1,
        visible_in=frozenset({ResponsiveWidthBand.MEDIUM, ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 100},
    ),
    TableColumnPolicy(
        column=2,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 150},
    ),
    TableColumnPolicy(
        column=3,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 160},
    ),
    TableColumnPolicy(column=4, resize_mode=QHeaderView.ResizeMode.Stretch),
    TableColumnPolicy(
        column=5,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 70},
    ),
)


class HealthView(QWidget):
    """Health page: rollup cards + subsystem table + alert timeline."""

    def __init__(
        self,
        vm: HealthViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Health",
            "Readiness summary with the next action for anything that needs attention.",
            self,
        )
        self._repair_button = QPushButton("Repair install", self)
        self._repair_button.setObjectName("RepairInstallButton")
        self._repair_button.setToolTip("Rebuild the local runtime without touching desktop.sqlite.")
        self._repair_button.setEnabled(False)
        self._repair_button.clicked.connect(self._on_repair_clicked)
        self._repair_status = QLabel("", self)
        self._repair_status.setObjectName("PanelSubtitle")
        self._repair_status.setVisible(False)
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No health snapshot")
        self._empty_state.set_message(
            "Health data will appear once the desktop process graph reports its first rollup."
        )

        self._overall_card = MetricCard("Overall", self)
        self._degraded_card = MetricCard("Needs attention", self)
        self._recovering_card = MetricCard("Getting ready", self)
        self._error_card = MetricCard("Error", self)

        self._cards_grid = ResponsiveMetricGrid(
            breakpoints=_HEALTH_BREAKPOINTS,
            columns=MetricGridColumns(wide=4, medium=2, narrow=1),
            parent=self,
        )
        self._cards_grid.set_widgets(
            [
                self._overall_card,
                self._degraded_card,
                self._recovering_card,
                self._error_card,
            ]
        )

        self._probe_matrix = _ProbeMatrix(self)
        self._subsystem_table = self._build_subsystem_table()
        self._alerts_timeline = EventTimelineWidget(self)
        self._alerts_timeline.set_model(self._vm.alerts_model())
        self._alerts_timeline.set_column_policies(
            _ALERT_TIMELINE_POLICIES,
            breakpoints=_HEALTH_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )

        self._probe_panel = _TablePanel(
            "Readiness checks",
            "Read-only checks; not configured is distinct from error.",
            self._probe_matrix,
            self,
        )
        self._subsystem_panel = _TablePanel(
            "Readiness details",
            "Each row shows what is happening and the next operator action.",
            self._subsystem_table,
            self,
        )
        self._alerts_panel = _TablePanel(
            "Alert timeline",
            "Newest alerts first. Retriable alerts may re-emit on each state tick.",
            self._alerts_timeline,
            self,
        )

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addWidget(self._cards_grid)
        body.addWidget(self._probe_panel)
        body.addWidget(self._subsystem_panel)
        body.addWidget(self._alerts_panel)
        body.addStretch(1)

        self._body_container = QWidget(self)
        self._body_container.setLayout(body)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("HealthScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._body_container)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(14)
        header_row.addWidget(self._header, 1)
        header_row.addWidget(self._repair_status)
        header_row.addWidget(self._repair_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addLayout(header_row)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._scroll, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._vm.changed.connect(self._refresh)
        self._vm.error_changed.connect(self._on_error_changed)
        self._vm.repair_requested.connect(self._on_repair_started)
        # Keep the alert timeline auto-scrolled to the latest event when
        # new rows land — operators watch the tail.
        self._vm.alerts_model().rowsInserted.connect(self._on_alerts_rows_inserted)

        self._refresh()

    # ------------------------------------------------------------------
    # Page lifecycle hooks
    # ------------------------------------------------------------------

    def on_activated(self) -> None:
        self._refresh()
        self._alerts_timeline.scroll_to_latest()

    def on_deactivated(self) -> None:
        return None

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_subsystem_table(self) -> QTableView:
        table = QTableView(self)
        table.setObjectName("HealthSubsystemTable")
        table.setModel(self._vm.health_model())
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setWordWrap(True)
        vertical = table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        horizontal = table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        snapshot = self._vm.snapshot()
        self._repair_button.setEnabled(self._vm.repair_available())
        if snapshot is None:
            self._empty_state.setVisible(True)
            self._scroll.setVisible(False)
            return
        self._empty_state.setVisible(False)
        self._scroll.setVisible(True)

        self._render_overall(snapshot)
        self._render_counts(snapshot)
        self._probe_matrix.set_probes(self._vm.subsystem_probes())
        self._apply_responsive_layout()

    def _render_overall(self, snapshot: HealthSnapshot) -> None:
        kind = _HEALTH_STATUS[snapshot.overall_state]
        self._overall_card.set_primary_text(format_health_state(snapshot.overall_state))
        self._overall_card.set_secondary_text(
            f"generated {format_timestamp(snapshot.generated_at_utc)}"
        )
        self._overall_card.set_status(kind, format_health_state(snapshot.overall_state))

    def _render_counts(self, snapshot: HealthSnapshot) -> None:
        self._degraded_card.set_primary_text(str(snapshot.degraded_count))
        self._degraded_card.set_secondary_text(
            "check the next action" if snapshot.degraded_count else "none"
        )
        self._degraded_card.set_status(
            UiStatusKind.WARN if snapshot.degraded_count else UiStatusKind.NEUTRAL,
            "degraded" if snapshot.degraded_count else None,
        )

        self._recovering_card.set_primary_text(str(snapshot.recovering_count))
        self._recovering_card.set_secondary_text(
            "wait for startup" if snapshot.recovering_count else "none"
        )
        self._recovering_card.set_status(
            UiStatusKind.PROGRESS if snapshot.recovering_count else UiStatusKind.NEUTRAL,
            "recovering" if snapshot.recovering_count else None,
        )

        self._error_card.set_primary_text(str(snapshot.error_count))
        self._error_card.set_secondary_text(
            _describe_error_hint(snapshot) if snapshot.error_count else "none"
        )
        self._error_card.set_status(
            UiStatusKind.ERROR if snapshot.error_count else UiStatusKind.NEUTRAL,
            "error" if snapshot.error_count else None,
        )

    def _apply_responsive_layout(self) -> None:
        width = max(self.width(), self._scroll.viewport().width())
        band = _HEALTH_BREAKPOINTS.band_for_width(width)
        self._probe_matrix.set_width_band(band)
        apply_table_column_policies(
            self._subsystem_table,
            _SUBSYSTEM_TABLE_POLICIES,
            width=width,
            breakpoints=_HEALTH_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )
        self._subsystem_panel.set_compact(band is ResponsiveWidthBand.NARROW)
        self._alerts_panel.set_compact(band is ResponsiveWidthBand.NARROW)
        self._probe_panel.set_compact(band is ResponsiveWidthBand.NARROW)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)

    def _on_repair_clicked(self) -> None:
        if not self._vm.request_repair():
            return

    def _on_repair_started(self) -> None:
        self._repair_status.setText("Repair requested")
        self._repair_status.setVisible(True)

    @Slot(QModelIndex, int, int)
    def _on_alerts_rows_inserted(
        self,
        _parent: QModelIndex,
        _first: int,
        _last: int,
    ) -> None:
        # Model rows append at the start (newest-first) for the alerts
        # feed; for the timeline wrapper we still want the latest row
        # visible, so ask it to scroll to the bottom defensively.
        self._alerts_timeline.scroll_to_latest()


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------


class _ProbeMatrix(QFrame):
    """Bounded subsystem probes with width-aware presentation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HealthProbeMatrix")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(18)
        self._grid.setVerticalSpacing(8)
        self._state_pills: list[StatusPill] = []
        self._latency_labels: list[QLabel] = []
        self._name_labels: list[QLabel] = []
        self._header_labels: list[QLabel] = []
        self._probes: list[HealthSubsystemProbe] = []
        self._width_band = ResponsiveWidthBand.WIDE
        self._empty_label = QLabel("No bounded probes reported", self)
        self._empty_label.setObjectName("PanelSubtitle")
        self._render_headers()
        self._show_empty()

    def set_probes(self, probes: list[HealthSubsystemProbe]) -> None:
        self._probes = list(probes)
        self._render_rows()

    def set_width_band(self, band: ResponsiveWidthBand) -> None:
        if band is self._width_band:
            return
        self._width_band = band
        self._render_rows()

    def _render_rows(self) -> None:
        self._clear_rows()
        if not self._probes:
            self._show_empty()
            return
        self._empty_label.setVisible(False)
        compact = self._width_band is ResponsiveWidthBand.NARROW
        for row_index, probe in enumerate(self._probes, start=1):
            detail = build_health_probe_detail(probe)
            name = QLabel(probe.label or probe.subsystem_key, self)
            name.setObjectName("MetricCardSecondary")
            name.setToolTip(detail)
            name.setWordWrap(True)

            pill = StatusPill(self)
            pill.set_kind(_PROBE_STATUS[probe.state])
            pill.set_text(format_health_probe_state(probe.state))
            pill.setToolTip(detail)

            latency = QLabel(format_latency_ms(probe.latency_ms), self)
            latency.setObjectName("MetricCardSecondary")
            latency.setToolTip(detail)

            self._name_labels.append(name)
            self._state_pills.append(pill)
            self._latency_labels.append(latency)

            if compact:
                self._grid.addWidget(name, row_index, 0, 1, 2)
                self._grid.addWidget(pill, row_index, 2)
                self._grid.addWidget(latency, row_index, 3)
            else:
                self._grid.addWidget(name, row_index, 0)
                self._grid.addWidget(pill, row_index, 1)
                self._grid.addWidget(latency, row_index, 2)
        self._apply_header_visibility(compact)

    def _render_headers(self) -> None:
        for column, title in enumerate(("Subsystem", "State", "Latency")):
            label = QLabel(title, self)
            label.setObjectName("PanelSubtitle")
            self._grid.addWidget(label, 0, column)
            self._header_labels.append(label)

    def _apply_header_visibility(self, compact: bool) -> None:
        for label in self._header_labels:
            label.setVisible(not compact)

    def _show_empty(self) -> None:
        self._apply_header_visibility(self._width_band is ResponsiveWidthBand.NARROW)
        self._empty_label.setVisible(True)
        span = 4 if self._width_band is ResponsiveWidthBand.NARROW else 3
        self._grid.addWidget(self._empty_label, 1, 0, 1, span)

    def _clear_rows(self) -> None:
        self._name_labels.clear()
        self._state_pills.clear()
        self._latency_labels.clear()
        while self._grid.count() > len(self._header_labels):
            item = self._grid.takeAt(len(self._header_labels))
            widget = item.widget() if item is not None else None
            if widget is self._empty_label:
                widget.setVisible(False)
            elif widget is not None:
                widget.deleteLater()


class _TablePanel(QFrame):
    """A framed titled panel hosting a tabular widget.

    Kept as a simple wrapper so the subsystem table and the alert
    timeline present consistently — title + subtitle + embedded widget.
    """

    def __init__(
        self,
        title: str,
        subtitle: str,
        inner: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel(title, self)
        self._title.setObjectName("PanelTitle")
        self._subtitle = QLabel(subtitle, self)
        self._subtitle.setObjectName("PanelSubtitle")
        self._subtitle.setWordWrap(True)
        self._inner = inner

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 12, 16, 12)
        self._layout.setSpacing(6)
        self._layout.addWidget(self._title)
        self._layout.addWidget(self._subtitle)
        self._layout.addWidget(inner, 1)

    def set_compact(self, compact: bool) -> None:
        self._layout.setContentsMargins(12, 10, 12, 10 if compact else 12)
        self._layout.setSpacing(4 if compact else 6)
        self._subtitle.setVisible(not compact)


def _describe_error_hint(snapshot: HealthSnapshot) -> str:
    """Pick the first error subsystem's operator hint, or a neutral line.

    The count card is a summary; surfacing a hint for the first error
    row gives the operator a nudge without having to scroll the table.
    """
    for row in snapshot.subsystems:
        if row.state == HealthState.ERROR:
            hint = row.operator_action_hint or build_health_detail(row)
            return hint
    return "see subsystem table below"
