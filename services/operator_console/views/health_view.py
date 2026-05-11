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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    CloudAuthState,
    CloudAuthStatus,
    CloudOutboxSummary,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    UiStatusKind,
)
from services.operator_console.formatters import (
    HealthActionCopy,
    build_health_detail,
    build_health_probe_detail,
    format_health_action_copy,
    format_health_action_error,
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
from services.operator_console.widgets.probe_sparkline import (
    ProbeSparkline,
    ProbeSparklineCell,
)
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
    HealthProbeState.NOT_CONFIGURED: UiStatusKind.MUTED,
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
            level="page",
        )
        self._cloud_sign_in_button = QPushButton("Cloud sign-in", self)
        self._cloud_sign_in_button.setObjectName("CloudSignInButton")
        self._cloud_sign_in_button.setAccessibleName("Cloud sign-in")
        self._cloud_sign_in_button.setAccessibleDescription(
            "Open browser sign-in so cloud sync can upload and refresh experiments."
        )
        self._cloud_sign_in_button.setToolTip(
            "Open browser sign-in for cloud sync and experiment refresh."
        )
        self._cloud_sign_in_button.setEnabled(False)
        self._cloud_sign_in_button.clicked.connect(self._on_cloud_sign_in_clicked)
        self._experiment_bundle_button = QPushButton("Refresh experiments", self)
        self._experiment_bundle_button.setObjectName("ExperimentBundleRefreshButton")
        self._experiment_bundle_button.setAccessibleName("Refresh experiments")
        self._experiment_bundle_button.setAccessibleDescription(
            "Download and apply the latest signed experiment bundle."
        )
        self._experiment_bundle_button.setToolTip(
            "Download and apply the latest signed experiment bundle."
        )
        self._experiment_bundle_button.setEnabled(False)
        self._experiment_bundle_button.clicked.connect(self._on_experiment_bundle_clicked)
        self._repair_button = QPushButton("Repair install", self)
        self._repair_button.setObjectName("RepairInstallButton")
        self._repair_button.setAccessibleName("Repair install")
        self._repair_button.setAccessibleDescription(
            "Rebuilds the local runtime without touching desktop.sqlite."
        )
        self._repair_button.setToolTip("Rebuild the local runtime without touching desktop.sqlite.")
        self._repair_button.setEnabled(False)
        self._repair_button.clicked.connect(self._on_repair_clicked)
        self._action_status = QLabel("", self)
        self._action_status.setObjectName("PanelSubtitle")
        self._action_status.setAccessibleName("Health action status")
        self._action_status.setAccessibleDescription("No health action requested.")
        self._action_status.setVisible(False)
        self._repair_status = self._action_status
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No health snapshot")
        self._empty_state.set_message(
            "Health data will appear once the app reports its first readiness update."
        )

        self._overall_card = MetricCard("Overall", self)
        self._degraded_card = MetricCard("Needs attention", self)
        self._recovering_card = MetricCard("Getting ready", self)
        self._error_card = MetricCard("Error", self)
        self._cloud_auth_card = MetricCard("Cloud sign-in", self)
        self._cloud_outbox_card = MetricCard("Cloud outbox", self)

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
                self._cloud_auth_card,
                self._cloud_outbox_card,
            ]
        )

        self._probe_matrix = _ProbeMatrix(self._vm, self)
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
        header_row.addWidget(self._action_status)
        header_row.addWidget(self._cloud_sign_in_button)
        header_row.addWidget(self._experiment_bundle_button)
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
        self._vm.cloud_sign_in_requested.connect(self._on_cloud_sign_in_started)
        self._vm.experiment_bundle_refresh_requested.connect(
            self._on_experiment_bundle_refresh_started
        )
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
        table.setAccessibleName("Readiness details table")
        table.setAccessibleDescription(
            "Subsystem readiness details with recovery mode, status, and next operator action."
        )
        table.setToolTip("Subsystem readiness details and next operator actions.")
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
        self._cloud_sign_in_button.setEnabled(self._vm.cloud_sign_in_available())
        self._experiment_bundle_button.setEnabled(self._vm.experiment_bundle_refresh_available())
        self._repair_button.setEnabled(self._vm.repair_available())

        sign_in_copy = format_health_action_copy(
            "cloud_sign_in",
            stage=("progress" if self._vm.cloud_sign_in_in_progress() else "idle"),
        )
        refresh_copy = format_health_action_copy(
            "experiment_bundle_refresh",
            stage=("progress" if self._vm.experiment_bundle_refresh_in_progress() else "idle"),
        )
        repair_copy = format_health_action_copy(
            "repair_install",
            stage=("progress" if self._vm.repair_in_progress() else "idle"),
        )
        self._cloud_sign_in_button.setText(sign_in_copy.button_label)
        self._experiment_bundle_button.setText(refresh_copy.button_label)
        self._repair_button.setText(repair_copy.button_label)

        action_state = self._vm.action_state()
        if action_state == "cloud_sign_in_progress":
            self._apply_action_status(format_health_action_copy("cloud_sign_in", stage="progress"))
        elif action_state == "cloud_sign_in_success":
            self._apply_action_status(format_health_action_copy("cloud_sign_in", stage="success"))
        elif action_state == "experiment_bundle_refresh_progress":
            self._apply_action_status(
                format_health_action_copy("experiment_bundle_refresh", stage="progress")
            )
        elif action_state == "experiment_bundle_refresh_success":
            self._apply_action_status(
                format_health_action_copy("experiment_bundle_refresh", stage="success")
            )
        elif action_state == "repair_install_progress":
            self._apply_action_status(format_health_action_copy("repair_install", stage="progress"))
        elif action_state == "repair_install_success":
            self._apply_action_status(format_health_action_copy("repair_install", stage="success"))
        elif action_state.endswith("_failure") or not self._action_status.text():
            self._action_status.setVisible(False)
        if snapshot is None:
            self._empty_state.setVisible(True)
            self._scroll.setVisible(False)
            return
        self._empty_state.setVisible(False)
        self._scroll.setVisible(True)

        self._render_overall(snapshot)
        self._render_counts(snapshot)
        self._render_cloud_auth(self._vm.cloud_auth_status())
        self._render_cloud_outbox(self._vm.cloud_outbox_summary())
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

    def _render_cloud_auth(self, status: CloudAuthStatus | None) -> None:
        if status is None:
            self._cloud_auth_card.set_primary_text("Unknown")
            self._cloud_auth_card.set_secondary_text("waiting for cloud status")
            self._cloud_auth_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        self._cloud_auth_card.set_primary_text(status.state.value.replace("_", " "))
        self._cloud_auth_card.set_secondary_text(status.message or "No message reported.")
        self._cloud_auth_card.set_status(_cloud_auth_kind(status.state), status.state.value)

    def _render_cloud_outbox(self, summary: CloudOutboxSummary | None) -> None:
        if summary is None:
            self._cloud_outbox_card.set_primary_text("Unknown")
            self._cloud_outbox_card.set_secondary_text("waiting for outbox summary")
            self._cloud_outbox_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        pending = summary.pending_count + summary.in_flight_count + summary.retry_scheduled_count
        blocked = summary.dead_letter_count
        self._cloud_outbox_card.set_primary_text(f"{pending} active")
        self._cloud_outbox_card.set_secondary_text(
            f"{blocked} dead-letter · {summary.redacted_count} redacted"
        )
        if blocked:
            self._cloud_outbox_card.set_status(UiStatusKind.ERROR, "dead-letter")
        elif pending:
            self._cloud_outbox_card.set_status(UiStatusKind.PROGRESS, "uploading")
        else:
            self._cloud_outbox_card.set_status(UiStatusKind.OK, "clear")

    def _apply_responsive_layout(self) -> None:
        viewport_width = self._scroll.viewport().width()
        width = viewport_width if viewport_width >= 320 else self.width()
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
            action_state = self._vm.action_state()
            if action_state == "cloud_sign_in_failure":
                banner_message = format_health_action_error("cloud_sign_in", message)
            elif action_state == "experiment_bundle_refresh_failure":
                banner_message = format_health_action_error(
                    "experiment_bundle_refresh",
                    message,
                )
            elif action_state == "repair_install_failure":
                banner_message = format_health_action_error("repair_install", message)
            else:
                banner_message = message
            self._error_banner.set_alert(AlertSeverity.WARNING, banner_message)
            self._action_status.setVisible(False)
            self._action_status.setText("")
            self._action_status.setAccessibleDescription("No health action requested.")
        else:
            self._error_banner.set_alert(None, None)

    def _apply_action_status(self, copy: HealthActionCopy) -> None:
        status_text = copy.status_text
        accessible_description = copy.accessible_description
        if status_text is None:
            self._action_status.setVisible(False)
            self._action_status.setText("")
            self._action_status.setAccessibleDescription("No health action requested.")
            return
        self._action_status.setText(status_text)
        self._action_status.setAccessibleDescription(
            accessible_description or "Health action status."
        )
        self._action_status.setVisible(True)

    def _on_cloud_sign_in_clicked(self) -> None:
        if not self._vm.request_cloud_sign_in():
            return

    def _on_experiment_bundle_clicked(self) -> None:
        if not self._vm.request_experiment_bundle_refresh():
            return

    def _on_repair_clicked(self) -> None:
        # Repair install rebuilds the runtime; treat it as deliberate
        # rather than as a one-tap action by confirming the consequence
        # before letting the VM dispatch.
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setWindowTitle("Repair install")
        confirmation.setText("Rebuild the local runtime?")
        confirmation.setInformativeText(
            "This will reinstall the local runtime without touching desktop.sqlite. "
            "The active session keeps running but readiness will dip while the "
            "subsystem comes back up. Continue?"
        )
        confirmation.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        confirmation.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirmation.exec() != QMessageBox.StandardButton.Yes:
            return
        if not self._vm.request_repair():
            return

    def _on_cloud_sign_in_started(self) -> None:
        self._action_status.setText("Waiting for sign-in…")
        self._action_status.setAccessibleDescription(
            "Browser sign-in in progress for cloud sync and experiment refresh."
        )
        self._action_status.setVisible(True)
        self._cloud_sign_in_button.setEnabled(False)
        self._cloud_sign_in_button.setText("Signing in…")

    def _on_experiment_bundle_refresh_started(self) -> None:
        self._action_status.setText("Refreshing experiments…")
        self._action_status.setAccessibleDescription("Experiment bundle refresh in progress.")
        self._action_status.setVisible(True)
        self._experiment_bundle_button.setEnabled(False)
        self._experiment_bundle_button.setText("Refreshing…")

    def _on_repair_started(self) -> None:
        self._action_status.setText("Installing runtime…")
        self._action_status.setAccessibleDescription(
            "Repair install in progress; subsystem rows will reflect recovery state."
        )
        self._action_status.setVisible(True)
        self._repair_button.setEnabled(False)
        self._repair_button.setText("Installing…")

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
    """Row-per-subsystem probe matrix with a 60-cell history sparkline.

    Each row pairs the current pill + latency with a sparkline of the
    last 60 probe samples for that subsystem, so flapping reads as a
    visible pattern without bouncing between the matrix and the alert
    timeline.
    """

    _HEADERS: tuple[str, ...] = ("Subsystem", "State", "Latency", "Last 60 probes")

    def __init__(self, vm: HealthViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vm = vm
        self.setObjectName("HealthProbeMatrix")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(18)
        self._grid.setVerticalSpacing(8)
        self._state_pills: list[StatusPill] = []
        self._latency_labels: list[QLabel] = []
        self._name_labels: list[QLabel] = []
        self._sparklines: list[ProbeSparkline] = []
        self._header_labels: list[QLabel] = []
        self._probes: list[HealthSubsystemProbe] = []
        self._width_band = ResponsiveWidthBand.WIDE
        self._empty_label: QLabel | None = None
        self._rebuild()

    def set_probes(self, probes: list[HealthSubsystemProbe]) -> None:
        new_probes = list(probes)
        # Refresh in place when the probe identity + state hasn't changed —
        # only the rolling sparkline cells are new every poll. Rebuilding
        # the entire grid every poll tore down hover state and queued
        # deleteLater() faster than the event loop could drain, which
        # surfaced as a flicker of small floating widgets on the Health
        # page during normal hover.
        if self._can_refresh_in_place(new_probes):
            self._probes = new_probes
            self._refresh_in_place()
            return
        self._probes = new_probes
        self._rebuild()

    def set_width_band(self, band: ResponsiveWidthBand) -> None:
        if band is self._width_band:
            return
        self._width_band = band
        self._rebuild()

    def _can_refresh_in_place(self, new_probes: list[HealthSubsystemProbe]) -> bool:
        if len(new_probes) != len(self._probes):
            return False
        if len(new_probes) != len(self._sparklines):
            return False
        for old, new in zip(self._probes, new_probes, strict=True):
            if old.subsystem_key != new.subsystem_key:
                return False
            if old.state is not new.state:
                return False
            if old.latency_ms != new.latency_ms:
                return False
        return True

    def _refresh_in_place(self) -> None:
        """Update sparkline cells without tearing the grid down.

        Probe identities + state are unchanged, so we only push the
        latest rolling history into each sparkline. Headers, pills,
        and labels keep their hover state and don't churn through
        `deleteLater()`.
        """

        for probe, sparkline in zip(self._probes, self._sparklines, strict=True):
            history = list(self._vm.probe_history(probe.subsystem_key))
            sparkline.set_cells(history)

    def _rebuild(self) -> None:
        """Repopulate the grid from `self._probes` end-to-end.

        Every widget owned by the matrix is removed and re-created.
        Surgically swapping rows produced phantom column widths from
        leftover layout state — easier and safer to lay it out fresh
        each time (≤4 rows + headers, so the cost is negligible).
        """

        # Detach every existing widget from the grid and queue it for
        # deletion. The grid itself stays so we don't have to wrestle
        # `QWidget.setLayout` with a previously-installed layout.
        while self._grid.count() > 0:
            item = self._grid.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        # Reset column stretch caches that QGridLayout would otherwise
        # remember from the prior population.
        for column in range(len(self._HEADERS) + 1):
            self._grid.setColumnStretch(column, 0)
            self._grid.setColumnMinimumWidth(column, 0)
        self._header_labels.clear()
        self._state_pills.clear()
        self._latency_labels.clear()
        self._name_labels.clear()
        self._sparklines.clear()
        self._empty_label = None

        compact = self._width_band is ResponsiveWidthBand.NARROW

        # Headers are always added so legacy callers/tests can introspect
        # `_header_labels`; we hide them in compact mode so they don't
        # take up visual space when the row layout collapses.
        for column, title in enumerate(self._HEADERS):
            label = QLabel(title, self)
            label.setObjectName("PanelSubtitle")
            self._grid.addWidget(label, 0, column)
            label.setVisible(not compact)
            self._header_labels.append(label)
        if not compact:
            self._grid.setColumnStretch(3, 1)  # sparkline column absorbs slack

        if not self._probes:
            empty = QLabel("No bounded probes reported", self)
            empty.setObjectName("PanelSubtitle")
            self._grid.addWidget(empty, 1, 0, 1, len(self._HEADERS))
            self._empty_label = empty
            return

        for index, probe in enumerate(self._probes):
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

            sparkline = ProbeSparkline(
                capacity=self._vm.probe_history_capacity(),
                parent=self,
            )
            history: list[ProbeSparklineCell] = list(self._vm.probe_history(probe.subsystem_key))
            sparkline.set_cells(history)
            sparkline.setToolTip(
                f"Last {self._vm.probe_history_capacity()} probe samples for "
                f"{probe.label or probe.subsystem_key}."
            )

            self._name_labels.append(name)
            self._state_pills.append(pill)
            self._latency_labels.append(latency)
            self._sparklines.append(sparkline)

            if compact:
                # Compact: name spans 2 cols on the first sub-row, pill +
                # latency on the right; sparkline on its own sub-row
                # underneath. Two grid rows per probe (offset by 1 for
                # the always-present header row) so a sparkline never
                # collides with the next probe's name.
                base = 1 + index * 2
                self._grid.addWidget(name, base, 0, 1, 2)
                self._grid.addWidget(pill, base, 2)
                self._grid.addWidget(latency, base, 3)
                self._grid.addWidget(sparkline, base + 1, 0, 1, 4)
            else:
                row = index + 1  # row 0 is the header row
                self._grid.addWidget(name, row, 0)
                self._grid.addWidget(pill, row, 1)
                self._grid.addWidget(latency, row, 2)
                self._grid.addWidget(sparkline, row, 3)


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


def _cloud_auth_kind(state: CloudAuthState) -> UiStatusKind:
    if state is CloudAuthState.SIGNED_IN:
        return UiStatusKind.OK
    if state in (CloudAuthState.SECRET_STORE_UNAVAILABLE, CloudAuthState.REFRESH_FAILED):
        return UiStatusKind.ERROR
    return UiStatusKind.WARN


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
