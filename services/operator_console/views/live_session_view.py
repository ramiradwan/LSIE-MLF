"""Live Session page — encounter timeline + reward explanation pane.

The operator-trust surface: every encounter row exposes the §7B reward
inputs the pipeline used (P90, semantic gate, gated reward,
n_frames_in_window, au12_baseline_pre). Selecting a row drops the full
explanation into the detail pane alongside the §4.C.4 physiology
freshness read for that segment.

The view never formats strings inline. All operator language comes
through `formatters.py`. All business logic — arm readback, stimulus
lifecycle, reward explanation, countdown arithmetic — lives in
`LiveSessionViewModel`; this file is layout + signal wiring only.

Spec references:
  §4.C           — `_active_arm`, `_expected_greeting`, authoritative
                   `_stimulus_time`; header reads from live-session DTO
                   never from the encounter rows, while the calibration
                   pill renders console safe-submit readiness
  §4.C.4         — physiology freshness badge in the detail pane
  §4.E.1         — Live Session operator surface
  §7B            — reward = p90_intensity × semantic_gate; detail pane
                   surfaces every input the pipeline used
  §12            — non-retryable errors surface on the page-level banner
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    EncounterState,
    EncounterSummary,
    SessionSummary,
    StimulusActionState,
    UiStatusKind,
)
from services.operator_console.formatters import (
    AcousticDetailDisplay,
    AcousticMetricCardDisplay,
    SemanticAttributionDiagnosticsDisplay,
    acoustic_section_labels,
    format_reward,
    format_semantic_confidence,
    format_semantic_gate,
    format_timestamp,
    truncate_expected_greeting,
)
from services.operator_console.viewmodels.live_session_vm import (
    LiveSessionViewModel,
    TtvSetupDisplay,
)
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

# §2 pipeline segments are 30s, which aligns with the VM's default §7B
# measurement-window length. 1-second tick is the rightful cadence for
# a human-facing countdown — any faster flickers, any slower reads slow.
_COUNTDOWN_TICK_MS: int = 1000
_LIVE_SESSION_BREAKPOINTS = ResponsiveBreakpoints(medium_min_width=720, wide_min_width=1040)
_NARROW_PHONE_PREVIEW_HEIGHT = 120
_WIDE_PHONE_PREVIEW_HEIGHT = 220

_ENCOUNTER_TABLE_POLICIES: tuple[TableColumnPolicy, ...] = (
    TableColumnPolicy(
        column=0,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 150},
    ),
    TableColumnPolicy(
        column=1,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 150},
    ),
    TableColumnPolicy(
        column=2,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.Stretch,
    ),
    TableColumnPolicy(
        column=3,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 130},
    ),
    TableColumnPolicy(
        column=4,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 120},
    ),
    TableColumnPolicy(
        column=5,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 120},
    ),
    TableColumnPolicy(
        column=6,
        visible_in=frozenset({ResponsiveWidthBand.MEDIUM, ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 135},
    ),
    TableColumnPolicy(
        column=7,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 110},
    ),
)

_TIMELINE_TABLE_POLICIES: tuple[TableColumnPolicy, ...] = (
    TableColumnPolicy(
        column=0,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 140},
    ),
    TableColumnPolicy(column=1, resize_mode=QHeaderView.ResizeMode.Stretch),
    TableColumnPolicy(
        column=2,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 90},
    ),
)


class LiveSessionView(QWidget):
    """Live Session page: header, encounter table, detail pane."""

    # Emitted when the operator selects a row — mostly for tests/shell;
    # the VM already tracks selection internally.
    encounter_selected = Signal(object)  # str | None

    def __init__(
        self,
        vm: LiveSessionViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Live Session",
            "Get from connected phone to first smile signal with transparent trust readback.",
            self,
        )
        self._session_panel = _SessionHeaderPanel(self)
        self._adb_pill = StatusPill(self)
        self._ml_pill = StatusPill(self)
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No session selected")
        self._empty_state.set_message(
            "Pick a session from Overview or Sessions, or start a new session here."
        )

        self._setup_overlay = _TtvSetupOverlay(self)
        self._phone_preview = _PhonePreviewPanel(self)
        self._smile_card = MetricCard("Smile Intensity", self)
        self._live_analytics_notice = AlertBanner(self)
        self._timeline_model = _LiveSessionTimelineModel(self)
        self._timeline = EventTimelineWidget(self)
        self._timeline.set_model(self._timeline_model)
        self._timeline.set_column_policies(
            _TIMELINE_TABLE_POLICIES,
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )
        self._table = self._build_table()
        self._detail_panel = _EncounterDetailPanel(self)

        header_status = QHBoxLayout()
        header_status.setContentsMargins(0, 0, 0, 0)
        header_status.setSpacing(16)
        header_status.addWidget(self._header, 1)
        header_status.addWidget(self._adb_pill)
        header_status.addWidget(self._ml_pill)

        self._dashboard_grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=2, medium=2, narrow=1),
            parent=self,
        )
        self._dashboard_grid.set_widgets(
            [
                self._phone_preview,
                self._smile_card,
                self._timeline,
            ]
        )

        self._trust_label = QLabel("Secondary trust data", self)
        self._trust_label.setObjectName("PanelTitle")
        trust_layout = QVBoxLayout()
        trust_layout.setContentsMargins(0, 0, 0, 0)
        trust_layout.setSpacing(10)
        trust_layout.addWidget(self._trust_label)
        trust_layout.addWidget(self._table, 2)
        trust_layout.addWidget(self._detail_panel, 3)

        self._body_container = QWidget(self)
        body = QVBoxLayout(self._body_container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addWidget(self._setup_overlay)
        body.addWidget(self._live_analytics_notice)
        body.addWidget(self._dashboard_grid, 2)
        body.addLayout(trust_layout, 3)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("LiveSessionScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._body_container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addLayout(header_status)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._session_panel)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._scroll, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(_COUNTDOWN_TICK_MS)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        self._session_panel.start_requested.connect(self._on_start_session_requested)
        self._session_panel.end_requested.connect(self._on_end_session_requested)

        # Subscriptions — the VM fans out all relevant store changes.
        self._vm.changed.connect(self._refresh)
        self._vm.state_changed.connect(self._on_state_changed)
        self._vm.error_changed.connect(self._on_error_changed)
        self._vm.selection_changed.connect(self._on_vm_selection_changed)
        self._vm.action_state_changed.connect(self._on_action_state_changed)

        self._refresh()

    # ------------------------------------------------------------------
    # Page lifecycle hooks
    # ------------------------------------------------------------------

    def on_activated(self) -> None:
        """Called by the shell when the page becomes visible."""
        self._refresh()
        # Kick the countdown timer only if a measurement window is live.
        self._sync_countdown_timer()

    def on_deactivated(self) -> None:
        """Stop the countdown tick while the page is hidden.

        The timer does nothing without a MEASURING stimulus anyway, but
        stopping it explicitly saves a per-second wakeup when the
        operator is on a different page.
        """
        self._countdown_timer.stop()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_table(self) -> QTableView:
        table = QTableView(self)
        table.setObjectName("EncounterTable")
        table.setModel(self._vm.encounters_model())
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vertical = table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        horizontal = table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setWordWrap(False)
        selection_model = table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_table_selection_changed)
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        session = self._vm.session()
        self._session_panel.set_session(
            session,
            active_arm=self._vm.active_arm(),
            expected_greeting=self._vm.expected_greeting(),
            calibration_status=self._vm.calibration_status(),
            start_enabled=self._vm.can_start_session(),
            end_enabled=self._vm.can_end_session(),
            start_in_progress=self._vm.session_start_in_progress(),
            end_in_progress=self._vm.session_end_in_progress(),
        )
        self._set_status_pill(self._adb_pill, self._vm.adb_status())
        self._set_status_pill(self._ml_pill, self._vm.ml_backend_status())
        setup_display = self._vm.ttv_setup_display()
        if setup_display.dashboard_mode == "gate":
            self._set_setup_gate(setup_display)
            self._scroll.setVisible(False)
            self._sync_countdown_timer()
            return
        self._empty_state.setVisible(False)
        self._scroll.setVisible(True)
        self._body_container.setVisible(True)
        self._set_setup_overlay(setup_display)
        self._set_dashboard_muted(setup_display.dashboard_mode == "calibrating")
        self._set_phone_preview_status(setup_display)
        live_analytics_notice = self._vm.live_analytics_notice()
        self._set_live_analytics_notice(live_analytics_notice)
        self._set_smile_card(self._vm.current_smile_intensity_percent(), live_analytics_notice)
        self._timeline_model.set_rows(self._vm.smile_timeline_points())
        self._timeline.scroll_to_latest()

        selected = self._vm.selected_encounter()
        if selected is None:
            # When no row is selected, fall back to the same encounter row
            # for subtitle, reward explanation, §7D acoustic details, and
            # §8/§7E diagnostics so the detail pane never mixes identities
            # while data exists.
            rows = self._vm.encounters_model()
            if rows.rowCount() > 0:
                selected = rows.row_at(rows.rowCount() - 1)
        self._detail_panel.set_encounter(
            selected,
            self._vm.reward_explanation_for_encounter(selected),
            self._vm.acoustic_detail_for_encounter(selected),
            self._vm.semantic_attribution_diagnostics_for_encounter(selected),
        )
        self._sync_countdown_timer()

    def _set_setup_gate(self, display: TtvSetupDisplay) -> None:
        self._empty_state.set_title(display.title)
        message = display.message
        if display.detail is not None:
            message = f"{message}\n\n{display.detail}"
        self._empty_state.set_message(message)
        self._empty_state.setVisible(True)
        self._setup_overlay.setVisible(False)
        self._set_dashboard_muted(False)
        self._set_phone_preview_status(display)

    def _set_setup_overlay(self, display: TtvSetupDisplay) -> None:
        if display.dashboard_mode == "ready":
            self._setup_overlay.setVisible(False)
            return
        self._setup_overlay.set_display(display)
        self._setup_overlay.setVisible(True)

    def _set_dashboard_muted(self, enabled: bool) -> None:
        widgets = (
            self._phone_preview,
            self._smile_card,
            self._timeline,
            self._trust_label,
            self._table,
            self._detail_panel,
        )
        for widget in widgets:
            widget.setEnabled(not enabled)
        self._body_container.setObjectName("ContentSurfaceMuted" if enabled else "")

    def _set_phone_preview_status(self, display: TtvSetupDisplay) -> None:
        if display.dashboard_mode == "calibrating":
            detail = f" {display.detail}." if display.detail is not None else ""
            self._phone_preview.set_status(
                "Raw phone frames are not shown. Face Tracking is calibrating from "
                f"the local capture stream.{detail}"
            )
            return
        if display.dashboard_mode == "ready":
            self._phone_preview.set_status(
                "Face locked. Smile Intensity updates after completed post-stimulus analytics."
            )
            return
        self._phone_preview.set_status(
            "Raw phone frames are not shown. Waiting for capture and face tracking."
        )

    def _set_status_pill(self, pill: StatusPill, value: object) -> None:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], UiStatusKind):
            pill.set_kind(value[0])
            pill.set_text(str(value[1]))
            return
        pill.set_kind(UiStatusKind.NEUTRAL)
        pill.set_text(str(value))

    def _set_live_analytics_notice(self, message: str | None) -> None:
        if message is None:
            self._live_analytics_notice.set_alert(None, None)
            return
        self._live_analytics_notice.set_alert(AlertSeverity.INFO, message)

    def _apply_responsive_layout(self) -> None:
        width = max(self.width(), self._scroll.viewport().width())
        band = _LIVE_SESSION_BREAKPOINTS.band_for_width(width)
        self._dashboard_grid.apply_width(width)
        apply_table_column_policies(
            self._table,
            _ENCOUNTER_TABLE_POLICIES,
            width=width,
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )
        self._timeline.apply_responsive_width(width)
        self._detail_panel.apply_responsive_width(width)
        self._phone_preview.set_compact(band is ResponsiveWidthBand.NARROW)

    def _set_smile_card(self, value: int | None, live_analytics_notice: str | None) -> None:
        if value is None:
            self._smile_card.set_primary_text("—")
            if live_analytics_notice is None:
                self._smile_card.set_secondary_text("Waiting for first valid smile window")
            else:
                self._smile_card.set_secondary_text("Waiting for completed analytics window")
            self._smile_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        self._smile_card.set_primary_text(f"{value}%")
        self._smile_card.set_secondary_text("Latest valid P90 AU12 smile window")
        self._smile_card.set_status(UiStatusKind.OK, "ready")

    def _sync_countdown_timer(self) -> None:
        """Start/stop the 1s countdown based on stimulus state."""
        state = self._vm.stimulus_ui_context().state
        if state == StimulusActionState.MEASURING:
            if not self._countdown_timer.isActive():
                self._countdown_timer.start()
            self._tick_countdown()
        else:
            if self._countdown_timer.isActive():
                self._countdown_timer.stop()
            self._detail_panel.set_countdown(None)

    def _tick_countdown(self) -> None:
        remaining = self._vm.measurement_window_remaining_s(datetime.now(UTC))
        self._detail_panel.set_countdown(remaining)
        if remaining == 0.0:
            # The VM will transition the context to COMPLETED on the
            # next encounters tick; stop busy-ticking in the meantime.
            self._countdown_timer.stop()

    # ------------------------------------------------------------------
    # Selection handling — table ↔ VM
    # ------------------------------------------------------------------

    def _on_table_selection_changed(self, *_: object) -> None:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        indexes = selection_model.selectedRows()
        if not indexes:
            self._vm.select_encounter(None)
            return
        row = indexes[0].row()
        encounter = self._vm.encounters_model().row_at(row)
        if encounter is None:
            self._vm.select_encounter(None)
            return
        self._vm.select_encounter(encounter.encounter_id)
        self.encounter_selected.emit(encounter.encounter_id)

    @Slot(object)
    def _on_vm_selection_changed(self, encounter_id: object) -> None:
        # Reflect a programmatic selection change (from the VM) back on
        # the table. Block the table's selection signal to avoid the
        # ping-pong back into `_on_table_selection_changed`.
        encounter_id_str = encounter_id if isinstance(encounter_id, str) else None
        if encounter_id_str is None:
            selection_model = self._table.selectionModel()
            if selection_model is not None:
                selection_model.clearSelection()
            return
        model = self._vm.encounters_model()
        index = model.index_of_encounter(encounter_id_str)
        if index is None:
            return
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        selection_model.blockSignals(True)
        try:
            self._table.selectRow(index)
        finally:
            selection_model.blockSignals(False)

    # ------------------------------------------------------------------
    # Error + action-state slots
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_state_changed(self, _state: str) -> None:
        self._refresh()

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)

    @Slot(object)
    def _on_action_state_changed(self, ctx: object) -> None:
        del ctx  # the current ctx is read via the VM getter
        # Entering MEASURING starts the countdown; leaving it stops.
        self._sync_countdown_timer()

    def _create_start_session_dialog(self) -> _StartSessionDialog:
        return _StartSessionDialog(self._vm.validate_start_session_inputs, self)

    @Slot()
    def _on_start_session_requested(self) -> None:
        dialog = self._create_start_session_dialog()
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        stream_url, experiment_id = dialog.values()
        self._vm.start_new_session(stream_url, experiment_id)

    @Slot()
    def _on_end_session_requested(self) -> None:
        self._vm.end_current_session()


# ----------------------------------------------------------------------
# Panels — small helper widgets kept private to this module
# ----------------------------------------------------------------------


class _LiveSessionTimelineModel(QAbstractTableModel):
    _HEADERS = ("Time", "Event", "Signal")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[object] = []

    def set_rows(self, rows: object) -> None:
        self.beginResetModel()
        self._rows = list(rows) if isinstance(rows, list | tuple) else []
        self.endResetModel()

    def rowCount(  # noqa: N802 — Qt override
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),  # noqa: B008
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(  # noqa: N802 — Qt override
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),  # noqa: B008
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._HEADERS)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        if index.column() == 0:
            timestamp = _timeline_value(row, "timestamp_utc")
            if isinstance(timestamp, datetime):
                return format_timestamp(timestamp)
            return str(timestamp) if timestamp is not None else "—"
        if index.column() == 1:
            return _timeline_text(row, "label")
        if index.column() == 2:
            intensity = _timeline_value(row, "intensity_percent")
            marker = _timeline_value(row, "marker")
            if intensity is not None:
                return f"{intensity}%"
            return str(marker) if marker is not None else "—"
        return None

    def headerData(  # noqa: N802 — Qt override
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._HEADERS[section]
        return None


class _TtvSetupOverlay(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._step_label = QLabel("", self)
        self._step_label.setObjectName("MetricCardSecondary")
        self._title = QLabel("", self)
        self._title.setObjectName("PanelTitle")
        self._message = QLabel("", self)
        self._message.setObjectName("MetricCardPrimary")
        self._message.setWordWrap(True)
        self._detail = QLabel("", self)
        self._detail.setObjectName("MetricCardSecondary")
        self._detail.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        layout.addWidget(self._step_label)
        layout.addWidget(self._title)
        layout.addWidget(self._message)
        layout.addWidget(self._detail)

    def set_display(self, display: TtvSetupDisplay) -> None:
        self._step_label.setText(display.step_label)
        self._title.setText(display.title)
        self._message.setText(display.message)
        self._detail.setText(display.detail or "")
        self._detail.setVisible(display.detail is not None)


class _PhonePreviewPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._title = QLabel("Live visual status", self)
        self._title.setObjectName("PanelTitle")
        self._placeholder = QLabel("Derived visual telemetry", self)
        self._placeholder.setObjectName("MetricCardPrimary")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setMinimumHeight(_WIDE_PHONE_PREVIEW_HEIGHT)
        self._status = QLabel(
            "Raw phone frames are not shown. Waiting for capture and face tracking.",
            self,
        )
        self._status.setObjectName("MetricCardSecondary")
        self._status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._placeholder, 1)
        layout.addWidget(self._status)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_compact(self, enabled: bool) -> None:
        self._placeholder.setMinimumHeight(
            _NARROW_PHONE_PREVIEW_HEIGHT if enabled else _WIDE_PHONE_PREVIEW_HEIGHT
        )


def _timeline_value(row: object, name: str) -> object:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _timeline_text(row: object, name: str) -> str:
    value = _timeline_value(row, name)
    return str(value) if value is not None else "—"


class _StartSessionDialog(QDialog):
    """Modal collecting the operator-provided stream target and experiment id."""

    def __init__(
        self,
        validator: Callable[[str, str], tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("StartSessionDialog")
        self.setModal(True)
        self.setWindowTitle("Start new session")
        self._validator = validator
        self._validated_stream_url: str | None = None
        self._validated_experiment_id: str | None = None

        self._stream_url_label = QLabel("Stream URL", self)
        self._stream_url_input = QLineEdit(self)
        self._stream_url_input.setObjectName("StartSessionStreamUrlInput")
        self._stream_url_input.setPlaceholderText("rtmp://example/live")

        self._experiment_id_label = QLabel("Experiment ID", self)
        self._experiment_id_input = QLineEdit(self)
        self._experiment_id_input.setObjectName("StartSessionExperimentIdInput")
        self._experiment_id_input.setPlaceholderText("greeting_line_v1")

        self._validation_label = QLabel("", self)
        self._validation_label.setObjectName("MetricCardSecondary")
        self._validation_label.setWordWrap(True)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self._start_button = self._buttons.addButton(
            "Start session",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._start_button.setObjectName("StartSessionSubmitButton")

        self._buttons.rejected.connect(self.reject)
        self._start_button.clicked.connect(self._on_submit_clicked)
        self._stream_url_input.textChanged.connect(self._revalidate)
        self._experiment_id_input.textChanged.connect(self._revalidate)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(self._stream_url_label)
        layout.addWidget(self._stream_url_input)
        layout.addWidget(self._experiment_id_label)
        layout.addWidget(self._experiment_id_input)
        layout.addWidget(self._validation_label)
        layout.addWidget(self._buttons)

        self._revalidate()

    def values(self) -> tuple[str, str]:
        self._revalidate()
        if self._validated_stream_url is None or self._validated_experiment_id is None:
            raise RuntimeError("start-session values requested while dialog is invalid")
        return self._validated_stream_url, self._validated_experiment_id

    def _revalidate(self) -> None:
        try:
            stream_url, experiment_id = self._validator(
                self._stream_url_input.text(),
                self._experiment_id_input.text(),
            )
        except ValueError as exc:
            self._validated_stream_url = None
            self._validated_experiment_id = None
            self._validation_label.setText(str(exc))
            self._validation_label.setVisible(True)
            self._start_button.setEnabled(False)
            return

        self._validated_stream_url = stream_url
        self._validated_experiment_id = experiment_id
        self._validation_label.setText("")
        self._validation_label.setVisible(False)
        self._start_button.setEnabled(True)

    def _on_submit_clicked(self) -> None:
        self._revalidate()
        if self._validated_stream_url is None or self._validated_experiment_id is None:
            return
        self.accept()


class _SessionHeaderPanel(QFrame):
    """Header row: session readback, calibration pill, and lifecycle controls.

    §4.C: arm and expected greeting come from the live-session DTO, not
    from any encounter row. The calibration pill receives the centralized
    console-readiness status. Start/end controls stay independent from
    calibration so AU12 readiness can only gate stimulus submission.
    """

    start_requested = Signal()
    end_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel("Session", self)
        self._title.setObjectName("PanelTitle")
        self._session_label = QLabel("No session selected", self)
        self._session_label.setObjectName("MetricCardPrimary")
        self._session_label.setWordWrap(True)
        self._arm_label = QLabel("", self)
        self._arm_label.setObjectName("MetricCardSecondary")
        self._arm_label.setWordWrap(True)
        self._greeting_label = QLabel("", self)
        self._greeting_label.setObjectName("MetricCardSecondary")
        self._greeting_label.setWordWrap(True)

        self._controls_label = QLabel("Session controls", self)
        self._controls_label.setObjectName("MetricCardSecondary")
        self._start_button = QPushButton("Start new session", self)
        self._start_button.setObjectName("SessionStartButton")
        self._end_button = QPushButton("End session", self)
        self._end_button.setObjectName("SessionEndButton")
        self._end_button.setVisible(False)

        self._calibration_pill = StatusPill(self)
        self._calibration_pill.set_kind(UiStatusKind.NEUTRAL)
        self._calibration_pill.set_text("No session")

        self._start_button.clicked.connect(lambda _checked=False: self.start_requested.emit())
        self._end_button.clicked.connect(lambda _checked=False: self.end_requested.emit())

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._controls_label)
        top.addWidget(self._start_button)
        top.addWidget(self._end_button)

        arm_row = QHBoxLayout()
        arm_row.setContentsMargins(0, 0, 0, 0)
        arm_row.setSpacing(12)
        arm_row.addWidget(self._arm_label)
        arm_row.addWidget(self._calibration_pill)
        arm_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._session_label)
        layout.addLayout(arm_row)
        layout.addWidget(self._greeting_label)

    def set_session(
        self,
        session: SessionSummary | None,
        *,
        active_arm: str | None,
        expected_greeting: str | None,
        calibration_status: tuple[UiStatusKind, str],
        start_enabled: bool,
        end_enabled: bool,
        start_in_progress: bool,
        end_in_progress: bool,
    ) -> None:
        if session is None:
            self._session_label.setText("No session selected")
            self._arm_label.setText("")
            self._arm_label.setVisible(False)
            self._greeting_label.setText("")
            self._greeting_label.setVisible(False)
            self._end_button.setVisible(False)
        else:
            self._session_label.setText(f"Session {session.session_id}")
            self._arm_label.setText(f"arm: {active_arm if active_arm else '—'}")
            self._arm_label.setVisible(True)
            if expected_greeting:
                truncated_greeting = truncate_expected_greeting(
                    expected_greeting,
                    limit=80,
                )
                self._greeting_label.setText(f"expected greeting: “{truncated_greeting}”")
                self._greeting_label.setVisible(True)
            else:
                self._greeting_label.setText("")
                self._greeting_label.setVisible(False)
            self._end_button.setVisible(session.ended_at_utc is None)

        self._start_button.setText("Starting…" if start_in_progress else "Start new session")
        self._end_button.setText("Ending…" if end_in_progress else "End session")
        self._start_button.setEnabled(start_enabled)
        self._end_button.setEnabled(end_enabled)

        kind, text = calibration_status
        self._calibration_pill.set_kind(kind)
        self._calibration_pill.set_text(text)


class _EncounterDetailPanel(QFrame):
    """Reward, §7D acoustic, and §8/§7E detail pane for one encounter.

    The existing §7B grid remains the first trust surface. The appended
    §7D and §8/§7E sections are observational only: validity, windowed
    means, semantic probabilities, and attribution diagnostics are displayed
    without coupling them to reward.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel("Encounter detail", self)
        self._title.setObjectName("PanelTitle")
        self._subtitle = QLabel("No encounter selected.", self)
        self._subtitle.setObjectName("PanelSubtitle")
        self._subtitle.setWordWrap(True)

        self._p90_card = MetricCard("P90 intensity", self)
        self._gate_card = MetricCard("Semantic gate", self)
        self._reward_card = MetricCard("Gated reward", self)
        self._frames_card = MetricCard("Frames in window", self)
        self._baseline_card = MetricCard("AU12 baseline pre", self)
        self._physiology_card = MetricCard("Physiology", self)

        self._reward_grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=3, medium=2, narrow=1),
            horizontal_spacing=10,
            vertical_spacing=10,
            parent=self,
        )
        self._reward_grid.set_widgets(
            [
                self._p90_card,
                self._gate_card,
                self._reward_card,
                self._frames_card,
                self._baseline_card,
                self._physiology_card,
            ]
        )

        self._explanation = QLabel("", self)
        self._explanation.setObjectName("MetricCardSecondary")
        self._explanation.setWordWrap(True)

        acoustic_labels = acoustic_section_labels()
        self._acoustic_title = QLabel(acoustic_labels.section_title, self)
        self._acoustic_title.setObjectName("PanelTitle")
        self._acoustic_empty = QLabel(acoustic_labels.empty_text, self)
        self._acoustic_empty.setObjectName("MetricCardSecondary")
        self._acoustic_empty.setWordWrap(True)

        self._f0_validity_pill = StatusPill(self)
        self._perturbation_validity_pill = StatusPill(self)
        validity_row = QHBoxLayout()
        validity_row.setContentsMargins(0, 0, 0, 0)
        validity_row.setSpacing(16)
        validity_row.addWidget(self._f0_validity_pill)
        validity_row.addWidget(self._perturbation_validity_pill)
        validity_row.addStretch(1)

        self._f0_mean_card = MetricCard(acoustic_labels.f0_metric_title, self)
        self._jitter_mean_card = MetricCard(acoustic_labels.jitter_metric_title, self)
        self._shimmer_mean_card = MetricCard(acoustic_labels.shimmer_metric_title, self)
        self._acoustic_grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=3, medium=2, narrow=1),
            horizontal_spacing=10,
            vertical_spacing=10,
            parent=self,
        )
        self._acoustic_grid.set_widgets(
            [self._f0_mean_card, self._jitter_mean_card, self._shimmer_mean_card]
        )

        self._voiced_coverage_label = QLabel("", self)
        self._voiced_coverage_label.setObjectName("MetricCardSecondary")
        self._voiced_coverage_label.setWordWrap(True)
        self._acoustic_explanation = QLabel("", self)
        self._acoustic_explanation.setObjectName("MetricCardSecondary")
        self._acoustic_explanation.setWordWrap(True)

        self._acoustic_metrics_container = QWidget(self)
        acoustic_layout = QVBoxLayout(self._acoustic_metrics_container)
        acoustic_layout.setContentsMargins(0, 0, 0, 0)
        acoustic_layout.setSpacing(8)
        acoustic_layout.addLayout(validity_row)
        acoustic_layout.addWidget(self._acoustic_grid)
        acoustic_layout.addWidget(self._voiced_coverage_label)
        acoustic_layout.addWidget(self._acoustic_explanation)

        self._semantic_title = QLabel("Semantic & Attribution (§8 / §7E)", self)
        self._semantic_title.setObjectName("PanelTitle")
        self._semantic_empty = QLabel("", self)
        self._semantic_empty.setObjectName("MetricCardSecondary")
        self._semantic_empty.setWordWrap(True)
        self._semantic_observational_note = QLabel("", self)
        self._semantic_observational_note.setObjectName("MetricCardSecondary")
        self._semantic_observational_note.setWordWrap(True)
        self._semantic_method_pill = StatusPill(self)
        self._semantic_match_pill = StatusPill(self)
        self._semantic_reason_label = QLabel("", self)
        self._semantic_reason_label.setObjectName("MetricCardSecondary")
        self._semantic_reason_label.setWordWrap(True)
        semantic_pill_row = QHBoxLayout()
        semantic_pill_row.setContentsMargins(0, 0, 0, 0)
        semantic_pill_row.setSpacing(16)
        semantic_pill_row.addWidget(self._semantic_method_pill)
        semantic_pill_row.addWidget(self._semantic_match_pill)
        semantic_pill_row.addStretch(1)

        self._confidence_card = MetricCard("Confidence score", self)
        self._attribution_finality_pill = StatusPill(self)
        self._soft_reward_card = MetricCard("soft_reward_candidate", self)
        self._au12_lifts_card = MetricCard("AU12 lifts", self)
        self._peak_latency_card = MetricCard("Peak latency", self)
        self._synchrony_card = MetricCard("Synchrony", self)
        self._outcome_link_lag_card = MetricCard("Outcome-link lag", self)

        self._semantic_grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=3, medium=2, narrow=1),
            horizontal_spacing=10,
            vertical_spacing=10,
            parent=self,
        )
        self._semantic_grid.set_widgets(
            [
                self._confidence_card,
                self._soft_reward_card,
                self._au12_lifts_card,
                self._peak_latency_card,
                self._synchrony_card,
                self._outcome_link_lag_card,
            ]
        )

        self._semantic_metrics_container = QWidget(self)
        semantic_layout = QVBoxLayout(self._semantic_metrics_container)
        semantic_layout.setContentsMargins(0, 0, 0, 0)
        semantic_layout.setSpacing(8)
        semantic_layout.addLayout(semantic_pill_row)
        semantic_layout.addWidget(self._semantic_reason_label)
        semantic_layout.addWidget(self._attribution_finality_pill)
        semantic_layout.addWidget(self._semantic_grid)

        self._countdown_label = QLabel("", self)
        self._countdown_label.setObjectName("ActionBarCountdown")
        self._countdown_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._reward_grid)
        layout.addWidget(self._explanation)
        layout.addWidget(self._acoustic_title)
        layout.addWidget(self._acoustic_empty)
        layout.addWidget(self._acoustic_metrics_container)
        layout.addWidget(self._semantic_title)
        layout.addWidget(self._semantic_empty)
        layout.addWidget(self._semantic_metrics_container)
        layout.addWidget(self._semantic_observational_note)
        layout.addWidget(self._countdown_label)

    def apply_responsive_width(self, width: int) -> ResponsiveWidthBand:
        band = self._reward_grid.apply_width(width)
        self._acoustic_grid.apply_width(width)
        self._semantic_grid.apply_width(width)
        return band

    def set_encounter(
        self,
        encounter: EncounterSummary | None,
        explanation: str,
        acoustic_detail: AcousticDetailDisplay,
        semantic_detail: SemanticAttributionDiagnosticsDisplay,
    ) -> None:
        if encounter is None:
            self._subtitle.setText("No encounter selected.")
            for card in (
                self._p90_card,
                self._gate_card,
                self._reward_card,
                self._frames_card,
                self._baseline_card,
                self._physiology_card,
            ):
                card.set_primary_text("—")
                card.set_secondary_text("")
                card.set_status(UiStatusKind.NEUTRAL, None)
            self._explanation.setText(explanation)
            self._set_acoustic(acoustic_detail)
            self._set_semantic_attribution(semantic_detail)
            return

        ts_text = format_timestamp(encounter.segment_timestamp_utc)
        self._subtitle.setText(
            f"Encounter {encounter.encounter_id} · {ts_text} · state {encounter.state.value}"
        )
        self._p90_card.set_primary_text(format_reward(encounter.p90_intensity))
        self._p90_card.set_secondary_text(
            f"confidence {format_semantic_confidence(encounter.semantic_confidence)}"
        )
        self._p90_card.set_status(UiStatusKind.INFO, None)

        self._gate_card.set_primary_text(
            str(encounter.semantic_gate) if encounter.semantic_gate is not None else "—"
        )
        self._gate_card.set_secondary_text(format_semantic_gate(encounter.semantic_gate))
        gate_status = (
            UiStatusKind.OK
            if encounter.semantic_gate == 1
            else (UiStatusKind.WARN if encounter.semantic_gate == 0 else UiStatusKind.NEUTRAL)
        )
        self._gate_card.set_status(gate_status, None)

        self._reward_card.set_primary_text(format_reward(encounter.gated_reward))
        reward_status = (
            UiStatusKind.OK if encounter.state == EncounterState.COMPLETED else UiStatusKind.NEUTRAL
        )
        self._reward_card.set_status(reward_status, encounter.state.value)
        self._reward_card.set_secondary_text("§7B r_t = p90 × gate")

        frames = encounter.n_frames_in_window
        self._frames_card.set_primary_text(str(frames) if frames is not None else "—")
        if frames is not None and frames == 0:
            self._frames_card.set_secondary_text("no valid AU12 frames — reward not computed")
            self._frames_card.set_status(UiStatusKind.WARN, None)
        else:
            self._frames_card.set_secondary_text("")
            self._frames_card.set_status(UiStatusKind.NEUTRAL, None)

        self._baseline_card.set_primary_text(format_reward(encounter.au12_baseline_pre))
        self._baseline_card.set_secondary_text("pre-stimulus AU12 baseline")
        self._baseline_card.set_status(UiStatusKind.NEUTRAL, None)

        if encounter.physiology_attached:
            if encounter.physiology_stale is True:
                self._physiology_card.set_primary_text("stale")
                self._physiology_card.set_status(UiStatusKind.WARN, "stale")
            else:
                self._physiology_card.set_primary_text("fresh")
                self._physiology_card.set_status(UiStatusKind.OK, "fresh")
            self._physiology_card.set_secondary_text("§4.C.4 snapshot attached")
        else:
            self._physiology_card.set_primary_text("absent")
            self._physiology_card.set_secondary_text("no snapshot for segment")
            self._physiology_card.set_status(UiStatusKind.NEUTRAL, "absent")

        self._explanation.setText(explanation)
        self._set_acoustic(acoustic_detail)
        self._set_semantic_attribution(semantic_detail)

    def _set_acoustic(self, detail: AcousticDetailDisplay) -> None:
        self._acoustic_title.setText(detail.section_title)
        self._acoustic_empty.setText(detail.empty_text)
        self._f0_validity_pill.set_kind(detail.f0_validity.status)
        self._f0_validity_pill.set_text(detail.f0_validity.text)
        self._perturbation_validity_pill.set_kind(detail.perturbation_validity.status)
        self._perturbation_validity_pill.set_text(detail.perturbation_validity.text)
        self._set_acoustic_card(self._f0_mean_card, detail.f0_mean)
        self._set_acoustic_card(self._jitter_mean_card, detail.jitter_mean)
        self._set_acoustic_card(self._shimmer_mean_card, detail.shimmer_mean)
        self._voiced_coverage_label.setText(detail.voiced_coverage_text)
        self._acoustic_explanation.setText(detail.explanation)

        self._acoustic_empty.setVisible(not detail.has_summary)
        self._acoustic_metrics_container.setVisible(detail.has_summary)

    def _set_semantic_attribution(
        self,
        detail: SemanticAttributionDiagnosticsDisplay,
    ) -> None:
        self._semantic_title.setText(detail.section_title)
        self._semantic_empty.setText(
            detail.empty_text if not detail.has_diagnostics else detail.attribution_empty_text
        )
        self._semantic_empty.setVisible(not detail.has_diagnostics or not detail.has_attribution)

        method_status = UiStatusKind.INFO if detail.has_semantic else UiStatusKind.NEUTRAL
        self._semantic_method_pill.set_kind(method_status)
        self._semantic_method_pill.set_text(f"method · {detail.semantic_method}")

        if detail.match_result == "match":
            match_status = UiStatusKind.OK
        elif detail.match_result == "non-match":
            match_status = UiStatusKind.WARN
        else:
            match_status = UiStatusKind.NEUTRAL
        self._semantic_match_pill.set_kind(match_status)
        self._semantic_match_pill.set_text(detail.match_result)
        self._semantic_reason_label.setText(f"Bounded reason code: {detail.bounded_reason_code}")

        if detail.attribution_finality == "offline final":
            finality_status = UiStatusKind.OK
        elif detail.has_attribution:
            finality_status = UiStatusKind.INFO
        else:
            finality_status = UiStatusKind.NEUTRAL
        self._attribution_finality_pill.set_kind(finality_status)
        self._attribution_finality_pill.set_text(
            f"attribution finality · {detail.attribution_finality}"
        )

        self._confidence_card.set_primary_text(detail.probability_confidence)
        self._confidence_card.set_secondary_text("§8 semantic probability estimate")
        self._confidence_card.set_status(method_status, None)

        self._soft_reward_card.set_primary_text(detail.soft_reward_candidate)
        self._soft_reward_card.set_secondary_text("observational candidate only")
        self._soft_reward_card.set_status(UiStatusKind.INFO, None)

        self._au12_lifts_card.set_primary_text(detail.au12_lift_metrics)
        self._au12_lifts_card.set_secondary_text("baseline-aware AU12 lift")
        self._au12_lifts_card.set_status(UiStatusKind.INFO, None)

        self._peak_latency_card.set_primary_text(detail.au12_peak_latency)
        self._peak_latency_card.set_secondary_text("stimulus→peak AU12")
        self._peak_latency_card.set_status(UiStatusKind.INFO, None)

        self._synchrony_card.set_primary_text(detail.synchrony_metrics)
        self._synchrony_card.set_secondary_text("lag-aware observational synchrony")
        self._synchrony_card.set_status(UiStatusKind.INFO, None)

        self._outcome_link_lag_card.set_primary_text(detail.outcome_link_lag)
        self._outcome_link_lag_card.set_secondary_text("event→outcome link")
        self._outcome_link_lag_card.set_status(UiStatusKind.INFO, None)

        self._confidence_card.setVisible(detail.has_semantic)
        self._attribution_finality_pill.setVisible(detail.has_attribution)
        for card in (
            self._soft_reward_card,
            self._au12_lifts_card,
            self._peak_latency_card,
            self._synchrony_card,
            self._outcome_link_lag_card,
        ):
            card.setVisible(detail.has_attribution)

        self._semantic_observational_note.setText(detail.observational_note)
        self._semantic_metrics_container.setVisible(detail.has_diagnostics)

    def _set_acoustic_card(
        self,
        card: MetricCard,
        display: AcousticMetricCardDisplay,
    ) -> None:
        card.set_primary_text(display.primary)
        card.set_secondary_text(display.secondary)
        card.set_status(display.status, display.status_text)

    def set_countdown(self, seconds: float | None) -> None:
        if seconds is None:
            self._countdown_label.setText("")
            self._countdown_label.setVisible(False)
            return
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        self._countdown_label.setText(f"Measurement window: {minutes:02d}:{secs:02d} remaining")
        self._countdown_label.setVisible(True)
