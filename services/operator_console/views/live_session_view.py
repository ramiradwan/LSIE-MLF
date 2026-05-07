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
    QSettings,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    EncounterState,
    EncounterSummary,
    ExperimentSummary,
    SessionSummary,
    StimulusActionState,
    UiStatusKind,
)
from services.operator_console.formatters import (
    AcousticDetailDisplay,
    AcousticMetricCardDisplay,
    CauseEffectDisplay,
    EncounterVerdictDisplay,
    LiveTelemetryDisplay,
    PhonePreviewStatusDisplay,
    ReadinessDisplay,
    SemanticAttributionDiagnosticsDisplay,
    acoustic_section_labels,
    build_encounter_verdict,
    format_phone_preview_status,
    format_reward,
    format_semantic_confidence,
    format_semantic_gate,
    format_timestamp,
    reward_detail_labels,
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

_COUNTDOWN_TICK_MS: int = 1000
_LIVE_SESSION_BREAKPOINTS = ResponsiveBreakpoints(medium_min_width=720, wide_min_width=1040)
_NARROW_PHONE_PREVIEW_HEIGHT = 120
_WIDE_PHONE_PREVIEW_HEIGHT = 220
_SPLITTER_SETTINGS_KEY = "live_session/operate_inspect_splitter"
_SPLITTER_DEFAULT_RATIO = (45, 55)

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
            "Connect the phone, send a stimulus, then watch the observed response.",
            self,
            level="page",
        )
        self._session_panel = _SessionHeaderPanel(self)
        self._readiness_strip = _ReadinessStrip(self)
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No session selected")
        self._empty_state.set_message(
            "Pick a session from Overview or Sessions, or start a new session here."
        )

        self._setup_overlay = _TtvSetupOverlay(self)
        self._readiness_panel = _ReadinessPanel(self)
        self._telemetry_panel = _LiveTelemetryPanel(self)
        self._phone_preview = _PhonePreviewPanel(self)
        self._smile_card = MetricCard("Response Signal", self)
        self._cause_effect_panel = _CauseEffectPanel(self)
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

        self._dashboard_grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=2, medium=2, narrow=1),
            parent=self,
        )
        self._dashboard_grid.set_widgets(
            [
                self._readiness_panel,
                self._telemetry_panel,
                self._phone_preview,
                self._smile_card,
            ]
        )

        self._trust_label = QLabel("Why this observed response counted", self)
        self._trust_label.setObjectName("PanelTitle")
        trust_layout = QVBoxLayout()
        trust_layout.setContentsMargins(0, 0, 0, 0)
        trust_layout.setSpacing(10)
        trust_layout.addWidget(self._trust_label)
        trust_layout.addWidget(self._table, 2)
        trust_layout.addWidget(self._detail_panel, 3)

        # Operate region — always-visible, glued to the top during a stimulus.
        self._operate_container = QWidget(self)
        operate_layout = QVBoxLayout(self._operate_container)
        operate_layout.setContentsMargins(0, 0, 0, 0)
        operate_layout.setSpacing(14)
        operate_layout.addWidget(self._setup_overlay)
        operate_layout.addWidget(self._live_analytics_notice)
        operate_layout.addWidget(self._dashboard_grid)
        operate_layout.addWidget(self._cause_effect_panel)
        operate_layout.addWidget(self._timeline)
        operate_layout.addStretch(1)

        # Inspect region — encounter table + drill-down detail. Lives in
        # its own scroll region so history scrolling never pushes the
        # live readback off the viewport.
        self._inspect_container = QWidget(self)
        inspect_layout = QVBoxLayout(self._inspect_container)
        inspect_layout.setContentsMargins(0, 0, 0, 0)
        inspect_layout.setSpacing(8)
        inspect_layout.addLayout(trust_layout)

        self._operate_scroll = QScrollArea(self)
        self._operate_scroll.setObjectName("LiveSessionOperateScroll")
        self._operate_scroll.setWidgetResizable(True)
        self._operate_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._operate_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._operate_scroll.setWidget(self._operate_container)

        self._inspect_scroll = QScrollArea(self)
        self._inspect_scroll.setObjectName("LiveSessionInspectScroll")
        self._inspect_scroll.setWidgetResizable(True)
        self._inspect_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inspect_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inspect_scroll.setWidget(self._inspect_container)

        # Wrap the two regions in a vertical splitter so the operator can
        # decide how much of the page is live readback versus history;
        # default ratio is 45/55 and the position is persisted via QSettings.
        self._splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._splitter.setObjectName("LiveSessionSplitter")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._operate_scroll)
        self._splitter.addWidget(self._inspect_scroll)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        self._scroll = self._operate_scroll  # backward-compat for tests/visibility checks
        self._body_container = self._operate_container
        self._restore_splitter()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._readiness_strip)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._session_panel)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._splitter, 1)
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
        self._readiness_strip.set_statuses(
            adb=self._vm.adb_status(),
            ml=self._vm.ml_backend_status(),
            audio=self._vm.audio_capture_status(),
            video=self._vm.video_capture_status(),
            calibration=self._vm.calibration_status(),
        )
        setup_display = self._vm.ttv_setup_display()
        if setup_display.dashboard_mode == "gate":
            self._set_setup_gate(setup_display)
            self._splitter.setVisible(False)
            self._operate_scroll.setVisible(False)
            self._inspect_scroll.setVisible(False)
            self._sync_countdown_timer()
            return
        self._empty_state.setVisible(False)
        self._splitter.setVisible(True)
        self._operate_scroll.setVisible(True)
        self._inspect_scroll.setVisible(True)
        self._operate_container.setVisible(True)
        self._set_setup_overlay(setup_display)
        self._set_dashboard_muted(setup_display.dashboard_mode == "calibrating")
        self._set_phone_preview_status(setup_display)
        live_analytics_notice = self._vm.live_analytics_notice()
        self._set_live_analytics_notice(live_analytics_notice)
        self._readiness_panel.set_display(self._vm.readiness_display())
        self._telemetry_panel.set_display(self._vm.live_telemetry_display())
        self._set_smile_card(self._vm.current_smile_intensity_percent(), live_analytics_notice)
        self._cause_effect_panel.set_display(self._vm.cause_effect_display())
        self._timeline_model.set_rows(self._vm.smile_timeline_points())
        self._timeline.scroll_to_latest()

        selected = self._vm.selected_encounter()
        if selected is None:
            rows = self._vm.encounters_model()
            if rows.rowCount() > 0:
                selected = rows.row_at(0)
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
            self._readiness_panel,
            self._telemetry_panel,
            self._phone_preview,
            self._smile_card,
            self._cause_effect_panel,
            self._timeline,
            self._trust_label,
            self._table,
            self._detail_panel,
        )
        for widget in widgets:
            widget.setEnabled(not enabled)
        self._body_container.setObjectName("ContentSurfaceMuted" if enabled else "")

    def _set_phone_preview_status(self, display: TtvSetupDisplay) -> None:
        # UX-12: positive readback first, caveat secondary. The operator
        # reads "what is" before "what isn't" so a healthy bar reads as a
        # confidence cue, not a disclaimer.
        status_display = format_phone_preview_status(
            dashboard_mode=display.dashboard_mode,
            detail=display.detail,
        )
        self._phone_preview.set_status_display(status_display)

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

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        settings = QSettings("LSIE-MLF", "OperatorConsole")
        settings.setValue(_SPLITTER_SETTINGS_KEY, self._splitter.saveState())

    def _restore_splitter(self) -> None:
        settings = QSettings("LSIE-MLF", "OperatorConsole")
        saved = settings.value(_SPLITTER_SETTINGS_KEY)
        if saved is not None:
            try:
                self._splitter.restoreState(saved)
                return
            except (TypeError, ValueError):
                pass
        # First run: anchor to 45/55 like the spec recommends.
        operate, inspect = _SPLITTER_DEFAULT_RATIO
        total = max(operate + inspect, 1)
        self._splitter.setSizes([operate * 1000 // total, inspect * 1000 // total])

    def _apply_responsive_layout(self) -> None:
        viewport_width = self._scroll.viewport().width()
        width = viewport_width if viewport_width >= 320 else self.width()
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
        self._cause_effect_panel.apply_responsive_width(width)

    def _set_smile_card(self, value: int | None, live_analytics_notice: str | None) -> None:
        if value is None:
            self._smile_card.set_primary_text("—")
            if live_analytics_notice is None:
                self._smile_card.set_secondary_text("Waiting for first result")
            else:
                self._smile_card.set_secondary_text("Waiting for first result")
            self._smile_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        self._smile_card.set_primary_text(f"{value}%")
        self._smile_card.set_secondary_text(
            "Strongest observed response signal from the latest usable window"
        )
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

    def _tick_countdown(self) -> None:
        remaining = self._vm.measurement_window_remaining_s(datetime.now(UTC))
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
        return _StartSessionDialog(
            source_summary=self._vm.start_session_source_summary(),
            summaries=self._vm.experiment_summaries(),
            current_experiment_id=self._vm.current_experiment_id(),
            disabled_reason=self._vm.start_session_disabled_reason(),
            validator=self._vm.validate_start_session_inputs,
            parent=self,
        )

    @Slot()
    def _on_start_session_requested(self) -> None:
        dialog = self._create_start_session_dialog()
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        experiment_id = dialog.values()
        self._vm.start_new_session(experiment_id)

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


class _ReadinessPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Live Session next step")
        self.setAccessibleDescription("Shows whether the operator can send the next stimulus.")
        self._title = QLabel("Next step", self)
        self._title.setObjectName("PanelTitle")
        self._status = StatusPill(self)
        self._primary = QLabel("", self)
        self._primary.setObjectName("MetricCardPrimary")
        self._primary.setWordWrap(True)
        self._detail = QLabel("", self)
        self._detail.setObjectName("MetricCardSecondary")
        self._detail.setWordWrap(True)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(self._primary)
        layout.addWidget(self._detail)

    def set_display(self, display: ReadinessDisplay) -> None:
        self._title.setText(display.title)
        self._status.set_kind(display.status)
        self._status.set_text(display.status.value)
        self._primary.setText(display.primary)
        self._detail.setText(display.detail)


class _LiveTelemetryPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Live response ticker")
        self.setAccessibleDescription("Shows current response measurement status.")
        self._title = QLabel("Live response ticker", self)
        self._title.setObjectName("PanelTitle")
        self._status = StatusPill(self)
        self._headline = QLabel("", self)
        self._headline.setObjectName("MetricCardPrimary")
        self._headline.setWordWrap(True)
        self._signal = QLabel("", self)
        self._signal.setObjectName("MetricCardPrimary")
        self._detail = QLabel("", self)
        self._detail.setObjectName("MetricCardSecondary")
        self._detail.setWordWrap(True)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._status)

        signal_row = QHBoxLayout()
        signal_row.setContentsMargins(0, 0, 0, 0)
        signal_row.setSpacing(12)
        signal_row.addWidget(self._headline, 1)
        signal_row.addWidget(self._signal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addLayout(signal_row)
        layout.addWidget(self._detail)

    def set_display(self, display: LiveTelemetryDisplay) -> None:
        self._status.set_kind(display.status)
        self._status.set_text(display.status.value)
        self._headline.setText(display.headline)
        self._signal.setText(display.response_signal)
        self._detail.setText(display.detail)


class _CauseEffectPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Observed response")
        self.setAccessibleDescription("Summarizes the result of the latest stimulus window.")
        self._title = QLabel("Observed response", self)
        self._title.setObjectName("PanelTitle")
        self._status = StatusPill(self)
        self._headline = QLabel("", self)
        self._headline.setObjectName("MetricCardPrimary")
        self._headline.setWordWrap(True)
        self._detail = QLabel("", self)
        self._detail.setObjectName("MetricCardSecondary")
        self._detail.setWordWrap(True)
        self._voice_card = MetricCard("Voice response", self)
        self._technical_card = MetricCard("Why it counted", self)
        self._grid = ResponsiveMetricGrid(
            breakpoints=_LIVE_SESSION_BREAKPOINTS,
            columns=MetricGridColumns(wide=2, medium=2, narrow=1),
            horizontal_spacing=10,
            vertical_spacing=10,
            parent=self,
        )
        self._grid.set_widgets([self._voice_card, self._technical_card])

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(self._headline)
        layout.addWidget(self._detail)
        layout.addWidget(self._grid)

    def set_display(self, display: CauseEffectDisplay) -> None:
        self._status.set_kind(display.status)
        self._status.set_text(display.status.value)
        self._headline.setText(display.headline)
        self._detail.setText(display.detail)
        self._voice_card.set_primary_text(display.voice_summary)
        self._voice_card.set_secondary_text("derived from stable voice coverage when available")
        self._voice_card.set_status(UiStatusKind.INFO, None)
        self._technical_card.set_primary_text(display.technical_summary)
        self._technical_card.set_secondary_text("details remain available below")
        self._technical_card.set_status(UiStatusKind.NEUTRAL, None)

    def apply_responsive_width(self, width: int) -> ResponsiveWidthBand:
        return self._grid.apply_width(width)


class _PhonePreviewPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Live visual status")
        self.setAccessibleDescription("Confirms that only derived visual telemetry is shown.")
        self._title = QLabel("Live visual status", self)
        self._title.setObjectName("PanelTitle")
        self._placeholder = QLabel("Derived visual telemetry", self)
        self._placeholder.setObjectName("MetricCardPrimary")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setMinimumHeight(_WIDE_PHONE_PREVIEW_HEIGHT)
        self._status = QLabel(
            "Awaiting capture · connect the phone to start live analysis.",
            self,
        )
        self._status.setObjectName("MetricCardPrimary")
        self._status.setWordWrap(True)
        self._caveat = QLabel("Raw phone frames are not shown.", self)
        self._caveat.setObjectName("MetricCardSecondary")
        self._caveat.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._placeholder, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._caveat)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_status_display(self, display: PhonePreviewStatusDisplay) -> None:
        self._status.setText(display.primary)
        self._caveat.setText(display.caveat)

    def set_compact(self, enabled: bool) -> None:
        self._placeholder.setMinimumHeight(
            _NARROW_PHONE_PREVIEW_HEIGHT if enabled else _WIDE_PHONE_PREVIEW_HEIGHT
        )


class _ReadinessStrip(QFrame):
    """Thin strip above the page header summarising readiness signals.

    Lifts ADB / ML / capture / safe-submit out of the title row so the
    operator picks up the worst-case severity peripherally — when any
    one signal flips off-OK the whole strip tints, no re-reading pills.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LiveSessionReadinessStrip")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(28)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAccessibleName("Readiness strip")
        self.setAccessibleDescription(
            "Compact readiness summary across phone tether, live analysis, "
            "capture, and submit-safe."
        )
        self.setProperty("severity", "")

        self._adb_pill = StatusPill(self)
        self._ml_pill = StatusPill(self)
        self._audio_pill = StatusPill(self)
        self._video_pill = StatusPill(self)
        self._submit_pill = StatusPill(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(18)
        layout.addWidget(self._adb_pill)
        layout.addWidget(self._ml_pill)
        layout.addWidget(self._audio_pill)
        layout.addWidget(self._video_pill)
        layout.addStretch(1)
        layout.addWidget(self._submit_pill)

    def set_statuses(
        self,
        *,
        adb: tuple[UiStatusKind, str],
        ml: tuple[UiStatusKind, str],
        audio: tuple[UiStatusKind, str],
        video: tuple[UiStatusKind, str],
        calibration: tuple[UiStatusKind, str],
    ) -> None:
        for pill, status in (
            (self._adb_pill, adb),
            (self._ml_pill, ml),
            (self._audio_pill, audio),
            (self._video_pill, video),
            (self._submit_pill, calibration),
        ):
            pill.set_kind(status[0])
            pill.set_text(status[1])
        worst = _worst_severity(adb[0], ml[0], audio[0], video[0], calibration[0])
        self.setProperty("severity", worst)
        self.style().unpolish(self)
        self.style().polish(self)


_SEVERITY_RANK: dict[UiStatusKind, int] = {
    UiStatusKind.ERROR: 4,
    UiStatusKind.WARN: 3,
    UiStatusKind.PROGRESS: 2,
    UiStatusKind.INFO: 1,
    UiStatusKind.NEUTRAL: 0,
    UiStatusKind.MUTED: 0,
    UiStatusKind.OK: 0,
}

_SEVERITY_PROP: dict[UiStatusKind, str] = {
    UiStatusKind.ERROR: "error",
    UiStatusKind.WARN: "warn",
    UiStatusKind.PROGRESS: "recovering",
    UiStatusKind.INFO: "",
    UiStatusKind.NEUTRAL: "",
    UiStatusKind.MUTED: "",
    UiStatusKind.OK: "",
}


def _worst_severity(*kinds: UiStatusKind) -> str:
    worst = max(kinds, key=lambda kind: _SEVERITY_RANK.get(kind, 0), default=UiStatusKind.OK)
    return _SEVERITY_PROP.get(worst, "")


def _timeline_value(row: object, name: str) -> object:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _timeline_text(row: object, name: str) -> str:
    value = _timeline_value(row, name)
    return str(value) if value is not None else "—"


class _StartSessionDialog(QDialog):
    """Modal for choosing the experiment for the connected-phone capture."""

    def __init__(
        self,
        *,
        source_summary: str,
        summaries: list[ExperimentSummary],
        current_experiment_id: str | None,
        disabled_reason: str | None,
        validator: Callable[[str], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("StartSessionDialog")
        self.setModal(True)
        self.setWindowTitle("Start new session")
        self.setAccessibleName("Start new session")
        self.setAccessibleDescription(
            "Choose the experiment to use for the connected-phone capture."
        )
        self._validator = validator
        self._disabled_reason = disabled_reason
        self._validated_experiment_id: str | None = None

        self._source_label = QLabel("Source", self)
        self._source_label.setObjectName("MetricCardSecondary")
        self._source_summary = QLabel(source_summary, self)
        self._source_summary.setObjectName("MetricCardPrimary")
        self._source_summary.setWordWrap(True)
        self._source_summary.setAccessibleName("Session source")
        self._source_summary.setAccessibleDescription(source_summary)

        self._experiment_label = QLabel("Experiment", self)
        self._experiment_picker = QComboBox(self)
        self._experiment_picker.setObjectName("StartSessionExperimentPicker")
        self._experiment_picker.setAccessibleName("Experiment")
        self._experiment_picker.setAccessibleDescription(
            "Select the experiment that provides the stimulus strategies for this session."
        )
        self._experiment_picker.setToolTip("Choose which experiment to run for this session.")
        self._experiment_label.setBuddy(self._experiment_picker)
        for summary in summaries:
            label = summary.label or summary.experiment_id
            self._experiment_picker.addItem(label, summary.experiment_id)
        if current_experiment_id is not None:
            index = self._experiment_picker.findData(current_experiment_id)
            if index >= 0:
                self._experiment_picker.setCurrentIndex(index)

        self._validation_label = QLabel(disabled_reason or "", self)
        self._validation_label.setObjectName("MetricCardSecondary")
        self._validation_label.setWordWrap(True)
        self._validation_label.setAccessibleName("Start-session status")
        self._validation_label.setAccessibleDescription(disabled_reason or "Ready to start.")

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self._start_button = self._buttons.addButton(
            "Start session",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._start_button.setObjectName("StartSessionSubmitButton")
        self._start_button.setAccessibleName("Start session")
        self._start_button.setAccessibleDescription(
            "Starts a new capture session with the selected experiment."
        )
        self._start_button.setToolTip("Start a new capture session with this experiment.")

        self._buttons.rejected.connect(self.reject)
        self._start_button.clicked.connect(self._on_submit_clicked)
        self._experiment_picker.currentIndexChanged.connect(self._revalidate)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(self._source_label)
        layout.addWidget(self._source_summary)
        layout.addWidget(self._experiment_label)
        layout.addWidget(self._experiment_picker)
        layout.addWidget(self._validation_label)
        layout.addWidget(self._buttons)

        if self._disabled_reason is not None:
            self._experiment_picker.setEnabled(False)
            self._start_button.setEnabled(False)
            self._validation_label.setVisible(True)
            return

        self._revalidate()

    def values(self) -> str:
        self._revalidate()
        if self._validated_experiment_id is None:
            raise RuntimeError("start-session values requested while dialog is invalid")
        return self._validated_experiment_id

    def _revalidate(self) -> None:
        try:
            experiment_id = self._validator(str(self._experiment_picker.currentData() or ""))
        except ValueError as exc:
            self._validated_experiment_id = None
            message = self._disabled_reason or str(exc)
            self._validation_label.setText(message)
            self._validation_label.setAccessibleDescription(message)
            self._validation_label.setVisible(True)
            self._start_button.setEnabled(False)
            return

        self._validated_experiment_id = experiment_id
        self._validation_label.setText("")
        self._validation_label.setAccessibleDescription("Ready to start.")
        self._validation_label.setVisible(False)
        self._start_button.setEnabled(True)

    def _on_submit_clicked(self) -> None:
        self._revalidate()
        if self._validated_experiment_id is None:
            return
        self.accept()


class _SessionHeaderPanel(QFrame):
    """Header row: session readback, calibration pill, and lifecycle controls.

    §4.C: stimulus strategy and expected response text come from the
    live-session DTO, not from any encounter row. The calibration pill receives the centralized
    console-readiness status. Start/end controls stay independent from
    calibration so AU12 readiness can only gate stimulus submission.
    """

    start_requested = Signal()
    end_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAccessibleName("Session controls")
        self.setAccessibleDescription("Shows the active session and start/end controls.")

        self._title = QLabel("Session", self)
        self._title.setObjectName("PanelTitle")
        self._session_label = QLabel("No session selected", self)
        self._session_label.setObjectName("MetricCardPrimary")
        self._session_label.setWordWrap(True)
        self._session_meta_label = QLabel("", self)
        self._session_meta_label.setObjectName("MetricCardSecondary")
        self._session_meta_label.setWordWrap(True)
        self._start_button = QPushButton("Start new session", self)
        self._start_button.setObjectName("SessionStartButton")
        self._start_button.setAccessibleName("Start new session")
        self._start_button.setAccessibleDescription(
            "Open the start-session dialog for the connected capture source."
        )
        self._start_button.setToolTip("Start a new session from the connected phone.")
        self._end_button = QPushButton("End session", self)
        self._end_button.setObjectName("SessionEndButton")
        self._end_button.setAccessibleName("End session")
        self._end_button.setAccessibleDescription("End the currently active capture session.")
        self._end_button.setToolTip("End the current session.")
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
        top.addWidget(self._start_button)
        top.addWidget(self._end_button)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(12)
        status_row.addWidget(self._session_meta_label)
        status_row.addWidget(self._calibration_pill)
        status_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._session_label)
        layout.addLayout(status_row)

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
        del active_arm, expected_greeting
        if session is None:
            self._session_label.setText("No session selected")
            self._session_label.setAccessibleDescription("No session is selected.")
            self._session_meta_label.setText("")
            self._session_meta_label.setVisible(False)
            self._end_button.setVisible(False)
        else:
            session_text = f"Session {session.session_id}"
            meta_text = f"Started {format_timestamp(session.started_at_utc)} · {session.status}"
            self._session_label.setText(session_text)
            self._session_label.setAccessibleDescription(session_text)
            self._session_meta_label.setText(meta_text)
            self._session_meta_label.setAccessibleDescription(meta_text)
            self._session_meta_label.setVisible(True)
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

        self._title = QLabel("Stimulus-response detail", self)
        self._title.setObjectName("PanelTitle")
        self._subtitle = QLabel("No encounter selected.", self)
        self._subtitle.setObjectName("PanelSubtitle")
        self._subtitle.setWordWrap(True)

        # UX-11: Verdict line first — operators ask "did this matter?"
        # before the math. Status pill + reason sentence; the existing
        # reward grid stays below as the indented input definition list.
        self._verdict_pill = StatusPill(self)
        self._verdict_headline = QLabel("", self)
        self._verdict_headline.setObjectName("EncounterVerdictHeadline")
        self._verdict_headline.setWordWrap(True)
        self._verdict_reason = QLabel("", self)
        self._verdict_reason.setObjectName("EncounterVerdictReason")
        self._verdict_reason.setWordWrap(True)
        verdict_row = QHBoxLayout()
        verdict_row.setContentsMargins(0, 0, 0, 0)
        verdict_row.setSpacing(10)
        verdict_row.addWidget(self._verdict_pill)
        verdict_row.addWidget(self._verdict_headline, 1)
        self._verdict_container = QWidget(self)
        verdict_layout = QVBoxLayout(self._verdict_container)
        verdict_layout.setContentsMargins(0, 0, 0, 4)
        verdict_layout.setSpacing(2)
        verdict_layout.addLayout(verdict_row)
        verdict_layout.addWidget(self._verdict_reason)

        # Inputs definition list — muted, monospace, indented. Provides
        # the "show your work" view without competing with the verdict.
        self._inputs_list = QWidget(self)
        self._inputs_form = QFormLayout(self._inputs_list)
        self._inputs_form.setContentsMargins(16, 4, 0, 8)
        self._inputs_form.setSpacing(4)
        self._inputs_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._inputs_list.setObjectName("EncounterInputsList")

        reward_labels = reward_detail_labels()
        self._p90_card = MetricCard(reward_labels.p90_title, self)
        self._gate_card = MetricCard(reward_labels.gate_title, self)
        self._reward_card = MetricCard(reward_labels.reward_title, self)
        self._frames_card = MetricCard(reward_labels.frames_title, self)
        self._baseline_card = MetricCard(reward_labels.baseline_title, self)
        self._physiology_card = MetricCard(reward_labels.physiology_title, self)

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

        self._transcription_title = QLabel("Speech-to-text", self)
        self._transcription_title.setObjectName("PanelTitle")
        self._transcription_text = QLabel("", self)
        self._transcription_text.setObjectName("MetricCardSecondary")
        self._transcription_text.setWordWrap(True)

        self._semantic_title = QLabel("Stimulus confirmation and follow-up signals", self)
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

        self._confidence_card = MetricCard("Stimulus confirmation confidence", self)
        self._attribution_finality_pill = StatusPill(self)
        self._soft_reward_card = MetricCard("Possible follow-up reward", self)
        self._au12_lifts_card = MetricCard("Response lift after stimulus", self)
        self._peak_latency_card = MetricCard("Time to strongest response", self)
        self._synchrony_card = MetricCard("Movement together", self)
        self._outcome_link_lag_card = MetricCard("Time to follow-up outcome", self)

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._verdict_container)
        layout.addWidget(self._inputs_list)
        layout.addWidget(self._reward_grid)
        layout.addWidget(self._explanation)
        layout.addWidget(self._acoustic_title)
        layout.addWidget(self._acoustic_empty)
        layout.addWidget(self._acoustic_metrics_container)
        layout.addWidget(self._transcription_title)
        layout.addWidget(self._transcription_text)
        layout.addWidget(self._semantic_title)
        layout.addWidget(self._semantic_empty)
        layout.addWidget(self._semantic_metrics_container)
        layout.addWidget(self._semantic_observational_note)

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
        verdict = build_encounter_verdict(encounter)
        self._set_verdict(verdict)
        self._set_inputs_definition_list(encounter)
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
            self._set_transcription(None)
            self._set_acoustic(acoustic_detail)
            self._set_semantic_attribution(semantic_detail)
            return

        ts_text = format_timestamp(encounter.segment_timestamp_utc)
        self._subtitle.setText(
            f"Encounter {encounter.encounter_id} · {ts_text} · state {encounter.state.value}"
        )
        self._p90_card.set_primary_text(format_reward(encounter.p90_intensity))
        confidence = format_semantic_confidence(encounter.semantic_confidence)
        self._p90_card.set_secondary_text(f"stimulus confirmation confidence {confidence}")
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
        self._reward_card.set_secondary_text(reward_detail_labels().reward_formula)

        frames = encounter.n_frames_in_window
        self._frames_card.set_primary_text(str(frames) if frames is not None else "—")
        if frames is not None and frames == 0:
            self._frames_card.set_secondary_text("no usable face frames — reward not computed")
            self._frames_card.set_status(UiStatusKind.WARN, None)
        else:
            self._frames_card.set_secondary_text("first 4.5 seconds after stimulus")
            self._frames_card.set_status(UiStatusKind.NEUTRAL, None)

        self._baseline_card.set_primary_text(format_reward(encounter.au12_baseline_pre))
        self._baseline_card.set_secondary_text("response-signal level before the stimulus")
        self._baseline_card.set_status(UiStatusKind.NEUTRAL, None)

        if encounter.physiology_attached:
            if encounter.physiology_stale is True:
                self._physiology_card.set_primary_text("old")
                self._physiology_card.set_status(UiStatusKind.WARN, "stale")
            else:
                self._physiology_card.set_primary_text("fresh")
                self._physiology_card.set_status(UiStatusKind.OK, "fresh")
            self._physiology_card.set_secondary_text("heart data attached for this segment")
        else:
            self._physiology_card.set_primary_text("absent")
            self._physiology_card.set_secondary_text("no heart data for this segment")
            self._physiology_card.set_status(UiStatusKind.NEUTRAL, "absent")

        self._explanation.setText(explanation)
        self._set_transcription(encounter.transcription)
        self._set_acoustic(acoustic_detail)
        self._set_semantic_attribution(semantic_detail)

    def _set_verdict(self, verdict: EncounterVerdictDisplay) -> None:
        self._verdict_pill.set_kind(verdict.status)
        self._verdict_pill.set_text(verdict.status.value)
        self._verdict_headline.setText(verdict.headline)
        self._verdict_reason.setText(verdict.reason)
        self._verdict_pill.setAccessibleName(verdict.headline)
        self._verdict_pill.setAccessibleDescription(verdict.reason)

    def _set_inputs_definition_list(self, encounter: EncounterSummary | None) -> None:
        # Clear and rebuild — operator inputs change every selection.
        while self._inputs_form.count() > 0:
            item = self._inputs_form.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if encounter is None:
            return
        rows = (
            ("p90 intensity", format_reward(encounter.p90_intensity)),
            (
                "semantic gate",
                str(encounter.semantic_gate) if encounter.semantic_gate is not None else "—",
            ),
            (
                "frames in window",
                str(encounter.n_frames_in_window)
                if encounter.n_frames_in_window is not None
                else "—",
            ),
            ("au12 baseline pre", format_reward(encounter.au12_baseline_pre)),
            ("gated reward", format_reward(encounter.gated_reward)),
        )
        for key, value in rows:
            key_label = QLabel(key, self._inputs_list)
            key_label.setObjectName("EncounterInputsLabel")
            value_label = QLabel(value, self._inputs_list)
            value_label.setObjectName("EncounterInputsLabel")
            self._inputs_form.addRow(key_label, value_label)

    def _set_transcription(self, transcription: str | None) -> None:
        text = transcription.strip() if transcription is not None else ""
        self._transcription_text.setText(text or "No spoken response was captured in this window.")

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
        self._semantic_method_pill.set_text(f"checker · {detail.semantic_method}")

        if detail.match_result == "match":
            match_status = UiStatusKind.OK
        elif detail.match_result == "non-match":
            match_status = UiStatusKind.WARN
        else:
            match_status = UiStatusKind.NEUTRAL
        self._semantic_match_pill.set_kind(match_status)
        self._semantic_match_pill.set_text(detail.match_result)
        self._semantic_reason_label.setText(f"Why: {detail.bounded_reason_code}")

        if detail.attribution_finality == "offline final":
            finality_status = UiStatusKind.OK
        elif detail.has_attribution:
            finality_status = UiStatusKind.INFO
        else:
            finality_status = UiStatusKind.NEUTRAL
        self._attribution_finality_pill.set_kind(finality_status)
        self._attribution_finality_pill.set_text(
            f"follow-up status · {detail.attribution_finality}"
        )

        self._confidence_card.set_primary_text(detail.probability_confidence)
        self._confidence_card.set_secondary_text(
            "confidence that the expected response was observed"
        )
        self._confidence_card.set_status(method_status, None)

        self._soft_reward_card.set_primary_text(detail.soft_reward_candidate)
        self._soft_reward_card.set_secondary_text("transparency only; not the live reward")
        self._soft_reward_card.set_status(UiStatusKind.INFO, None)

        self._au12_lifts_card.set_primary_text(detail.au12_lift_metrics)
        self._au12_lifts_card.set_secondary_text(
            "response change compared with before the stimulus"
        )
        self._au12_lifts_card.set_status(UiStatusKind.INFO, None)

        self._peak_latency_card.set_primary_text(detail.au12_peak_latency)
        self._peak_latency_card.set_secondary_text(
            "how long after the stimulus the response peaked"
        )
        self._peak_latency_card.set_status(UiStatusKind.INFO, None)

        self._synchrony_card.set_primary_text(detail.synchrony_metrics)
        self._synchrony_card.set_secondary_text("whether signals moved together after timing shift")
        self._synchrony_card.set_status(UiStatusKind.INFO, None)

        self._outcome_link_lag_card.set_primary_text(detail.outcome_link_lag)
        self._outcome_link_lag_card.set_secondary_text("time from stimulus event to outcome")
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
