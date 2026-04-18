"""Live Session page — encounter timeline + reward explanation pane.

The operator-trust surface: every encounter row exposes the §7B reward
inputs the pipeline used (P90, semantic gate, gated reward, frames,
baseline B_neutral). Selecting a row drops the full explanation into
the detail pane alongside the §4.C.4 physiology freshness read for
that segment.

The view never formats strings inline. All operator language comes
through `formatters.py`. All business logic — arm readback, stimulus
lifecycle, reward explanation, countdown arithmetic — lives in
`LiveSessionViewModel`; this file is layout + signal wiring only.

Spec references:
  §4.C           — `_active_arm`, `_expected_greeting`, authoritative
                   `_stimulus_time`; header reads from live-session DTO
                   never from the encounter rows
  §4.C.4         — physiology freshness badge in the detail pane
  §4.E.1         — Live Session operator surface
  §7B            — reward = p90_intensity × semantic_gate; detail pane
                   surfaces every input the pipeline used
  §12            — non-retryable errors surface on the page-level banner
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
    format_reward,
    format_semantic_confidence,
    format_semantic_gate,
    format_timestamp,
    truncate_expected_greeting,
)
from services.operator_console.state import StimulusUiContext
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.widgets.section_header import SectionHeader
from services.operator_console.widgets.status_pill import StatusPill

# §2 pipeline segments are 30s, which aligns with the VM's default §7B
# measurement-window length. 1-second tick is the rightful cadence for
# a human-facing countdown — any faster flickers, any slower reads slow.
_COUNTDOWN_TICK_MS: int = 1000


# Visual state for the readiness pill on the header. `active_arm` +
# `expected_greeting` on the live-session DTO is what drives it — if
# either is missing the orchestrator has not yet chosen an arm, so the
# operator should not fire a stimulus.
def _readiness_status(session: SessionSummary | None) -> tuple[UiStatusKind, str]:
    if session is None:
        return UiStatusKind.NEUTRAL, "no session"
    if session.active_arm is None or session.expected_greeting is None:
        return UiStatusKind.WARN, "awaiting arm"
    return UiStatusKind.OK, "ready"


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
            "Encounter timeline with per-segment reward explanation.",
            self,
        )
        self._session_panel = _SessionHeaderPanel(self)
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No session selected")
        self._empty_state.set_message(
            "Pick a session from Overview or Sessions to start the live view."
        )

        self._table = self._build_table()
        self._detail_panel = _EncounterDetailPanel(self)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addWidget(self._session_panel)
        body.addWidget(self._table, 2)
        body.addWidget(self._detail_panel, 3)

        self._body_container = QWidget(self)
        self._body_container.setLayout(body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._empty_state)
        layout.addWidget(self._body_container, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(_COUNTDOWN_TICK_MS)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        # Subscriptions — the VM fans out all relevant store changes.
        self._vm.changed.connect(self._refresh)
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
        selection_model = table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_table_selection_changed)
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        session = self._vm.session()
        if session is None:
            self._empty_state.setVisible(True)
            self._body_container.setVisible(False)
            return
        self._empty_state.setVisible(False)
        self._body_container.setVisible(True)

        self._session_panel.set_session(
            session,
            active_arm=self._vm.active_arm(),
            expected_greeting=self._vm.expected_greeting(),
            readiness=_readiness_status(session),
        )

        selected = self._vm.selected_encounter()
        if selected is None:
            # When no row is selected, fall back to the latest encounter
            # so the detail pane never goes blank while data exists.
            selected = _latest_encounter(self._vm.encounters_model().rowCount())
            if selected is None:
                rows = self._vm.encounters_model()
                if rows.rowCount() > 0:
                    last = rows.row_at(rows.rowCount() - 1)
                    if last is not None:
                        selected = last
        self._detail_panel.set_encounter(selected, self._vm.reward_explanation())
        self._sync_countdown_timer()

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


# ----------------------------------------------------------------------
# Panels — small helper widgets kept private to this module
# ----------------------------------------------------------------------


class _SessionHeaderPanel(QFrame):
    """Header row: session id, arm, expected greeting, readiness pill.

    §4.C: arm and expected greeting come from the live-session DTO, not
    from any encounter row. The panel enforces this at the view layer
    by exposing a single `set_session` entry point.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel("Session", self)
        self._title.setObjectName("PanelTitle")
        self._session_label = QLabel("—", self)
        self._session_label.setObjectName("MetricCardPrimary")
        self._session_label.setWordWrap(True)
        self._arm_label = QLabel("", self)
        self._arm_label.setObjectName("MetricCardSecondary")
        self._arm_label.setWordWrap(True)
        self._greeting_label = QLabel("", self)
        self._greeting_label.setObjectName("MetricCardSecondary")
        self._greeting_label.setWordWrap(True)
        self._readiness_pill = StatusPill(self)
        self._readiness_pill.set_kind(UiStatusKind.NEUTRAL)
        self._readiness_pill.set_text("no session")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._readiness_pill)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._session_label)
        layout.addWidget(self._arm_label)
        layout.addWidget(self._greeting_label)

    def set_session(
        self,
        session: SessionSummary,
        *,
        active_arm: str | None,
        expected_greeting: str | None,
        readiness: tuple[UiStatusKind, str],
    ) -> None:
        self._session_label.setText(f"Session {session.session_id}")
        self._arm_label.setText(f"arm: {active_arm if active_arm else '—'}")
        if expected_greeting:
            self._greeting_label.setText(
                f"expected greeting: “{truncate_expected_greeting(expected_greeting, limit=80)}”"
            )
            self._greeting_label.setVisible(True)
        else:
            self._greeting_label.setText("")
            self._greeting_label.setVisible(False)
        kind, text = readiness
        self._readiness_pill.set_kind(kind)
        self._readiness_pill.set_text(text)


class _EncounterDetailPanel(QFrame):
    """Reward-explanation detail pane for the selected encounter.

    Grid of §7B inputs (P90, Gate, Gated reward, Frames, Baseline) +
    a human-readable explanation line. Physiology freshness for the
    segment sits at the bottom so the operator can tie physiology to
    the reward they are reading above.
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
        self._baseline_card = MetricCard("Baseline B_neutral", self)
        self._physiology_card = MetricCard("Physiology", self)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        order: list[MetricCard] = [
            self._p90_card,
            self._gate_card,
            self._reward_card,
            self._frames_card,
            self._baseline_card,
            self._physiology_card,
        ]
        for idx, card in enumerate(order):
            row, col = divmod(idx, 3)
            grid.addWidget(card, row, col)
        for col in range(3):
            grid.setColumnStretch(col, 1)

        self._explanation = QLabel("", self)
        self._explanation.setObjectName("MetricCardSecondary")
        self._explanation.setWordWrap(True)

        self._countdown_label = QLabel("", self)
        self._countdown_label.setObjectName("ActionBarCountdown")
        self._countdown_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addLayout(grid)
        layout.addWidget(self._explanation)
        layout.addWidget(self._countdown_label)

    def set_encounter(
        self,
        encounter: EncounterSummary | None,
        explanation: str,
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
            return

        self._subtitle.setText(
            f"Encounter {encounter.encounter_id} · {format_timestamp(encounter.segment_timestamp_utc)} · "
            f"state {encounter.state.value}"
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
            UiStatusKind.OK
            if encounter.state == EncounterState.COMPLETED
            else UiStatusKind.NEUTRAL
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

        self._baseline_card.set_primary_text(format_reward(encounter.baseline_b_neutral))
        self._baseline_card.set_secondary_text("neutral AU12 baseline")
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

    def set_countdown(self, seconds: float | None) -> None:
        if seconds is None:
            self._countdown_label.setText("")
            self._countdown_label.setVisible(False)
            return
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        self._countdown_label.setText(
            f"Measurement window: {minutes:02d}:{secs:02d} remaining"
        )
        self._countdown_label.setVisible(True)


def _latest_encounter(_row_count: int) -> EncounterSummary | None:
    """Placeholder so `_refresh` can fall back without re-walking rows.

    Kept as a deliberate no-op hook — the fallback in `_refresh` reads
    the last row directly off the model because the VM's `row_at` is
    the authoritative accessor and we would otherwise duplicate the
    bounds check here.
    """
    return None
