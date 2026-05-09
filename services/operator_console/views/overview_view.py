"""Overview page — the operator's at-a-glance dashboard.

Renders the six cards the operator scans on every page entry: Active
Session, Experiment, Physiology, Health, Latest Encounter, and the
Attention queue (alerts). Every value is pulled through
`OverviewViewModel`; the view itself holds no cached state, talks to no
network layer, and formats nothing inline — every string comes out of
`services/operator_console/formatters.py`.

Clicking the Active Session card emits `session_activated(UUID)` so the
shell can switch to Live Session with the same session already selected.

`on_activated()` / `on_deactivated()` are hooks the `MainWindow` invokes
when this page enters or leaves the stacked widget; they do not drive
poll scoping (that is the `PollingCoordinator`'s job off `route_changed`)
but they let the view nudge a fresh fetch so the operator does not have
to wait a full tick after switching back.

Spec references:
  §4.E.1         — operator-facing Overview surface
  §7B            — reward-explanation surfacing on Latest Encounter card
  §7C            — Co-Modulation Index null-valid rendering
  §12            — attention queue fed by the alert feed
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertEvent,
    AlertSeverity,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    LatestEncounterSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
    UiStatusKind,
)
from services.operator_console.formatters import (
    build_physiology_explanation,
    format_active_session_readback,
    format_health_state,
    format_reward,
    format_semantic_gate,
    format_session_id_compact,
    format_timestamp,
    physiology_labels,
    reward_detail_labels,
)
from services.operator_console.viewmodels.overview_vm import OverviewViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.widgets.responsive_layout import ResponsiveMetricGrid
from services.operator_console.widgets.section_header import SectionHeader

_ACTIVE_CONFLICT_STATUS = "active conflict"

# Cap on the attention queue length on the Overview — the full alert
# timeline lives on the Health page. Keeping the overview
# tight means the operator sees the top items and clicks through when
# they need history.
_ATTENTION_MAX: int = 8


# Severity → UI-status bucket so the attention-queue rows colour consistently
# with pills elsewhere. §12 severity taxonomy: INFO / WARNING / CRITICAL.
_SEVERITY_STATUS: dict[AlertSeverity, UiStatusKind] = {
    AlertSeverity.INFO: UiStatusKind.INFO,
    AlertSeverity.WARNING: UiStatusKind.WARN,
    AlertSeverity.CRITICAL: UiStatusKind.ERROR,
}


# §12 health-state → card-level UiStatusKind. OK/DEGRADED/RECOVERING/ERROR
# render as OK/WARN/PROGRESS/ERROR respectively so the operator can
# distinguish a subsystem that is self-healing from one that is broken.
_HEALTH_STATUS: dict[HealthState, UiStatusKind] = {
    HealthState.OK: UiStatusKind.OK,
    HealthState.DEGRADED: UiStatusKind.WARN,
    HealthState.RECOVERING: UiStatusKind.PROGRESS,
    HealthState.ERROR: UiStatusKind.ERROR,
    HealthState.UNKNOWN: UiStatusKind.NEUTRAL,
}


class OverviewView(QWidget):
    """Operator Overview page — six cards + attention queue."""

    # Emitted when the operator clicks the Active Session card. Shell
    # uses this to push the session id into the store and switch route.
    session_activated = Signal(object)  # UUID

    def __init__(
        self,
        vm: OverviewViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Overview",
            "What is running now, what needs attention, and why the latest result counted.",
            self,
            level="page",
        )
        self._error_banner = AlertBanner(self)

        self._active_session_card = MetricCard("Active Session", self)
        # Clicking the active-session card is the single navigation
        # shortcut on this page — takes the operator into Live Session
        # with the same session already selected.
        self._active_session_card.set_clickable(True, destination="Live Session")
        self._active_session_card.clicked.connect(self._on_active_session_clicked)

        self._experiment_card = MetricCard("Experiment", self)
        self._physiology_card = MetricCard("Physiology", self)
        self._health_card = MetricCard("Health", self)
        self._latest_encounter_card = MetricCard("Latest Encounter", self)
        self._attention_card = MetricCard("Attention", self)

        # Three operator jobs, three bands. Each pair of cards lives under
        # its own SectionHeader so the operator scans for "what's
        # running" / "do I trust the last result" / "what needs me"
        # without reading every card.
        self._now_header = SectionHeader(
            "Now",
            "What is currently running.",
            self,
            level="sub",
        )
        self._trust_header = SectionHeader(
            "Trust",
            "Whether the latest result and physiology can be relied on.",
            self,
            level="sub",
        )
        self._attention_header = SectionHeader(
            "Needs attention",
            "Subsystems and alerts the operator should triage.",
            self,
            level="sub",
        )

        self._now_grid = ResponsiveMetricGrid(parent=self)
        self._now_grid.set_widgets([self._active_session_card, self._experiment_card])
        self._trust_grid = ResponsiveMetricGrid(parent=self)
        self._trust_grid.set_widgets([self._latest_encounter_card, self._physiology_card])
        self._attention_grid = ResponsiveMetricGrid(parent=self)
        self._attention_grid.set_widgets([self._health_card, self._attention_card])

        self._attention_list = _AttentionList(self)

        # Body wraps in a scroll area so the three bands and the
        # attention list keep their natural heights when the page is
        # narrower than the total content needs. Without scroll, the
        # outer QVBoxLayout would squash MetricCards below their
        # wrap-aware minimumHeight at <900px viewports.
        self._body_container = QWidget(self)
        body = QVBoxLayout(self._body_container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self._now_header)
        body.addWidget(self._now_grid)
        body.addWidget(self._trust_header)
        body.addWidget(self._trust_grid)
        body.addWidget(self._attention_header)
        body.addWidget(self._attention_grid)
        body.addWidget(self._attention_list)
        body.addStretch(1)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("OverviewScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._body_container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._scroll, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Cache the session id so the click handler does not re-read the
        # VM (which might have moved on) between render and click.
        self._active_session_id: UUID | None = None

        self._vm.changed.connect(self._refresh)
        self._vm.error_changed.connect(self._on_error_changed)
        self._refresh()

    # ------------------------------------------------------------------
    # Page lifecycle hooks
    # ------------------------------------------------------------------

    def on_activated(self) -> None:
        """Called by the shell when the page enters the stacked widget.

        Renders from the latest store state so the operator sees the
        current snapshot immediately rather than waiting for the next
        poll tick to land.
        """
        self._refresh()

    def on_deactivated(self) -> None:
        """No-op: all signal connections remain live across pages.

        The `PollingCoordinator` is the authority on poll lifecycle —
        it stops the overview job on route-change. The view keeps its
        subscriptions so a delayed tick still mutates the cards even if
        the operator has stepped away.
        """
        return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._render_active_session(self._vm.active_session())
        self._render_experiment(self._vm.experiment_summary())
        self._render_physiology(self._vm.physiology_summary())
        self._render_health(self._vm.health_summary())
        self._render_latest_encounter(self._vm.latest_encounter())
        self._render_attention(self._vm.alerts())

    def _render_active_session(self, session: SessionSummary | None) -> None:
        if session is None:
            self._active_session_id = None
            self._active_session_card.set_primary_text("No active session")
            self._active_session_card.set_secondary_text(
                "Nothing is running. Start a session when the phone is ready."
            )
            self._active_session_card.set_status(UiStatusKind.NEUTRAL, "idle")
            # Disable the click affordance when nothing to navigate to.
            self._active_session_card.set_clickable(False)
            return
        self._active_session_id = session.session_id
        self._active_session_card.set_clickable(True, destination="Live Session")
        self._active_session_card.set_primary_text(format_session_id_compact(session.session_id))
        self._active_session_card.set_secondary_text(format_active_session_readback(session))
        status_kind = (
            UiStatusKind.ERROR if session.status == _ACTIVE_CONFLICT_STATUS else UiStatusKind.OK
        )
        self._active_session_card.set_status(status_kind, session.status)

    def _render_experiment(self, summary: ExperimentSummary | None) -> None:
        if summary is None:
            self._experiment_card.set_primary_text("—")
            self._experiment_card.set_secondary_text("No experiment data.")
            self._experiment_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        label = summary.label or summary.experiment_id
        self._experiment_card.set_primary_text(label)
        reward = format_reward(summary.latest_reward)
        self._experiment_card.set_secondary_text(
            f"{summary.arm_count} strategy option(s) · latest reward {reward}"
        )
        self._experiment_card.set_status(UiStatusKind.INFO, "sampling")

    def _render_physiology(self, snapshot: SessionPhysiologySnapshot | None) -> None:
        if snapshot is None:
            self._physiology_card.set_primary_text("—")
            self._physiology_card.set_secondary_text("No physiology data for this session.")
            self._physiology_card.set_status(UiStatusKind.NEUTRAL, "absent")
            return
        # §4.C.4: stale is a first-class state — surface it on the pill.
        any_stale = any(
            snap is not None and snap.is_stale is True
            for snap in (snapshot.operator, snapshot.streamer)
        )
        any_present = any(
            snap is not None and snap.rmssd_ms is not None
            for snap in (snapshot.operator, snapshot.streamer)
        )
        if not any_present:
            self._physiology_card.set_primary_text(physiology_labels().no_rmssd_summary)
        else:
            self._physiology_card.set_primary_text("Live heart data")
        self._physiology_card.set_secondary_text(build_physiology_explanation(snapshot))
        if not any_present:
            self._physiology_card.set_status(UiStatusKind.NEUTRAL, "absent")
        elif any_stale:
            self._physiology_card.set_status(UiStatusKind.WARN, "stale")
        else:
            self._physiology_card.set_status(UiStatusKind.OK, "fresh")

    def _render_health(self, snapshot: HealthSnapshot | None) -> None:
        if snapshot is None:
            self._health_card.set_primary_text("—")
            self._health_card.set_secondary_text("No health snapshot yet.")
            self._health_card.set_status(UiStatusKind.NEUTRAL, "unknown")
            return
        self._health_card.set_primary_text(format_health_state(snapshot.overall_state))
        secondary_parts: list[str] = []
        if snapshot.degraded_count > 0:
            secondary_parts.append(f"{snapshot.degraded_count} degraded")
        if snapshot.recovering_count > 0:
            secondary_parts.append(f"{snapshot.recovering_count} recovering")
        if snapshot.error_count > 0:
            secondary_parts.append(f"{snapshot.error_count} error")
        if not secondary_parts:
            secondary_parts.append("all subsystems ok")
        self._health_card.set_secondary_text(" · ".join(secondary_parts))
        self._health_card.set_status(
            _HEALTH_STATUS[snapshot.overall_state],
            format_health_state(snapshot.overall_state),
        )

    def _render_latest_encounter(self, encounter: LatestEncounterSummary | None) -> None:
        if encounter is None:
            self._latest_encounter_card.set_primary_text("—")
            self._latest_encounter_card.set_secondary_text("No completed encounter yet.")
            self._latest_encounter_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        self._latest_encounter_card.set_primary_text(
            f"reward {format_reward(encounter.gated_reward)}"
        )
        # §7B: surface the explicit reward inputs on the card so the
        # operator can reason about a zero without opening the detail pane.
        # §8/§7E diagnostics stay compact here; the full readback lives in
        # Live Session and does not add Overview table columns.
        diagnostics = self._vm.latest_encounter_semantic_attribution_diagnostics()
        reward_labels = reward_detail_labels()
        parts: list[str] = [
            f"{reward_labels.p90_title} {format_reward(encounter.p90_intensity)}",
            f"{reward_labels.gate_title} {format_semantic_gate(encounter.semantic_gate)}",
        ]
        if encounter.n_frames_in_window is not None:
            parts.append(f"{encounter.n_frames_in_window} frames")
        parts.append(diagnostics.compact_summary)
        parts.append(format_timestamp(encounter.segment_timestamp_utc))
        self._latest_encounter_card.set_secondary_text(" · ".join(parts))
        status_kind = (
            UiStatusKind.WARN
            if (encounter.semantic_gate == 0 or encounter.n_frames_in_window == 0)
            else UiStatusKind.OK
        )
        self._latest_encounter_card.set_status(status_kind, encounter.state.value)

    def _render_attention(self, alerts: list[AlertEvent]) -> None:
        # Top-N + summary pill on the Attention card; the full list goes
        # into the scrollable panel below the cards.
        if not alerts:
            self._attention_card.set_primary_text("0")
            self._attention_card.set_secondary_text("All clear.")
            self._attention_card.set_status(UiStatusKind.OK, "clear")
        else:
            critical = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            warnings = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
            self._attention_card.set_primary_text(f"{len(alerts)} open")
            self._attention_card.set_secondary_text(f"{critical} critical · {warnings} warning")
            if critical:
                self._attention_card.set_status(UiStatusKind.ERROR, "critical")
            elif warnings:
                self._attention_card.set_status(UiStatusKind.WARN, "warning")
            else:
                self._attention_card.set_status(UiStatusKind.INFO, "info")

        self._attention_list.set_alerts(alerts[:_ATTENTION_MAX])

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_active_session_clicked(self) -> None:
        if self._active_session_id is not None:
            self.session_activated.emit(self._active_session_id)

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)


class _AttentionList(QScrollArea):
    """Scrollable panel rendering the top alerts as simple row labels.

    Kept private to this module — the full alert timeline on the Health
    page uses `EventTimelineWidget` + `AlertsTableModel`, which is
    overkill for the Overview's short preview.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AttentionList")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._inner = QWidget(self)
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._empty = EmptyStateWidget(self._inner)
        self._empty.set_title("Attention queue")
        self._empty.set_message("All clear. New alerts will show here.")
        self._layout.addWidget(self._empty)

        self.setWidget(self._inner)

    def set_alerts(self, alerts: list[AlertEvent]) -> None:
        # Clear existing rows (leave the empty-state widget — we toggle it).
        while self._layout.count() > 1:
            item = self._layout.takeAt(1)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not alerts:
            self._empty.setVisible(True)
            return
        self._empty.setVisible(False)
        for alert in alerts:
            self._layout.addWidget(_AlertRow(alert, self._inner))


class _AlertRow(QFrame):
    """One row in the Overview attention queue."""

    def __init__(self, alert: AlertEvent, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        status = _SEVERITY_STATUS[alert.severity]
        severity_label = QLabel(alert.severity.value, self)
        severity_label.setObjectName("MetricCardTitle")
        message_label = QLabel(alert.message, self)
        message_label.setObjectName("MetricCardPrimary")
        message_label.setWordWrap(True)
        kind_label = QLabel(
            f"{alert.kind.value} · {format_timestamp(alert.emitted_at_utc)}",
            self,
        )
        kind_label.setObjectName("MetricCardSecondary")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(severity_label)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        layout.addLayout(top)
        layout.addWidget(message_label)
        layout.addWidget(kind_label)
        # `status` is not painted here — the severity label text and row
        # colour come from the theme via object name; the variable is
        # referenced so mypy does not flag the mapping lookup as dead.
        self.setProperty("ui_status", status.value)
