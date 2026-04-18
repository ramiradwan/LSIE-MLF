"""Experiments page — §7B Thompson Sampling arm readback.

Surfaces the state the operator needs to reason about the adaptive
experiment: which arm is live, the posterior α/β and evaluation variance
per arm, and the plain-language update summary the API's read service
formats. The view deliberately does *not* imply that
`semantic_confidence` moves the reward — §7B's gated reward is
`p90_intensity × semantic_gate`, full stop, so confidence stays off this
page.

The view talks to `ExperimentsViewModel` only: no direct store access,
no network calls, no inline string formatting (everything passes through
`formatters.py` at the VM/helper boundary).

Spec references:
  §4.E.1         — Experiments operator surface
  §7B            — Thompson Sampling reward = p90 × gate; posterior α/β,
                   evaluation variance, selection count drive the table
  §12            — errors on the experiment slice surface via banner
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    ArmSummary,
    ExperimentDetail,
    UiStatusKind,
)
from services.operator_console.formatters import (
    format_percentage,
    format_reward,
    format_timestamp,
)
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.widgets.section_header import SectionHeader


class ExperimentsView(QWidget):
    """Experiments page: summary cards + arm table + update summary."""

    def __init__(
        self,
        vm: ExperimentsViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Experiments",
            "Thompson Sampling posteriors per arm — §7B reward = p90 × gate.",
            self,
        )
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No experiment yet")
        self._empty_state.set_message(
            "Experiment data will appear once an arm has been selected "
            "for the active session."
        )

        self._experiment_card = MetricCard("Experiment", self)
        self._active_arm_card = MetricCard("Active arm", self)
        self._arms_card = MetricCard("Arms", self)

        cards = QHBoxLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(14)
        cards.addWidget(self._experiment_card, 1)
        cards.addWidget(self._active_arm_card, 1)
        cards.addWidget(self._arms_card, 1)

        self._table = self._build_table()
        self._update_panel = _LatestUpdatePanel(self)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addLayout(cards)
        body.addWidget(self._table, 2)
        body.addWidget(self._update_panel, 1)

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

        self._vm.changed.connect(self._refresh)
        self._vm.error_changed.connect(self._on_error_changed)

        self._refresh()

    # ------------------------------------------------------------------
    # Page lifecycle hooks
    # ------------------------------------------------------------------

    def on_activated(self) -> None:
        """Re-render from the latest store state on page entry."""
        self._refresh()

    def on_deactivated(self) -> None:
        """No-op — subscriptions remain live; poll scoping is the
        coordinator's concern, not the view's."""
        return None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_table(self) -> QTableView:
        table = QTableView(self)
        table.setObjectName("ExperimentArmsTable")
        table.setModel(self._vm.arms_model())
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
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        detail = self._vm.detail()
        if detail is None:
            self._empty_state.setVisible(True)
            self._body_container.setVisible(False)
            return
        self._empty_state.setVisible(False)
        self._body_container.setVisible(True)

        self._render_experiment_card(detail)
        self._render_active_arm_card(detail)
        self._render_arms_card(detail)
        self._update_panel.set_summary(
            self._vm.latest_update_summary(),
            last_updated_utc_text=format_timestamp(detail.last_updated_utc),
        )

    def _render_experiment_card(self, detail: ExperimentDetail) -> None:
        label = detail.label or detail.experiment_id
        self._experiment_card.set_primary_text(label)
        self._experiment_card.set_secondary_text(f"id {detail.experiment_id}")
        self._experiment_card.set_status(UiStatusKind.INFO, "sampling")

    def _render_active_arm_card(self, detail: ExperimentDetail) -> None:
        arm_id = detail.active_arm_id
        if arm_id is None:
            self._active_arm_card.set_primary_text("—")
            self._active_arm_card.set_secondary_text("No active arm selected.")
            self._active_arm_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        active = _find_arm(detail.arms, arm_id)
        self._active_arm_card.set_primary_text(arm_id)
        if active is None:
            self._active_arm_card.set_secondary_text(
                "Active arm id not present in arm list."
            )
            self._active_arm_card.set_status(UiStatusKind.WARN, "missing")
            return
        secondary_bits: list[str] = [
            f"α {format_reward(active.posterior_alpha)}",
            f"β {format_reward(active.posterior_beta)}",
        ]
        if active.evaluation_variance is not None:
            secondary_bits.append(
                f"var {format_reward(active.evaluation_variance)}"
            )
        secondary_bits.append(f"{active.selection_count} selection(s)")
        self._active_arm_card.set_secondary_text(" · ".join(secondary_bits))
        self._active_arm_card.set_status(UiStatusKind.OK, "active")

    def _render_arms_card(self, detail: ExperimentDetail) -> None:
        arm_count = len(detail.arms)
        self._arms_card.set_primary_text(str(arm_count))
        if arm_count == 0:
            self._arms_card.set_secondary_text("No arms registered.")
            self._arms_card.set_status(UiStatusKind.NEUTRAL, None)
            return
        best = _best_recent_reward_arm(detail.arms)
        if best is None:
            self._arms_card.set_secondary_text("No recent reward data yet.")
            self._arms_card.set_status(UiStatusKind.INFO, None)
            return
        self._arms_card.set_secondary_text(
            f"best recent reward {format_reward(best.recent_reward_mean)} "
            f"on arm {best.arm_id} "
            f"(semantic pass {format_percentage(best.recent_semantic_pass_rate, digits=0)})"
        )
        self._arms_card.set_status(UiStatusKind.INFO, None)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)


# ----------------------------------------------------------------------
# Panels + helpers
# ----------------------------------------------------------------------


class _LatestUpdatePanel(QFrame):
    """Shows the API-provided last-update summary + its timestamp.

    The update summary is pre-formatted server-side (see
    `OperatorReadService._build_experiment_summary`) so the operator
    reads the same phrasing the API logs use — consistent language
    across the two surfaces prevents confusion during incident review.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel("Latest update", self)
        self._title.setObjectName("PanelTitle")
        self._timestamp = QLabel("—", self)
        self._timestamp.setObjectName("PanelSubtitle")
        self._summary = QLabel("", self)
        self._summary.setObjectName("MetricCardSecondary")
        self._summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._timestamp)
        layout.addWidget(self._summary)

    def set_summary(self, summary: str, *, last_updated_utc_text: str) -> None:
        self._summary.setText(summary)
        self._timestamp.setText(f"last updated {last_updated_utc_text}")


def _find_arm(arms: list[ArmSummary], arm_id: str) -> ArmSummary | None:
    for arm in arms:
        if arm.arm_id == arm_id:
            return arm
    return None


def _best_recent_reward_arm(arms: list[ArmSummary]) -> ArmSummary | None:
    """Pick the arm with the highest recent reward mean for the summary card.

    Returns None when no arm has a recorded recent_reward_mean yet —
    the early-experiment window where Thompson Sampling is still
    exploring with uninformative priors.
    """
    with_reward = [a for a in arms if a.recent_reward_mean is not None]
    if not with_reward:
        return None
    return max(with_reward, key=lambda a: a.recent_reward_mean or 0.0)
