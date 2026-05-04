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
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
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
from services.operator_console.widgets.responsive_layout import (
    MetricGridColumns,
    ResponsiveBreakpoints,
    ResponsiveMetricGrid,
    ResponsiveWidthBand,
    TableColumnPolicy,
    apply_table_column_policies,
)
from services.operator_console.widgets.section_header import SectionHeader

_EXPERIMENT_BREAKPOINTS = ResponsiveBreakpoints(medium_min_width=760, wide_min_width=1040)

_EXPERIMENT_TABLE_POLICIES: tuple[TableColumnPolicy, ...] = (
    TableColumnPolicy(
        column=0,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 130},
    ),
    TableColumnPolicy(column=1, resize_mode=QHeaderView.ResizeMode.Stretch),
    TableColumnPolicy(
        column=2,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 96},
    ),
    TableColumnPolicy(
        column=3,
        visible_in=frozenset({ResponsiveWidthBand.MEDIUM, ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 112},
    ),
    TableColumnPolicy(
        column=4,
        visible_in=frozenset({ResponsiveWidthBand.MEDIUM, ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 112},
    ),
    TableColumnPolicy(
        column=5,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 120},
    ),
    TableColumnPolicy(
        column=6,
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 104},
    ),
    TableColumnPolicy(
        column=7,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 148},
    ),
    TableColumnPolicy(
        column=8,
        visible_in=frozenset({ResponsiveWidthBand.WIDE}),
        resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        widths={ResponsiveWidthBand.WIDE: 164},
    ),
)


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
            "Compare greeting options and see which one is currently being tried.",
            self,
        )
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No experiment yet")
        self._empty_state.set_message(
            "Experiment data will appear once an arm has been selected for the active session."
        )

        self._experiment_card = MetricCard("Experiment", self)
        self._active_arm_card = MetricCard("Active arm", self)
        self._arms_card = MetricCard("Arms", self)

        self._cards_grid = ResponsiveMetricGrid(
            breakpoints=_EXPERIMENT_BREAKPOINTS,
            columns=MetricGridColumns(wide=3, medium=2, narrow=1),
            parent=self,
        )
        self._cards_grid.set_widgets(
            [
                self._experiment_card,
                self._active_arm_card,
                self._arms_card,
            ]
        )

        self._manage_panel = _ManagePanel(self._vm, self)
        self._table = self._build_table()
        self._update_panel = _LatestUpdatePanel(self)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addWidget(self._cards_grid)
        body.addWidget(self._table, 2)
        body.addWidget(self._update_panel, 1)

        self._body_container = QWidget(self)
        self._body_container.setLayout(body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._header)
        layout.addWidget(self._error_banner)
        layout.addWidget(self._manage_panel)
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

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

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
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        table.setWordWrap(True)
        vertical = table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        horizontal = table.horizontalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        apply_table_column_policies(
            table,
            _EXPERIMENT_TABLE_POLICIES,
            breakpoints=_EXPERIMENT_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )
        return table

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        detail = self._vm.detail()
        self._manage_panel.set_detail(
            detail,
            default_experiment_id=self._vm.current_experiment_id(),
        )
        self._apply_responsive_layout(self.width())
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
            self._active_arm_card.set_secondary_text("Active arm id not present in arm list.")
            self._active_arm_card.set_status(UiStatusKind.WARN, "missing")
            return
        secondary_bits: list[str] = [
            f"positive history {format_reward(active.posterior_alpha)}",
            f"miss history {format_reward(active.posterior_beta)}",
        ]
        if active.evaluation_variance is not None:
            secondary_bits.append(f"uncertainty {format_reward(active.evaluation_variance)}")
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
            f"(greeting matched {format_percentage(best.recent_semantic_pass_rate, digits=0)})"
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

    def _apply_responsive_layout(self, width: int) -> None:
        effective_width = max(width, self._body_container.width())
        self._cards_grid.apply_width(effective_width)
        apply_table_column_policies(
            self._table,
            _EXPERIMENT_TABLE_POLICIES,
            width=effective_width,
            breakpoints=_EXPERIMENT_BREAKPOINTS,
            default_resize_mode=QHeaderView.ResizeMode.Stretch,
        )
        self._manage_panel.apply_responsive_width(effective_width)


# ----------------------------------------------------------------------
# Panels + helpers
# ----------------------------------------------------------------------


class _ManagePanel(QFrame):
    """Compact experiment/arm management controls.

    The panel intentionally owns only human-entered strings. Posterior
    fields stay in the read-only table columns; writes are emitted via
    `ExperimentsViewModel`, which routes them to the coordinator.
    """

    def __init__(
        self,
        vm: ExperimentsViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ExperimentManagePanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._vm = vm
        self._has_detail = False

        self._title = QLabel("Manage", self)
        self._title.setObjectName("PanelTitle")
        self._hint = QLabel(
            "Create or seed the current experiment, add greeting options, double-click a "
            "Greeting cell to rename it, or uncheck Enabled to disable an option. "
            "Learning-history columns are read-only.",
            self,
        )
        self._hint.setObjectName("PanelSubtitle")
        self._hint.setWordWrap(True)

        self._create_experiment_id = QLineEdit(self)
        self._create_experiment_id.setObjectName("CreateExperimentIdInput")
        self._create_experiment_id.setPlaceholderText("experiment id")
        self._create_label = QLineEdit(self)
        self._create_label.setObjectName("CreateExperimentLabelInput")
        self._create_label.setPlaceholderText("label")
        self._create_arm_id = QLineEdit(self)
        self._create_arm_id.setObjectName("CreateInitialArmInput")
        self._create_arm_id.setPlaceholderText("initial arm id")
        self._create_greeting = QLineEdit(self)
        self._create_greeting.setObjectName("CreateInitialGreetingInput")
        self._create_greeting.setPlaceholderText("initial greeting")
        self._create_button = QPushButton("Create experiment", self)
        self._create_button.setObjectName("CreateExperimentButton")

        self._create_row = QGridLayout()
        self._create_row.setContentsMargins(0, 0, 0, 0)
        self._create_row.setHorizontalSpacing(8)
        self._create_row.setVerticalSpacing(8)

        self._add_arm_id = QLineEdit(self)
        self._add_arm_id.setObjectName("AddArmIdInput")
        self._add_arm_id.setPlaceholderText("new arm id")
        self._add_greeting = QLineEdit(self)
        self._add_greeting.setObjectName("AddArmGreetingInput")
        self._add_greeting.setPlaceholderText("greeting text")
        self._add_button = QPushButton("Add arm", self)
        self._add_button.setObjectName("AddArmButton")

        self._add_row = QGridLayout()
        self._add_row.setContentsMargins(0, 0, 0, 0)
        self._add_row.setHorizontalSpacing(8)
        self._add_row.setVerticalSpacing(8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addLayout(self._create_row)
        layout.addLayout(self._add_row)

        for edit in (
            self._create_experiment_id,
            self._create_label,
            self._create_arm_id,
            self._create_greeting,
            self._add_arm_id,
            self._add_greeting,
        ):
            edit.textChanged.connect(self._sync_enabled)
        self._create_button.clicked.connect(self._on_create_clicked)
        self._add_button.clicked.connect(self._on_add_clicked)
        self._apply_grid_layout(band=ResponsiveWidthBand.WIDE)
        self._sync_enabled()

    def set_detail(
        self,
        detail: ExperimentDetail | None,
        *,
        default_experiment_id: str | None,
    ) -> None:
        self._has_detail = detail is not None
        if not self._create_experiment_id.text():
            candidate_id = detail.experiment_id if detail is not None else default_experiment_id
            self._create_experiment_id.setText(candidate_id or "")
        if detail is not None and not self._create_label.text():
            self._create_label.setText(detail.label or detail.experiment_id)
        self._add_arm_id.setEnabled(self._has_detail)
        self._add_greeting.setEnabled(self._has_detail)
        self._add_button.setToolTip(
            "Add a Beta(1,1) arm to the loaded experiment."
            if self._has_detail
            else "Load or create an experiment before adding arms."
        )
        self._sync_enabled()

    def apply_responsive_width(self, width: int) -> None:
        band = _EXPERIMENT_BREAKPOINTS.band_for_width(width)
        self._apply_grid_layout(band=band)

    def _apply_grid_layout(self, *, band: ResponsiveWidthBand) -> None:
        while self._create_row.count() > 0:
            self._create_row.takeAt(0)
        while self._add_row.count() > 0:
            self._add_row.takeAt(0)

        if band is ResponsiveWidthBand.WIDE:
            self._create_row.addWidget(self._create_experiment_id, 0, 0)
            self._create_row.addWidget(self._create_label, 0, 1)
            self._create_row.addWidget(self._create_arm_id, 0, 2)
            self._create_row.addWidget(self._create_greeting, 0, 3)
            self._create_row.addWidget(self._create_button, 0, 4)
            self._add_row.addWidget(self._add_arm_id, 0, 0)
            self._add_row.addWidget(self._add_greeting, 0, 1, 1, 3)
            self._add_row.addWidget(self._add_button, 0, 4)
            create_stretches = (1, 1, 1, 2, 0)
            add_stretches = (1, 2, 0, 0, 0)
        elif band is ResponsiveWidthBand.MEDIUM:
            self._create_row.addWidget(self._create_experiment_id, 0, 0)
            self._create_row.addWidget(self._create_label, 0, 1)
            self._create_row.addWidget(self._create_arm_id, 1, 0)
            self._create_row.addWidget(self._create_greeting, 1, 1)
            self._create_row.addWidget(self._create_button, 2, 0, 1, 2)
            self._add_row.addWidget(self._add_arm_id, 0, 0)
            self._add_row.addWidget(self._add_greeting, 0, 1)
            self._add_row.addWidget(self._add_button, 1, 0, 1, 2)
            create_stretches = (1, 1, 0, 0, 0)
            add_stretches = (1, 2, 0, 0, 0)
        else:
            self._create_row.addWidget(self._create_experiment_id, 0, 0)
            self._create_row.addWidget(self._create_label, 1, 0)
            self._create_row.addWidget(self._create_arm_id, 2, 0)
            self._create_row.addWidget(self._create_greeting, 3, 0)
            self._create_row.addWidget(self._create_button, 4, 0)
            self._add_row.addWidget(self._add_arm_id, 0, 0)
            self._add_row.addWidget(self._add_greeting, 1, 0)
            self._add_row.addWidget(self._add_button, 2, 0)
            create_stretches = (1, 0, 0, 0, 0)
            add_stretches = (1, 0, 0, 0, 0)

        for column, stretch in enumerate(create_stretches):
            self._create_row.setColumnStretch(column, stretch)
        for column, stretch in enumerate(add_stretches):
            self._add_row.setColumnStretch(column, stretch)

    def _on_create_clicked(self) -> None:
        self._vm.create_experiment(
            self._create_experiment_id.text(),
            self._create_label.text(),
            self._create_arm_id.text(),
            self._create_greeting.text(),
        )

    def _on_add_clicked(self) -> None:
        self._vm.add_arm(self._add_arm_id.text(), self._add_greeting.text())

    def _sync_enabled(self, *_: object) -> None:
        create_ready = all(
            edit.text().strip()
            for edit in (
                self._create_experiment_id,
                self._create_label,
                self._create_arm_id,
                self._create_greeting,
            )
        )
        self._create_button.setEnabled(create_ready)
        add_ready = (
            self._has_detail
            and bool(self._add_arm_id.text().strip())
            and bool(self._add_greeting.text().strip())
        )
        self._add_button.setEnabled(add_ready)


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
