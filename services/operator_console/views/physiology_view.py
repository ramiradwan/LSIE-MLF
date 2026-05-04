"""Physiology page — per-`subject_role` snapshots + §7C Co-Modulation Index.

Renders the four physiology states explicitly per §4.C.4 and §7C:
  * **fresh**  — recent non-stale snapshot (§4.C.4)
  * **stale**  — snapshot present but `is_stale=True` (§4.C.4)
  * **absent** — no snapshot at all (strap off, provider outage)
  * **null-valid** co-modulation — §7C's documented legitimate outcome
    when insufficient aligned non-stale pairs exist for the rolling
    window; rendered with the `null_reason` verbatim so it reads as
    "not enough data yet", not as a subsystem error

The page deliberately avoids any fake high-frequency animation; values
only update when the store tick actually lands new data. Every string
goes through `formatters.py`.

Spec references:
  §4.C.4         — Physiological State Buffer freshness / staleness
  §4.E.2         — physiology_log fields (rmssd_ms, heart_rate_bpm, ...)
  §7C            — Co-Modulation Index (rolling Pearson, null-valid)
  §12            — physiology staleness alerts + error surfaces
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    AlertSeverity,
    CoModulationSummary,
    PhysiologyCurrentSnapshot,
    UiStatusKind,
)
from services.operator_console.formatters import (
    format_comodulation_index,
    format_freshness,
    format_percentage,
    format_timestamp,
)
from services.operator_console.viewmodels.physiology_vm import PhysiologyViewModel
from services.operator_console.widgets.alert_banner import AlertBanner
from services.operator_console.widgets.empty_state import EmptyStateWidget
from services.operator_console.widgets.metric_card import MetricCard
from services.operator_console.widgets.responsive_layout import ResponsiveMetricGrid
from services.operator_console.widgets.section_header import SectionHeader
from services.operator_console.widgets.status_pill import StatusPill


class PhysiologyView(QWidget):
    """Physiology page: operator/streamer cards + co-modulation summary."""

    def __init__(
        self,
        vm: PhysiologyViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._vm = vm

        self._header = SectionHeader(
            "Physiology",
            "Streamer / operator freshness and §7C Co-Modulation Index.",
            self,
        )
        self._error_banner = AlertBanner(self)
        self._empty_state = EmptyStateWidget(self)
        self._empty_state.set_title("No physiology snapshot")
        self._empty_state.set_message(
            "Physiology data will appear once an active session is "
            "publishing snapshots via the Orchestrator."
        )

        self._streamer_panel = _RolePanel("Streamer", self)
        self._operator_panel = _RolePanel("Operator", self)
        self._comodulation_panel = _CoModulationPanel(self)

        self._roles_grid = ResponsiveMetricGrid(parent=self)
        self._roles_grid.set_widgets([self._streamer_panel, self._operator_panel])

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)
        body.addWidget(self._roles_grid)
        body.addWidget(self._comodulation_panel)
        body.addStretch(1)

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
        self._refresh()

    def on_deactivated(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        snapshot = self._vm.snapshot()
        if snapshot is None:
            self._empty_state.setVisible(True)
            self._body_container.setVisible(False)
            return
        self._empty_state.setVisible(False)
        self._body_container.setVisible(True)

        self._streamer_panel.set_snapshot(self._vm.streamer_snapshot())
        self._operator_panel.set_snapshot(self._vm.operator_snapshot())
        self._comodulation_panel.set_summary(
            self._vm.comodulation(),
            explanation=self._vm.comodulation_explanation(),
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_error_changed(self, message: str) -> None:
        if message:
            self._error_banner.set_alert(AlertSeverity.WARNING, message)
        else:
            self._error_banner.set_alert(None, None)


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------


class _RolePanel(QFrame):
    """One role's card: RMSSD, HR, provider, freshness badge.

    Four visual states: fresh / stale / absent / no-rmssd. The stale
    state has its own pill wording (not just a red colour) so the
    operator reads the cause rather than guessing from a shade.
    """

    def __init__(self, role_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._role_label = role_label
        self._title = QLabel(role_label, self)
        self._title.setObjectName("PanelTitle")
        self._status = StatusPill(self)
        self._status.set_kind(UiStatusKind.NEUTRAL)
        self._status.set_text("absent")

        self._rmssd_card = MetricCard("RMSSD", self)
        self._hr_card = MetricCard("Heart rate", self)
        self._freshness_card = MetricCard("Freshness", self)
        self._provider_card = MetricCard("Provider", self)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._status)

        self._metrics_grid = ResponsiveMetricGrid(parent=self)
        self._metrics_grid.set_widgets(
            [
                self._rmssd_card,
                self._hr_card,
                self._freshness_card,
                self._provider_card,
            ]
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(self._metrics_grid)

    def set_snapshot(self, snap: PhysiologyCurrentSnapshot | None) -> None:
        if snap is None:
            self._apply_absent()
            return
        if snap.rmssd_ms is None:
            self._apply_no_rmssd(snap)
            return
        self._rmssd_card.set_primary_text(f"{snap.rmssd_ms:.0f} ms")
        self._rmssd_card.set_secondary_text("root mean square of successive differences")
        self._rmssd_card.set_status(UiStatusKind.INFO, None)

        if snap.heart_rate_bpm is not None:
            self._hr_card.set_primary_text(f"{snap.heart_rate_bpm} bpm")
        else:
            self._hr_card.set_primary_text("—")
        self._hr_card.set_secondary_text("")
        self._hr_card.set_status(UiStatusKind.NEUTRAL, None)

        freshness_text = format_freshness(snap.freshness_s, is_stale=snap.is_stale)
        self._freshness_card.set_primary_text(freshness_text)
        self._freshness_card.set_secondary_text(
            f"source ts {format_timestamp(snap.source_timestamp_utc)}"
        )
        if snap.is_stale is True:
            self._freshness_card.set_status(UiStatusKind.WARN, "stale")
            self._status.set_kind(UiStatusKind.WARN)
            self._status.set_text("stale")
        else:
            self._freshness_card.set_status(UiStatusKind.OK, "fresh")
            self._status.set_kind(UiStatusKind.OK)
            self._status.set_text("fresh")

        self._provider_card.set_primary_text(snap.provider or "—")
        self._provider_card.set_secondary_text("")
        self._provider_card.set_status(UiStatusKind.NEUTRAL, None)

    def _apply_absent(self) -> None:
        for card in (
            self._rmssd_card,
            self._hr_card,
            self._freshness_card,
            self._provider_card,
        ):
            card.set_primary_text("—")
            card.set_secondary_text("")
            card.set_status(UiStatusKind.NEUTRAL, None)
        self._status.set_kind(UiStatusKind.NEUTRAL)
        self._status.set_text("absent")

    def _apply_no_rmssd(self, snap: PhysiologyCurrentSnapshot) -> None:
        # Snapshot exists but the provider hasn't delivered RMSSD yet —
        # treat as absent for the numeric cards but surface the provider
        # so the operator can tell "strap off" from "strap on, no data".
        self._rmssd_card.set_primary_text("—")
        self._rmssd_card.set_secondary_text("no RMSSD in snapshot")
        self._rmssd_card.set_status(UiStatusKind.NEUTRAL, None)
        if snap.heart_rate_bpm is not None:
            self._hr_card.set_primary_text(f"{snap.heart_rate_bpm} bpm")
            self._hr_card.set_status(UiStatusKind.INFO, None)
        else:
            self._hr_card.set_primary_text("—")
            self._hr_card.set_status(UiStatusKind.NEUTRAL, None)
        self._hr_card.set_secondary_text("")
        freshness_text = format_freshness(snap.freshness_s, is_stale=snap.is_stale)
        self._freshness_card.set_primary_text(freshness_text)
        self._freshness_card.set_secondary_text(
            f"source ts {format_timestamp(snap.source_timestamp_utc)}"
        )
        self._freshness_card.set_status(UiStatusKind.NEUTRAL, None)
        self._provider_card.set_primary_text(snap.provider or "—")
        self._provider_card.set_secondary_text("")
        self._provider_card.set_status(UiStatusKind.NEUTRAL, None)
        self._status.set_kind(UiStatusKind.NEUTRAL)
        self._status.set_text("no RMSSD")


class _CoModulationPanel(QFrame):
    """§7C Co-Modulation Index summary.

    Three visual states:
      * numeric index — rendered as `+0.342`
      * null-valid — `null_reason` read verbatim on the secondary line;
        card status reads "null-valid" (info, not warn) so the operator
        does not misread it as a subsystem failure
      * absent — no co-modulation row at all (yet)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._title = QLabel("Co-Modulation Index", self)
        self._title.setObjectName("PanelTitle")
        self._subtitle = QLabel(
            "Rolling Pearson correlation over aligned RMSSD bins.",
            self,
        )
        self._subtitle.setObjectName("PanelSubtitle")
        self._subtitle.setWordWrap(True)

        self._index_card = MetricCard("Index", self)
        self._observations_card = MetricCard("Observations", self)
        self._coverage_card = MetricCard("Coverage", self)
        self._window_card = MetricCard("Window", self)

        self._metrics_grid = ResponsiveMetricGrid(parent=self)
        self._metrics_grid.set_widgets(
            [
                self._index_card,
                self._observations_card,
                self._coverage_card,
                self._window_card,
            ]
        )

        self._explanation = QLabel("", self)
        self._explanation.setObjectName("MetricCardSecondary")
        self._explanation.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._metrics_grid)
        layout.addWidget(self._explanation)

    def set_summary(
        self,
        summary: CoModulationSummary | None,
        *,
        explanation: str,
    ) -> None:
        if summary is None:
            self._apply_absent(explanation)
            return
        self._index_card.set_primary_text(format_comodulation_index(summary))
        if summary.co_modulation_index is None:
            # §7C null-valid: info pill (not warn/error).
            self._index_card.set_status(UiStatusKind.INFO, "null-valid")
            self._index_card.set_secondary_text(
                summary.null_reason or "insufficient aligned non-stale pairs"
            )
        else:
            self._index_card.set_status(UiStatusKind.OK, "valid")
            self._index_card.set_secondary_text("Pearson r ∈ [-1, 1]")

        self._observations_card.set_primary_text(str(summary.n_paired_observations))
        self._observations_card.set_secondary_text("aligned non-stale pairs")
        self._observations_card.set_status(UiStatusKind.NEUTRAL, None)

        self._coverage_card.set_primary_text(format_percentage(summary.coverage_ratio, digits=0))
        self._coverage_card.set_secondary_text("window coverage")
        self._coverage_card.set_status(UiStatusKind.NEUTRAL, None)

        window_text = (
            f"{format_timestamp(summary.window_start_utc)} — "
            f"{format_timestamp(summary.window_end_utc)}"
        )
        self._window_card.set_primary_text(window_text)
        self._window_card.set_secondary_text("")
        self._window_card.set_status(UiStatusKind.NEUTRAL, None)

        self._explanation.setText(explanation)

    def _apply_absent(self, explanation: str) -> None:
        for card in (
            self._index_card,
            self._observations_card,
            self._coverage_card,
            self._window_card,
        ):
            card.set_primary_text("—")
            card.set_secondary_text("")
            card.set_status(UiStatusKind.NEUTRAL, None)
        self._explanation.setText(explanation)
