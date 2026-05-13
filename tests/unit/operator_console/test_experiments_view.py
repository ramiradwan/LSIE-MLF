"""Tests for `ExperimentsView` — Phase 10.

Locks the summary cards + arm table render path and the `null` / empty
detail branches. Reward-explanation text is the VM's responsibility and
is covered in `test_viewmodels.py`; this file focuses on the view.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QWidget

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.operator_console import (
    ArmDecisionEvidence,
    ArmSummary,
    BanditDecisionEvidence,
    ExperimentDetail,
)
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.views.experiments_view import ExperimentsView
from services.operator_console.widgets.responsive_layout import ResponsiveWidthBand

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _grid_widget(grid: object, row: int, column: int) -> QWidget:
    item = grid.itemAtPosition(row, column)  # type: ignore[attr-defined]
    assert item is not None
    widget = item.widget()
    assert widget is not None
    return cast(QWidget, widget)


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(text=text),
        expected_stimulus_rule="Deliver the spoken greeting to the creator",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


def _arm(
    arm_id: str,
    *,
    alpha: float = 2.0,
    beta: float = 3.0,
    recent_reward: float | None = 0.4,
    recent_pass: float | None = 0.6,
    selection_count: int = 5,
    enabled: bool = True,
) -> ArmSummary:
    return ArmSummary(
        arm_id=arm_id,
        stimulus_definition=_stimulus_definition(f"Hei {arm_id}"),
        posterior_alpha=alpha,
        posterior_beta=beta,
        evaluation_variance=0.01,
        selection_count=selection_count,
        recent_reward_mean=recent_reward,
        recent_semantic_pass_rate=recent_pass,
        decision_evidence=ArmDecisionEvidence(
            arm_id=arm_id,
            pre_update_alpha=alpha,
            pre_update_beta=beta,
            sampled_theta=0.73,
        ),
        enabled=enabled,
    )


def _detail(active_arm_id: str | None = "a1") -> ExperimentDetail:
    return ExperimentDetail(
        experiment_id="exp-42",
        label="greeting v1",
        active_arm_id=active_arm_id,
        arms=[_arm("a1", recent_reward=0.7), _arm("a2", recent_reward=0.3)],
        decision_evidence=BanditDecisionEvidence(
            selection_time_utc=_NOW,
            selected_arm_id="a1",
            policy_version="7B.v4",
            decision_context_hash="a" * 64,
            random_seed=42,
            arm_evidence=[],
        ),
        last_update_summary="arm a1 updated posterior to α=2, β=3",
        last_updated_utc=_NOW,
    )


def _view() -> tuple[ExperimentsView, OperatorStore]:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model, default_experiment_id="exp-42")
    return ExperimentsView(vm), store


def test_experiments_view_shows_empty_state_when_no_detail() -> None:
    view, _store = _view()
    # Offscreen QPA: parents aren't shown, so `isVisible()` is False. Read
    # the local hide flag instead.
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is True  # type: ignore[attr-defined]
    assert view._scroll.widget() is view._body_container  # type: ignore[attr-defined]


def test_experiments_view_populates_cards_on_detail() -> None:
    view, store = _view()
    store.set_experiment(_detail())
    # Experiment card shows the human label and id as secondary.
    assert "greeting v1" in view._experiment_card._primary.text()  # type: ignore[attr-defined]
    assert "exp-42" in view._experiment_card._secondary.text()  # type: ignore[attr-defined]
    # Active strategy card shows the id and plain-language posterior summary.
    assert view._active_arm_card._primary.text() == "a1"  # type: ignore[attr-defined]
    secondary = view._active_arm_card._secondary.text()  # type: ignore[attr-defined]
    assert "positive history" in secondary and "miss history" in secondary
    arms_secondary = view._arms_card._secondary.text()  # type: ignore[attr-defined]
    assert "strongest recent reward" in arms_secondary
    assert "strategy a1" in arms_secondary
    assert "confirmed" in arms_secondary


def test_experiments_view_strategy_evidence_panel_ranks_active_arm() -> None:
    view, store = _view()
    assert view._strategy_panel.accessibleName() == "Strategy evidence"  # type: ignore[attr-defined]
    store.set_experiment(_detail())

    cards = view._strategy_panel._cards  # type: ignore[attr-defined]
    assert cards[0]._title.text() == "a1"  # type: ignore[attr-defined]
    assert "Active · Lower uncertainty so far" in cards[0]._primary.text()  # type: ignore[attr-defined]
    secondary = cards[0]._secondary.text()  # type: ignore[attr-defined]
    assert "stimulus confirmed 60%" in secondary
    assert "Stimulus text: Hei a1" in secondary
    assert "Decision evidence: picked with decision-time history 2.000/3.000" in secondary
    assert "sample 73%" in secondary
    subtitle = view._strategy_panel._subtitle.text()  # type: ignore[attr-defined]
    assert subtitle == "Compare observed responses, current history, and decision-time evidence."
    assert cards[0]._status._label.text() == "active"  # type: ignore[attr-defined]


def test_experiments_view_strategy_evidence_panel_handles_sparse_disabled_arm() -> None:
    view, store = _view()
    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp-sparse",
            label="sparse",
            active_arm_id="new",
            arms=[
                _arm(
                    "new",
                    recent_reward=None,
                    recent_pass=None,
                    selection_count=0,
                ),
                _arm(
                    "off",
                    recent_reward=None,
                    recent_pass=None,
                    selection_count=0,
                    enabled=False,
                ),
            ],
            last_updated_utc=_NOW,
        )
    )

    cards = view._strategy_panel._cards  # type: ignore[attr-defined]
    assert "Active · Needs first try" in cards[0]._primary.text()  # type: ignore[attr-defined]
    assert "No recent outcome yet" in cards[0]._secondary.text()  # type: ignore[attr-defined]
    assert cards[1]._primary.text() == "Disabled"  # type: ignore[attr-defined]


def test_experiments_view_active_arm_missing_surfaces_warn() -> None:
    view, store = _view()
    # Active strategy id not present in the strategy list — surface as WARN.
    detail = _detail(active_arm_id="ghost")
    store.set_experiment(detail)
    assert view._active_arm_card._primary.text() == "ghost"  # type: ignore[attr-defined]
    assert (
        "not present" in view._active_arm_card._secondary.text()  # type: ignore[attr-defined]
    )


def test_experiments_view_no_active_arm_uses_neutral_placeholder() -> None:
    view, store = _view()
    store.set_experiment(_detail(active_arm_id=None))
    assert view._active_arm_card._primary.text() == "—"  # type: ignore[attr-defined]


def test_experiments_view_error_changed_shows_alert_banner() -> None:
    view, store = _view()
    store.set_error("experiment", "unreachable")
    assert view._error_banner.isHidden() is False  # type: ignore[attr-defined]


def test_experiments_view_latest_update_falls_back_to_decision_metadata() -> None:
    view, store = _view()
    detail = _detail()
    detail.last_update_summary = None
    store.set_experiment(detail)

    summary = view._update_panel._summary.text()  # type: ignore[attr-defined]
    assert "Latest decision 2026-04-17 12:00:00Z" in summary
    assert "selected a1" in summary
    assert "policy 7B.v4" in summary
    assert "seed 42" in summary


def test_experiments_view_manage_create_visible_without_detail() -> None:
    view, _store = _view()
    panel = view._manage_panel  # type: ignore[attr-defined]
    assert panel.isHidden() is False
    assert panel._create_experiment_id.text() == "exp-42"  # type: ignore[attr-defined]
    assert panel._add_button.isEnabled() is False  # type: ignore[attr-defined]


def test_experiments_view_add_arm_button_follows_loaded_detail_and_inputs() -> None:
    view, store = _view()
    panel = view._manage_panel  # type: ignore[attr-defined]
    store.set_experiment(_detail())
    assert panel._add_button.isEnabled() is False  # type: ignore[attr-defined]
    assert panel._add_button.text() == "Add strategy"  # type: ignore[attr-defined]
    assert "neutral starting history" in panel._add_button.toolTip()  # type: ignore[attr-defined]
    panel._add_arm_id.setText("a3")  # type: ignore[attr-defined]
    panel._add_stimulus.setText("Hei uusi")  # type: ignore[attr-defined]
    assert panel._add_button.isEnabled() is True  # type: ignore[attr-defined]


def test_experiments_manage_panel_labels_write_fields() -> None:
    view, _store = _view()
    panel = view._manage_panel  # type: ignore[attr-defined]

    assert panel._create_experiment_id_label.buddy() is panel._create_experiment_id  # type: ignore[attr-defined]
    assert panel._create_label_label.buddy() is panel._create_label  # type: ignore[attr-defined]
    assert panel._create_arm_id_label.buddy() is panel._create_arm_id  # type: ignore[attr-defined]
    assert panel._create_stimulus_label.buddy() is panel._create_stimulus  # type: ignore[attr-defined]
    assert panel._add_arm_id_label.buddy() is panel._add_arm_id  # type: ignore[attr-defined]
    assert panel._add_stimulus_label.buddy() is panel._add_stimulus  # type: ignore[attr-defined]
    assert panel._create_arm_id.accessibleName() == "Initial strategy ID"  # type: ignore[attr-defined]
    assert "stimulus strategy" in panel._create_arm_id.accessibleDescription()  # type: ignore[attr-defined]
    assert panel._add_stimulus.accessibleName() == "Stimulus text"  # type: ignore[attr-defined]
    assert "delivered for this stimulus strategy" in panel._add_stimulus.accessibleDescription()  # type: ignore[attr-defined]


def test_experiments_view_reflects_rename_disable_create_readback() -> None:
    view, store = _view()
    store.set_experiment(
        ExperimentDetail(
            experiment_id="exp-created",
            label="created label",
            arms=[
                _arm(
                    "new",
                    enabled=False,
                    recent_reward=None,
                    recent_pass=None,
                    selection_count=0,
                )
            ],
            last_updated_utc=_NOW,
        )
    )
    model = view._vm.arms_model()  # type: ignore[attr-defined]
    assert view._experiment_card._primary.text() == "created label"  # type: ignore[attr-defined]
    assert model.data(model.index(0, 1)) == "Hei new"
    assert model.data(model.index(0, 2)) == "disabled"


def test_experiments_view_narrow_width_reflows_cards_and_table() -> None:
    view, store = _view()
    store.set_experiment(_detail())

    view.resize(620, 520)
    view._apply_responsive_layout(620)  # type: ignore[attr-defined]

    assert view._scroll.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.widget() is view._body_container  # type: ignore[attr-defined]
    assert view._cards_grid.current_band() is ResponsiveWidthBand.NARROW  # type: ignore[attr-defined]
    assert view._cards_grid.column_count() == 1  # type: ignore[attr-defined]
    table = view._table  # type: ignore[attr-defined]
    assert table.isColumnHidden(0) is False
    assert table.isColumnHidden(1) is False
    assert table.isColumnHidden(2) is False
    assert table.isColumnHidden(3) is True
    assert table.isColumnHidden(4) is True
    assert table.isColumnHidden(5) is True
    assert table.isColumnHidden(6) is False
    assert table.isColumnHidden(7) is True
    assert table.isColumnHidden(8) is True


def test_experiments_view_wide_width_shows_friendly_learning_headers() -> None:
    view, store = _view()
    store.set_experiment(_detail())

    view._apply_responsive_layout(1200)  # type: ignore[attr-defined]

    table = view._table  # type: ignore[attr-defined]
    model = table.model()
    assert model is not None
    assert table.isColumnHidden(1) is False
    assert table.isColumnHidden(8) is False
    assert model.headerData(1, Qt.Orientation.Horizontal) == "Stimulus text"
    assert model.headerData(8, Qt.Orientation.Horizontal) == "Recent stimulus confirmed"
    header = table.horizontalHeader()
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(8) == QHeaderView.ResizeMode.ResizeToContents


def test_experiments_manage_panel_reflows_for_medium_and_narrow_widths() -> None:
    view, _store = _view()
    panel = view._manage_panel  # type: ignore[attr-defined]

    panel.apply_responsive_width(900)
    create_row = panel._create_row  # type: ignore[attr-defined]
    add_row = panel._add_row  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 0, 0) is panel._create_experiment_id_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 0, 1) is panel._create_label_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 1, 0) is panel._create_experiment_id  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 1, 1) is panel._create_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 2, 0) is panel._create_arm_id_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 2, 1) is panel._create_stimulus_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 3, 0) is panel._create_arm_id  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 3, 1) is panel._create_stimulus  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 4, 0) is panel._create_button  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 0, 0) is panel._add_arm_id_label  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 0, 1) is panel._add_stimulus_label  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 1, 0) is panel._add_arm_id  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 1, 1) is panel._add_stimulus  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 2, 0) is panel._add_button  # type: ignore[attr-defined]

    panel.apply_responsive_width(620)
    assert _grid_widget(create_row, 0, 0) is panel._create_experiment_id_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 1, 0) is panel._create_experiment_id  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 2, 0) is panel._create_label_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 3, 0) is panel._create_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 4, 0) is panel._create_arm_id_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 5, 0) is panel._create_arm_id  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 6, 0) is panel._create_stimulus_label  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 7, 0) is panel._create_stimulus  # type: ignore[attr-defined]
    assert _grid_widget(create_row, 8, 0) is panel._create_button  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 0, 0) is panel._add_arm_id_label  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 1, 0) is panel._add_arm_id  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 2, 0) is panel._add_stimulus_label  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 3, 0) is panel._add_stimulus  # type: ignore[attr-defined]
    assert _grid_widget(add_row, 4, 0) is panel._add_button  # type: ignore[attr-defined]
