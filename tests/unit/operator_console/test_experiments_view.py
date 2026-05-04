"""Tests for `ExperimentsView` — Phase 10.

Locks the summary cards + arm table render path and the `null` / empty
detail branches. Reward-explanation text is the VM's responsibility and
is covered in `test_viewmodels.py`; this file focuses on the view.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.schemas.operator_console import ArmSummary, ExperimentDetail
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.views.experiments_view import ExperimentsView
from services.operator_console.widgets.responsive_layout import ResponsiveWidthBand

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


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
        greeting_text=f"Hei {arm_id}",
        posterior_alpha=alpha,
        posterior_beta=beta,
        evaluation_variance=0.01,
        selection_count=selection_count,
        recent_reward_mean=recent_reward,
        recent_semantic_pass_rate=recent_pass,
        enabled=enabled,
    )


def _detail(active_arm_id: str | None = "a1") -> ExperimentDetail:
    return ExperimentDetail(
        experiment_id="exp-42",
        label="greeting v1",
        active_arm_id=active_arm_id,
        arms=[_arm("a1", recent_reward=0.7), _arm("a2", recent_reward=0.3)],
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
    assert view._body_container.isHidden() is True  # type: ignore[attr-defined]


def test_experiments_view_populates_cards_on_detail() -> None:
    view, store = _view()
    store.set_experiment(_detail())
    # Experiment card shows the human label and id as secondary.
    assert "greeting v1" in view._experiment_card._primary.text()  # type: ignore[attr-defined]
    assert "exp-42" in view._experiment_card._secondary.text()  # type: ignore[attr-defined]
    # Active arm card shows the arm id and its α/β summary.
    assert view._active_arm_card._primary.text() == "a1"  # type: ignore[attr-defined]
    secondary = view._active_arm_card._secondary.text()  # type: ignore[attr-defined]
    assert "α" in secondary and "β" in secondary
    # Arms card lists best-recent arm in the secondary line.
    arms_secondary = view._arms_card._secondary.text()  # type: ignore[attr-defined]
    assert "a1" in arms_secondary  # a1 has the higher recent reward


def test_experiments_view_active_arm_missing_surfaces_warn() -> None:
    view, store = _view()
    # Active arm id not present in the arm list — surface as WARN.
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
    panel._add_arm_id.setText("a3")  # type: ignore[attr-defined]
    panel._add_greeting.setText("Hei uusi")  # type: ignore[attr-defined]
    assert panel._add_button.isEnabled() is True  # type: ignore[attr-defined]


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

    view._apply_responsive_layout(620)  # type: ignore[attr-defined]

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


def test_experiments_manage_panel_reflows_for_medium_and_narrow_widths() -> None:
    view, _store = _view()
    panel = view._manage_panel  # type: ignore[attr-defined]

    panel.apply_responsive_width(900)
    assert panel._create_row.itemAtPosition(0, 0).widget() is panel._create_experiment_id  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(0, 1).widget() is panel._create_label  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(1, 0).widget() is panel._create_arm_id  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(1, 1).widget() is panel._create_greeting  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(2, 0).widget() is panel._create_button  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(0, 0).widget() is panel._add_arm_id  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(0, 1).widget() is panel._add_greeting  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(1, 0).widget() is panel._add_button  # type: ignore[attr-defined]

    panel.apply_responsive_width(620)
    assert panel._create_row.itemAtPosition(0, 0).widget() is panel._create_experiment_id  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(1, 0).widget() is panel._create_label  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(2, 0).widget() is panel._create_arm_id  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(3, 0).widget() is panel._create_greeting  # type: ignore[attr-defined]
    assert panel._create_row.itemAtPosition(4, 0).widget() is panel._create_button  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(0, 0).widget() is panel._add_arm_id  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(1, 0).widget() is panel._add_greeting  # type: ignore[attr-defined]
    assert panel._add_row.itemAtPosition(2, 0).widget() is panel._add_button  # type: ignore[attr-defined]
