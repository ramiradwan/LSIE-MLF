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
    vm = ExperimentsViewModel(store, model)
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
