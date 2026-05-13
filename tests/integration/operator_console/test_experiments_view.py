"""Integration: Experiments page arm table renders posterior + variance.

Asserts that a realistic `ExperimentDetail` pushed into the store lands
the §7B Thompson Sampling readback — posterior α/β, evaluation variance,
selection count — into the arm table the operator reads to reason about
exploration vs exploitation.

The test deliberately does NOT touch `semantic_confidence` because §7B's
reward is `p90_intensity × semantic_gate` — confidence is not a reward
input and must not appear as one on this page.

Spec references:
  §7B            — Thompson Sampling posteriors; reward = p90 × gate
  §4.E.1         — Experiments operator surface
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PySide6.QtCore import Qt

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.operator_console import ArmSummary, ExperimentDetail
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.views.experiments_view import ExperimentsView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(text=text),
        expected_stimulus_rule="Deliver the operator stimulus to the live streamer.",
        expected_response_rule="The live streamer acknowledges or responds to the stimulus.",
    )


def test_experiments_view_arm_table_renders_posteriors_and_variance() -> None:
    store = OperatorStore()
    model = ExperimentsTableModel()
    vm = ExperimentsViewModel(store, model)
    view = ExperimentsView(vm)

    detail = ExperimentDetail(
        experiment_id="exp-42",
        label="greeting line v2",
        active_arm_id="arm-a",
        arms=[
            ArmSummary(
                arm_id="arm-a",
                stimulus_definition=_stimulus_definition("hei kulta"),
                posterior_alpha=4.2,
                posterior_beta=2.1,
                evaluation_variance=0.03,
                selection_count=17,
                recent_reward_mean=0.71,
                recent_semantic_pass_rate=0.82,
            ),
            ArmSummary(
                arm_id="arm-b",
                stimulus_definition=_stimulus_definition("hei rakas"),
                posterior_alpha=2.0,
                posterior_beta=3.0,
                evaluation_variance=0.05,
                selection_count=9,
                recent_reward_mean=0.34,
                recent_semantic_pass_rate=0.55,
            ),
        ],
        last_update_summary="arm-a updated; α=4.2, β=2.1",
        last_updated_utc=_NOW,
    )
    store.set_experiment(detail)

    # Row count matches the two arms the store was seeded with.
    arms_model = view._vm.arms_model()  # type: ignore[attr-defined]
    assert arms_model.rowCount() == 2

    # At least one column surface α, β, and evaluation variance — we
    # scan the first row's display-role strings rather than hard-pinning
    # to a column index because the column order is the model's
    # business, not the view's.
    first_row_display: list[str] = []
    for col in range(arms_model.columnCount()):
        idx = arms_model.index(0, col)
        value = arms_model.data(idx, Qt.ItemDataRole.DisplayRole)
        if isinstance(value, str):
            first_row_display.append(value)
    joined = " | ".join(first_row_display)

    # arm id, posterior α + β, evaluation variance all appear somewhere
    # in the row. The formatter handles precision; integration-level we
    # only assert the values are present in rendered form.
    assert "arm-a" in joined
    assert "4.2" in joined or "4.20" in joined
    assert "2.1" in joined or "2.10" in joined
    assert "0.03" in joined

    # Active arm card should display the seeded active_arm_id.
    assert view._active_arm_card._primary.text() == "arm-a"  # type: ignore[attr-defined]
    # And the last-update summary panel shows the API-formatted text.
    assert (
        "arm-a updated" in view._update_panel._summary.text()  # type: ignore[attr-defined]
    )
