"""Experiments page viewmodel.

Surfaces the §7B Thompson Sampling arm readback and the task-scoped
management intents for experiments and arms. Management commands are
only intents: the viewmodel validates operator input, emits typed
requests, and lets the coordinator execute one-shot API writes. It does
not import datastore clients and it never exposes posterior-owned
numeric fields for editing.

Semantic confidence is explicitly *not* modeled here as an input to the
reward. §7B defines the gated reward as `r_t = p90_intensity × semantic_gate`
with the gate being the integer 0/1 — confidence is informational on
the per-encounter surface and must not appear to move the posterior on
the Experiments page.

Spec references:
  §4.E.1         — Experiments operator surface
  §7B            — Thompson Sampling posterior; reward math is
                   intensity × gate, not confidence-scaled
"""

from __future__ import annotations

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.experiments import (
    ExperimentArmCreateRequest,
    ExperimentArmSeedRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import ExperimentDetail
from services.operator_console.formatters import (
    StrategyEvidenceDisplay,
    build_strategy_evidence_display,
    format_experiment_decision_summary,
)
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.viewmodels.base import ViewModelBase


class ExperimentsViewModel(ViewModelBase):
    """Owns the arms table and emits safe experiment-management intents."""

    # fmt: off
    create_experiment_requested = Signal(object)         # ExperimentCreateRequest
    add_arm_requested           = Signal(str, object)    # experiment_id, ExperimentArmCreateRequest
    update_arm_requested        = Signal(
        str, str, object
    )  # experiment_id, arm_id, stimulus_definition
    disable_arm_requested       = Signal(str, str)       # experiment_id, arm_id
    # fmt: on

    def __init__(
        self,
        store: OperatorStore,
        arms_model: ExperimentsTableModel,
        parent: QObject | None = None,
        *,
        default_experiment_id: str | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._arms_model = arms_model
        self._default_experiment_id = default_experiment_id
        store.experiment_changed.connect(self._on_experiment_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)
        arms_model.stimulus_text_edit_requested.connect(self.update_arm_stimulus_text)
        arms_model.disable_requested.connect(self.disable_arm)
        # Seed from whatever the store already holds.
        self._sync_rows(self._store.experiment())

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def detail(self) -> ExperimentDetail | None:
        return self._store.experiment()

    def arms_model(self) -> ExperimentsTableModel:
        return self._arms_model

    def active_arm_id(self) -> str | None:
        detail = self._store.experiment()
        return detail.active_arm_id if detail is not None else None

    def current_experiment_id(self) -> str | None:
        """Return the managed experiment id or the configured default id."""
        managed_id = self._store.managed_experiment_id()
        if managed_id:
            return managed_id
        detail = self._store.experiment()
        if detail is not None:
            return detail.experiment_id
        return self._default_experiment_id

    def can_add_arm(self) -> bool:
        return self._store.experiment() is not None

    def latest_update_summary(self) -> str:
        """Return the API-provided update summary, or a neutral placeholder.

        The update summary is pre-formatted server-side (see
        `OperatorReadService._build_experiment_summary`) so the
        operator sees the same phrasing the API logs use.
        """
        detail = self._store.experiment()
        if detail is None:
            return "No experiment update yet."
        if detail.last_update_summary is not None:
            return detail.last_update_summary
        decision_summary = format_experiment_decision_summary(detail)
        if decision_summary is not None:
            return decision_summary
        return "No experiment update yet."

    def strategy_evidence(self) -> list[StrategyEvidenceDisplay]:
        """Display-ready current strategy evidence from exposed arm summaries."""

        return build_strategy_evidence_display(self._store.experiment())

    # ------------------------------------------------------------------
    # Management commands — emit intents, coordinator performs writes
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        experiment_id: str,
        label: str,
        initial_arm_id: str,
        initial_stimulus_text: str,
    ) -> bool:
        """Validate a create request and emit it for coordinator execution."""
        normalized_experiment_id = experiment_id.strip()
        normalized_label = label.strip()
        normalized_arm = initial_arm_id.strip()
        normalized_stimulus_text = initial_stimulus_text.strip()
        if not all(
            (
                normalized_experiment_id,
                normalized_label,
                normalized_arm,
                normalized_stimulus_text,
            )
        ):
            self.set_error("Experiment id, label, arm id, and stimulus text are required.")
            return False
        try:
            request = ExperimentCreateRequest(
                experiment_id=normalized_experiment_id,
                label=normalized_label,
                arms=[
                    ExperimentArmSeedRequest(
                        arm=normalized_arm,
                        stimulus_definition=_stimulus_definition_from_text(
                            normalized_stimulus_text
                        ),
                    )
                ],
            )
        except ValidationError as exc:
            self.set_error(_validation_message(exc))
            return False
        self.set_error(None)
        self.create_experiment_requested.emit(request)
        return True

    def add_arm(self, arm_id: str, stimulus_text: str) -> bool:
        """Validate an add-arm request for the currently loaded experiment."""
        experiment_id = self.current_experiment_id()
        if not experiment_id or self._store.experiment() is None:
            self.set_error("Load or create an experiment before adding arms.")
            return False
        normalized_arm = arm_id.strip()
        normalized_stimulus_text = stimulus_text.strip()
        if not normalized_arm or not normalized_stimulus_text:
            self.set_error("Arm id and stimulus text are required.")
            return False
        try:
            request = ExperimentArmCreateRequest(
                arm=normalized_arm,
                stimulus_definition=_stimulus_definition_from_text(normalized_stimulus_text),
            )
        except ValidationError as exc:
            self.set_error(_validation_message(exc))
            return False
        self.set_error(None)
        self.add_arm_requested.emit(experiment_id, request)
        return True

    def update_arm_stimulus_text(self, arm_id: str, stimulus_text: str) -> bool:
        """Emit a stimulus-definition update for one arm; posterior fields stay read-only."""
        detail = self._store.experiment()
        if detail is None:
            self.set_error("Load or create an experiment before updating arms.")
            return False
        arm = self._arms_model.arm_by_id(arm_id)
        if arm is None:
            self.set_error(f"Arm {arm_id!r} is not present in the current experiment.")
            return False
        normalized_stimulus_text = stimulus_text.strip()
        if not normalized_stimulus_text:
            self.set_error("Stimulus text is required.")
            return False
        current_stimulus_text = arm.stimulus_definition.stimulus_payload.text.strip()
        if normalized_stimulus_text == current_stimulus_text:
            self.set_error(None)
            return True
        self.set_error(None)
        self.update_arm_requested.emit(
            detail.experiment_id,
            arm_id,
            _replace_stimulus_text(arm.stimulus_definition, normalized_stimulus_text),
        )
        return True

    def disable_arm(self, arm_id: str) -> bool:
        """Emit the supported one-way disable command for an enabled arm."""
        detail = self._store.experiment()
        if detail is None:
            self.set_error("Load or create an experiment before disabling arms.")
            return False
        arm = self._arms_model.arm_by_id(arm_id)
        if arm is None:
            self.set_error(f"Arm {arm_id!r} is not present in the current experiment.")
            return False
        if not arm.enabled:
            self.set_error(None)
            return True
        self.set_error(None)
        self.disable_arm_requested.emit(detail.experiment_id, arm_id)
        return True

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_experiment_changed(self, detail: object) -> None:
        if isinstance(detail, ExperimentDetail) or detail is None:
            self._sync_rows(detail if isinstance(detail, ExperimentDetail) else None)
        else:
            self._sync_rows(self._store.experiment())
        self.emit_changed()

    def _sync_rows(self, detail: ExperimentDetail | None) -> None:
        if detail is None:
            self._arms_model.set_rows([])
        else:
            self._arms_model.set_rows(list(detail.arms))

    def _on_error(self, scope: str, message: str) -> None:
        if scope == "experiment":
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope == "experiment":
            self.set_error(None)


def _stimulus_definition_from_text(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(text=text),
        expected_stimulus_rule="Deliver the operator stimulus to the live streamer.",
        expected_response_rule="The live streamer acknowledges or responds to the stimulus.",
    )


def _replace_stimulus_text(
    stimulus_definition: StimulusDefinition,
    text: str,
) -> StimulusDefinition:
    return stimulus_definition.model_copy(update={"stimulus_payload": StimulusPayload(text=text)})


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else None
    if isinstance(first, dict):
        message = first.get("msg")
        if isinstance(message, str) and message:
            return message
    return str(exc)
