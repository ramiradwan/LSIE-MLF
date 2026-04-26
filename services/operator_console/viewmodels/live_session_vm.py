"""Live Session page viewmodel — Phase 8.

The Live Session VM is the center of operator trust. It owns four
concerns:

  1. The paired `EncountersTableModel`, updated whenever the store's
     `encounters_changed` fires, without losing operator selection.
  2. The authoritative arm/greeting, read from the store's live-session
     DTO (never from the last row of the table — rows are historical
     and the header can update independently when a new §4.C stimulus
     lands).
  3. The stimulus action-bar lifecycle, driven by a `StimulusUiContext`
     held in the store. The VM is the *only* writer of the context
     aside from the coordinator's success/failure callbacks — the view
     hands operator notes in, the VM composes the context and stamps
     it back.
  4. The §7B reward explanation and the measurement-window countdown.
     The countdown is derived from the *authoritative* `stimulus_time`
     on the encounter read-back, not from the operator's click wall
     clock — this is explicit in §4.C and is why a single stimulus
     submission does not drive the countdown until the orchestrator
     writes the timestamp back.

Spec references:
  §2             — 30s segment size anchors the default measurement window
  §4.C           — `_active_arm`, `_expected_greeting`, authoritative
                   `_stimulus_time`; idempotent writes via client_action_id
  §4.E.1         — Live Session operator surface
  §7B            — r_t = p90_intensity × semantic_gate; reward
                   explanation surfaces the exact inputs the pipeline
                   used (P90, semantic gate, frames, baseline)
  §12            — stimulus submission errors flow through the store's
                   per-scope error signal
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import (
    EncounterState,
    EncounterSummary,
    ObservationalAcousticSummary,
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
)
from services.operator_console.formatters import (
    AcousticDetailDisplay,
    build_acoustic_detail_display,
    build_reward_explanation,
)
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.viewmodels.base import ViewModelBase

# §2 pipeline uses 30s segments; the §7B measurement window aligns to
# the post-stimulus segment, so 30 seconds is the UI-side default until
# the orchestrator publishes the actual window length. This constant is
# used only for countdown display; the authoritative encounter state
# transition remains orchestrator-driven.
_MEASUREMENT_WINDOW_S: float = 30.0


class LiveSessionViewModel(ViewModelBase):
    """Owns the encounters table, stimulus lifecycle, and reward text."""

    # fmt: off
    selection_changed    = Signal(object)  # str | None (encounter_id)
    action_state_changed = Signal(object)  # StimulusUiContext
    # fmt: on

    def __init__(
        self,
        store: OperatorStore,
        encounters_model: EncountersTableModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(store, parent)
        self._encounters_model = encounters_model
        self._selected_encounter_id: str | None = None

        # Subscriptions — the VM does not refresh the model on its own
        # tick; it reacts to store mutations the coordinator drives.
        store.encounters_changed.connect(self._on_encounters_changed)
        store.live_session_changed.connect(self._on_live_session_changed)
        store.stimulus_state_changed.connect(self._on_stimulus_state_changed)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)

        # Seed from whatever the store already has — useful when the VM
        # is constructed after a first poll has already landed.
        self._encounters_model.set_rows(self._store.encounters())

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def session(self) -> SessionSummary | None:
        return self._store.live_session()

    def encounters_model(self) -> EncountersTableModel:
        return self._encounters_model

    def selected_encounter(self) -> EncounterSummary | None:
        if self._selected_encounter_id is None:
            return None
        return self._encounters_model.encounter_by_id(self._selected_encounter_id)

    def selected_acoustic(self) -> ObservationalAcousticSummary | None:
        """Canonical §7D acoustic summary for the selected encounter, if any."""
        encounter = self.selected_encounter()
        return encounter.observational_acoustic if encounter is not None else None

    def acoustic_detail_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> AcousticDetailDisplay:
        """Preformatted §7D detail payload for exactly one encounter."""
        summary = encounter.observational_acoustic if encounter is not None else None
        return build_acoustic_detail_display(summary)

    def acoustic_detail(self) -> AcousticDetailDisplay:
        """Preformatted §7D detail payload for the VM's default encounter."""
        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._store.encounters())
        return self.acoustic_detail_for_encounter(encounter)

    def select_encounter(self, encounter_id: str | None) -> None:
        if encounter_id == self._selected_encounter_id:
            return
        self._selected_encounter_id = encounter_id
        self.selection_changed.emit(encounter_id)
        self.emit_changed()

    def active_arm(self) -> str | None:
        # §4.C: arm comes from the orchestrator-owned live-session DTO,
        # not from a row in the encounter table. The header can change
        # between encounters when the operator swaps arms; historical
        # rows must not override that.
        session = self._store.live_session()
        return session.active_arm if session is not None else None

    def expected_greeting(self) -> str | None:
        session = self._store.live_session()
        return session.expected_greeting if session is not None else None

    # ------------------------------------------------------------------
    # Stimulus lifecycle
    # ------------------------------------------------------------------

    def stimulus_ui_context(self) -> StimulusUiContext:
        return self._store.stimulus_ui_context()

    def build_stimulus_request_id(self) -> UUID:
        """Mint a fresh idempotency key for the next submission.

        Kept as a method (not inlined at the call site) so the VM owns
        the dedup-key lifetime: the shell's action bar asks the VM for
        a key, the VM remembers it on the context, and the coordinator
        carries the same key to the API.
        """
        return uuid4()

    def set_stimulus_submitting(self, note: str | None) -> UUID:
        """Move the action bar into SUBMITTING state and return the key.

        The context holds `client_action_id` through the full
        submitting → accepted → measuring → completed path so a
        re-send (retry, double-click) can reuse the same key — §4.C's
        idempotency contract is UI-facing here, not just server-side.
        """
        action_id = uuid4()
        ctx = StimulusUiContext(
            state=StimulusActionState.SUBMITTING,
            client_action_id=action_id,
            operator_note=note,
            submitted_at_utc=None,
            accepted_at_utc=None,
            authoritative_stimulus_time_utc=None,
        )
        self._store.set_stimulus_ui_context(ctx)
        return action_id

    def apply_stimulus_accepted(self, accepted: StimulusAccepted) -> None:
        """Apply the API Server's ack. `accepted=False` → FAILED.

        Note: `received_at_utc` is the API receive time for audit only;
        the authoritative `stimulus_time` that anchors the §7B
        measurement window is not on this payload. It will arrive via
        the next encounters poll and be reconciled into the context by
        `reconcile_authoritative_stimulus_time()`.
        """
        ctx = self._store.stimulus_ui_context()
        if not accepted.accepted:
            new_ctx = StimulusUiContext(
                state=StimulusActionState.FAILED,
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                message=accepted.message,
            )
        else:
            new_ctx = StimulusUiContext(
                state=StimulusActionState.ACCEPTED,
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                accepted_at_utc=accepted.received_at_utc,
                message=accepted.message,
            )
        self._store.set_stimulus_ui_context(new_ctx)

    def reconcile_authoritative_stimulus_time(self) -> None:
        """Promote the context to MEASURING when the orchestrator's
        `stimulus_time_utc` lands on the latest encounter.

        Looking at the latest encounter — the one whose state is
        `STIMULUS_ISSUED` or `MEASURING` — is safe here because only
        one stimulus can be in flight at a time per §4.C.
        """
        ctx = self._store.stimulus_ui_context()
        if ctx.state not in (
            StimulusActionState.ACCEPTED,
            StimulusActionState.MEASURING,
        ):
            return
        encounters = self._store.encounters()
        latest = _latest_encounter_for_stimulus(encounters)
        if latest is None or latest.stimulus_time_utc is None:
            return
        next_state = (
            StimulusActionState.COMPLETED
            if latest.state == EncounterState.COMPLETED
            else StimulusActionState.MEASURING
        )
        if (
            next_state == ctx.state
            and ctx.authoritative_stimulus_time_utc == latest.stimulus_time_utc
        ):
            return
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=next_state,
                client_action_id=ctx.client_action_id,
                operator_note=ctx.operator_note,
                accepted_at_utc=ctx.accepted_at_utc,
                authoritative_stimulus_time_utc=latest.stimulus_time_utc,
                message=ctx.message,
            )
        )

    def measurement_window_remaining_s(self, now_utc: datetime) -> float | None:
        """Seconds remaining in the §7B measurement window.

        Returns ``None`` when no authoritative stimulus time has landed
        yet (the operator's click time is never used — §4.C). Clamps
        to zero rather than going negative so the view can render 0
        while the encounter transitions to COMPLETED.
        """
        ctx = self._store.stimulus_ui_context()
        if ctx.authoritative_stimulus_time_utc is None:
            return None
        elapsed = (now_utc - ctx.authoritative_stimulus_time_utc).total_seconds()
        remaining = _MEASUREMENT_WINDOW_S - elapsed
        if remaining < 0.0:
            return 0.0
        return remaining

    # ------------------------------------------------------------------
    # Reward explanation
    # ------------------------------------------------------------------

    def reward_explanation_for_encounter(
        self,
        encounter: EncounterSummary | None,
    ) -> str:
        """Formatted §7B reward text for exactly one detail-pane encounter."""
        if encounter is None:
            return "No completed encounter yet."
        return build_reward_explanation(encounter)

    def reward_explanation(self) -> str:
        """Formatted §7B reward text for the selected (or latest) encounter."""
        encounter = self.selected_encounter()
        if encounter is None:
            encounter = _latest_completed_encounter(self._store.encounters())
        return self.reward_explanation_for_encounter(encounter)

    def acoustic_explanation(self) -> str:
        """Formatted §7D acoustic text for the selected (or latest) encounter."""
        return self.acoustic_detail().explanation

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_encounters_changed(self, rows: object) -> None:
        # Defensive: the store always hands a list, but a typed object
        # parameter (for Qt marshalling) forces this shape-check.
        if isinstance(rows, list):
            self._encounters_model.set_rows(rows)
        else:
            self._encounters_model.set_rows(self._store.encounters())
        # Reconcile the stimulus UI context against the fresh rows —
        # the authoritative `stimulus_time` may have just landed.
        self.reconcile_authoritative_stimulus_time()
        # If the previously selected encounter has been evicted, drop
        # the selection silently so the view can update.
        if (
            self._selected_encounter_id is not None
            and self._encounters_model.encounter_by_id(self._selected_encounter_id) is None
        ):
            self._selected_encounter_id = None
            self.selection_changed.emit(None)
        self.emit_changed()

    def _on_live_session_changed(self, _session: object) -> None:
        # Arm / expected greeting / readiness may have moved.
        self.emit_changed()

    def _on_stimulus_state_changed(self, ctx: object) -> None:
        if isinstance(ctx, StimulusUiContext):
            self.action_state_changed.emit(ctx)
        else:
            self.action_state_changed.emit(self._store.stimulus_ui_context())
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope in ("live_session", "encounters", "stimulus"):
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope in ("live_session", "encounters", "stimulus"):
            self.set_error(None)


# ----------------------------------------------------------------------
# Helpers — kept module-level so the core class stays under one screen
# ----------------------------------------------------------------------


def _latest_encounter_for_stimulus(
    encounters: list[EncounterSummary],
) -> EncounterSummary | None:
    """Return the most recent encounter that carries a stimulus_time.

    Encounters are assumed newest-first per the API's ordering (the
    aggregate read-service `fetch_session_encounters` sorts by segment
    timestamp descending). We still scan and pick the max defensively.
    """
    if not encounters:
        return None
    with_stim = [e for e in encounters if e.stimulus_time_utc is not None]
    if not with_stim:
        return None
    return max(with_stim, key=lambda e: e.segment_timestamp_utc)


def _latest_completed_encounter(
    encounters: list[EncounterSummary],
) -> EncounterSummary | None:
    completed = [e for e in encounters if e.state == EncounterState.COMPLETED]
    if not completed:
        return None
    return max(completed, key=lambda e: e.segment_timestamp_utc)
