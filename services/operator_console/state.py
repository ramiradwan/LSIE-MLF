"""
Operator Console app-scoped state store.

`OperatorStore` is a single in-memory QObject that holds every piece of
state the UI reads. Views and viewmodels subscribe to its Qt signals;
the `PollingCoordinator` writes into it as fetched DTOs arrive. This
keeps views oblivious to the network layer and makes the whole surface
unit-testable without spinning up threads.

The store is deliberately dumb: no network I/O, no DTO construction, no
derivation logic beyond trivial setters that emit on change. Anything
more computed (reward-explanation text, freshness wording, etc.) lives
in `formatters.py` or the viewmodel layer.

Design constraints:
  - Selected session id is held separately from route so switching
    between Overview and Live Session preserves the operator's context.
  - Managed experiment id is held separately from the config default so
    polling stays pinned to the experiment the operator loaded or mutated.
  - `error_changed` carries a scope (`"overview"`, `"alerts"`, …) so a
    transient failure on one surface does not overwrite another's
    error banner.
  - `stimulus_state_changed` carries a `StimulusUiContext` value object
    so `ActionBar` can render idle / submitting / accepted /
    measuring / completed entirely from a single signal.

Spec references:
  §4.C           — stimulus lifecycle (authoritative stimulus_time stays
                   orchestrator-side; the context here is UI-only)
  §4.E.1         — operator-facing surfaces + multi-page layout
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from PySide6.QtCore import QObject, Signal

from packages.schemas.operator_console import (
    AlertEvent,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusActionState,
)


class AppRoute(StrEnum):
    """Six operator-facing pages for the navigation layout.

    `StrEnum` so the enum value doubles as a stable signal payload
    (`route_changed = Signal(str)` per the checklist).
    """

    OVERVIEW = "overview"
    LIVE_SESSION = "live_session"
    EXPERIMENTS = "experiments"
    PHYSIOLOGY = "physiology"
    HEALTH = "health"
    SESSIONS = "sessions"


@dataclass(frozen=True)
class StimulusUiContext:
    """UI-only state for the stimulus rail.

    `state` drives the action-bar button visuals. `client_action_id`
    is the idempotency key that `ActionBar` echoes back on
    the next submission. `authoritative_stimulus_time_utc` comes from
    the encounter read-back — the viewmodel reconciles the
    countdown against this value rather than the click wall-clock.
    """

    state: StimulusActionState = StimulusActionState.IDLE
    client_action_id: UUID | None = None
    operator_note: str | None = None
    message: str | None = None
    submitted_at_utc: datetime | None = None
    accepted_at_utc: datetime | None = None
    authoritative_stimulus_time_utc: datetime | None = None


class OperatorStore(QObject):
    """App-scoped state holder. Dumb: setters emit, no I/O.

    Signals are all `object`-typed where they carry DTOs so Qt's C++
    signal-marshalling does not choke on Python `BaseModel` subclasses.
    `error_changed` is `(scope, message)` so the UI can attribute an
    error to a specific card without overwriting another's state.
    """

    # fmt: off
    route_changed              = Signal(str)
    selected_session_changed   = Signal(object)
    managed_experiment_changed = Signal(object)
    overview_changed           = Signal(object)
    sessions_changed           = Signal(object)
    live_session_changed       = Signal(object)
    encounters_changed         = Signal(object)
    experiment_changed         = Signal(object)
    physiology_changed         = Signal(object)
    health_changed             = Signal(object)
    alerts_changed             = Signal(object)
    stimulus_state_changed     = Signal(object)
    error_changed              = Signal(str, str)
    error_cleared              = Signal(str)
    # fmt: on

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._route: AppRoute = AppRoute.OVERVIEW
        self._selected_session_id: UUID | None = None
        self._managed_experiment_id: str | None = None
        self._overview: OverviewSnapshot | None = None
        self._sessions: list[SessionSummary] = []
        self._live_session: SessionSummary | None = None
        self._encounters: list[EncounterSummary] = []
        self._experiment: ExperimentDetail | None = None
        self._physiology: SessionPhysiologySnapshot | None = None
        self._health: HealthSnapshot | None = None
        self._alerts: list[AlertEvent] = []
        self._stimulus_ui_context: StimulusUiContext = StimulusUiContext()
        self._errors: dict[str, str] = {}

    # ---- route ---------------------------------------------------------

    def route(self) -> AppRoute:
        return self._route

    def set_route(self, route: AppRoute) -> None:
        if route == self._route:
            return
        self._route = route
        # emit the str value so slots typed as `str` can bind directly
        self.route_changed.emit(route.value)

    # ---- selected session ---------------------------------------------

    def selected_session_id(self) -> UUID | None:
        return self._selected_session_id

    def set_selected_session_id(self, session_id: UUID | None) -> None:
        if session_id == self._selected_session_id:
            return
        self._selected_session_id = session_id
        self.selected_session_changed.emit(session_id)

    # ---- managed experiment -------------------------------------------

    def managed_experiment_id(self) -> str | None:
        return self._managed_experiment_id

    def set_managed_experiment_id(self, experiment_id: str | None) -> None:
        normalized = experiment_id.strip() if isinstance(experiment_id, str) else None
        if normalized == "":
            normalized = None
        if normalized == self._managed_experiment_id:
            return
        self._managed_experiment_id = normalized
        self.managed_experiment_changed.emit(normalized)

    # ---- overview ------------------------------------------------------

    def overview(self) -> OverviewSnapshot | None:
        return self._overview

    def set_overview(self, snapshot: OverviewSnapshot | None) -> None:
        self._overview = snapshot
        self.overview_changed.emit(snapshot)

    # ---- recent sessions ----------------------------------------------

    def sessions(self) -> list[SessionSummary]:
        return list(self._sessions)

    def set_sessions(self, sessions: list[SessionSummary]) -> None:
        self._sessions = list(sessions)
        self.sessions_changed.emit(list(self._sessions))

    # ---- live session header ------------------------------------------

    def live_session(self) -> SessionSummary | None:
        return self._live_session

    def set_live_session(self, session: SessionSummary | None) -> None:
        self._live_session = session
        self.live_session_changed.emit(session)

    # ---- encounters ----------------------------------------------------

    def encounters(self) -> list[EncounterSummary]:
        return list(self._encounters)

    def set_encounters(self, rows: list[EncounterSummary]) -> None:
        self._encounters = list(rows)
        self.encounters_changed.emit(list(self._encounters))

    # ---- experiment detail --------------------------------------------

    def experiment(self) -> ExperimentDetail | None:
        return self._experiment

    def set_experiment(self, detail: ExperimentDetail | None) -> None:
        if detail is not None:
            self.set_managed_experiment_id(detail.experiment_id)
        self._experiment = detail
        self.experiment_changed.emit(detail)

    # ---- physiology ----------------------------------------------------

    def physiology(self) -> SessionPhysiologySnapshot | None:
        return self._physiology

    def set_physiology(self, snapshot: SessionPhysiologySnapshot | None) -> None:
        self._physiology = snapshot
        self.physiology_changed.emit(snapshot)

    # ---- health --------------------------------------------------------

    def health(self) -> HealthSnapshot | None:
        return self._health

    def set_health(self, snapshot: HealthSnapshot | None) -> None:
        self._health = snapshot
        self.health_changed.emit(snapshot)

    # ---- alerts --------------------------------------------------------

    def alerts(self) -> list[AlertEvent]:
        return list(self._alerts)

    def set_alerts(self, events: list[AlertEvent]) -> None:
        self._alerts = list(events)
        self.alerts_changed.emit(list(self._alerts))

    # ---- stimulus UI context ------------------------------------------

    def stimulus_ui_context(self) -> StimulusUiContext:
        return self._stimulus_ui_context

    def set_stimulus_ui_context(self, ctx: StimulusUiContext) -> None:
        self._stimulus_ui_context = ctx
        self.stimulus_state_changed.emit(ctx)

    # ---- errors --------------------------------------------------------

    def error(self, scope: str) -> str | None:
        return self._errors.get(scope)

    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def set_error(self, scope: str, message: str) -> None:
        if self._errors.get(scope) == message:
            return
        self._errors[scope] = message
        self.error_changed.emit(scope, message)

    def clear_error(self, scope: str) -> None:
        if scope not in self._errors:
            return
        del self._errors[scope]
        self.error_cleared.emit(scope)
