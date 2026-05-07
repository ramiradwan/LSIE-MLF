"""Operator Console shell — sidebar + stacked content + persistent ActionBar.

The window composes the scaffold window around the `OperatorStore`
and `PollingCoordinator`. The window is the composition root for the
six operator pages; views are eager-instantiated into a
`QStackedWidget` so switching between routes is a cheap index change
(no re-polling, no re-wiring). A single `ActionBar` mounts once below
the content area per the §4.E.1 persistent stimulus rail —
mounting per-page would re-create its state on every route change.

The shell never talks to the API directly. Route selection is
forwarded to the store (which fans out to the coordinator's
route-scoped poll lifecycle), and stimulus submission goes through the
coordinator's one-shot write path so idempotency (`client_action_id`)
is consistent with the rest of the pipeline.

Spec references:
  §4.C           — stimulus lifecycle (authoritative stimulus_time is
                   orchestrator-owned; UI state derives from
                   `StimulusUiContext` only)
  §4.E.1         — operator-facing multi-page layout
  §12            — close handler must stop polling + join threads
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import (
    SessionSummary,
    StimulusAccepted,
    StimulusActionState,
    StimulusRequest,
)
from services.operator_console.config import OperatorConsoleConfig
from services.operator_console.polling import PollingCoordinator
from services.operator_console.state import AppRoute, OperatorStore, StimulusUiContext
from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.table_models.sessions_table_model import (
    SessionsTableModel,
)
from services.operator_console.viewmodels.experiments_vm import ExperimentsViewModel
from services.operator_console.viewmodels.health_vm import HealthViewModel
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.viewmodels.overview_vm import OverviewViewModel
from services.operator_console.viewmodels.physiology_vm import PhysiologyViewModel
from services.operator_console.viewmodels.sessions_vm import SessionsViewModel
from services.operator_console.views.experiments_view import ExperimentsView
from services.operator_console.views.health_view import HealthView
from services.operator_console.views.live_session_view import LiveSessionView
from services.operator_console.views.overview_view import OverviewView
from services.operator_console.views.physiology_view import PhysiologyView
from services.operator_console.views.sessions_view import SessionsView
from services.operator_console.widgets.action_bar import ActionBar
from services.operator_console.workers import OneShotSignals


@dataclass(frozen=True)
class _NavEntry:
    """One sidebar nav entry — the route it selects and its button label."""

    route: AppRoute
    label: str
    description: str


# Declaration order is also rendering order in the sidebar and the stack.
# The operator's eye sweeps top-to-bottom: Overview first (at-a-glance),
# Live Session (the active action surface), then drill-downs, then the
# historical Sessions page last.
_NAV_SPEC: tuple[_NavEntry, ...] = (
    _NavEntry(AppRoute.OVERVIEW, "Overview", "Open the at-a-glance operator overview."),
    _NavEntry(AppRoute.LIVE_SESSION, "Live Session", "Open the active stimulus workflow."),
    _NavEntry(AppRoute.EXPERIMENTS, "Experiments", "Open stimulus strategy management."),
    _NavEntry(AppRoute.PHYSIOLOGY, "Physiology", "Open derived heart-data and sync signals."),
    _NavEntry(AppRoute.HEALTH, "Health", "Open readiness checks and operator actions."),
    _NavEntry(AppRoute.SESSIONS, "Sessions", "Open recent sessions history."),
)
_INITIAL_ROUTE = AppRoute.LIVE_SESSION
_MIN_WINDOW_WIDTH = 900
_MIN_WINDOW_HEIGHT = 640
_SIDEBAR_MIN_WIDTH = 160
_SIDEBAR_MAX_WIDTH = 220
_SIDEBAR_RATIO = 0.18
_ACTION_BAR_COMPACT_WIDTH = 1024


class MainWindow(QMainWindow):
    """Shell window: sidebar + stacked content + persistent ActionBar."""

    def __init__(
        self,
        config: OperatorConsoleConfig,
        store: OperatorStore,
        coordinator: PollingCoordinator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._store = store
        self._coordinator = coordinator
        # Keep references to in-flight stimulus signal buses so Qt does
        # not GC them mid-flight. Keyed by the UUID str of the request.
        self._inflight_stimulus: dict[str, OneShotSignals] = {}

        self.setWindowTitle(f"LSIE-MLF Operator Console — {config.environment_label}")
        self.setMinimumSize(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT)
        self.resize(1280, 800)

        self._pages: dict[AppRoute, QWidget] = {}
        self._nav_buttons: dict[AppRoute, QPushButton] = {}
        self._sidebar: QWidget | None = None

        self._register_pages()
        self._stack = self._build_stack()
        self._action_bar = self._build_action_bar()
        sidebar = self._build_sidebar()

        content = QWidget(self)
        content.setObjectName("ContentSurface")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._stack, stretch=1)
        content_layout.addWidget(self._action_bar)

        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, stretch=1)
        self.setCentralWidget(root)
        self._update_responsive_layout(self.width())

        status_bar = QStatusBar(self)
        api_label = QLabel(
            f"API · {config.api_base_url} · env {config.environment_label}",
            status_bar,
        )
        api_label.setObjectName("StatusBarLabel")
        status_bar.addWidget(api_label)
        self.setStatusBar(status_bar)

        self._bind_navigation()
        self._bind_store()

        self._store.set_route(_INITIAL_ROUTE)
        self._on_store_route_changed(self._store.route().value)
        self._update_action_bar_context()
        self._update_action_bar_progress()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _register_pages(self) -> None:
        """Eager-instantiate every route's page widget.

        All six routes now have real implementations. Table models + VMs
        are constructed here so they share the window's lifetime and the
        coordinator's route scoping is the single source of truth for
        "is this page active?" via `on_route_changed`.
        """
        # Table models — one per tabular surface. Parented to
        # the window so their lifetime is shell-scoped.
        self._encounters_model = EncountersTableModel(self)
        self._experiments_arms_model = ExperimentsTableModel(self)
        self._health_model = HealthTableModel(self)
        self._alerts_model = AlertsTableModel(self)
        self._sessions_model = SessionsTableModel(self)

        # Viewmodels — subscribe to the store, expose
        # read-only getters plus safe operator intents (stimulus,
        # experiment management, and Sessions' selection push).
        self._overview_vm = OverviewViewModel(self._store, self)
        self._live_session_vm = LiveSessionViewModel(
            self._store,
            self._encounters_model,
            self,
            default_experiment_id=self._config.default_experiment_id,
        )
        self._live_session_vm.bind_session_lifecycle_actions(
            self._coordinator.request_session_start,
            self._coordinator.request_session_end,
        )
        self._live_session_vm.action_state_changed.connect(
            self._on_live_session_action_state_changed
        )
        self._experiments_vm = ExperimentsViewModel(
            self._store,
            self._experiments_arms_model,
            self,
            default_experiment_id=self._config.default_experiment_id,
        )
        self._experiments_vm.create_experiment_requested.connect(
            self._coordinator.create_experiment
        )
        self._experiments_vm.add_arm_requested.connect(self._coordinator.add_experiment_arm)
        self._experiments_vm.rename_arm_requested.connect(self._coordinator.rename_experiment_arm)
        self._experiments_vm.disable_arm_requested.connect(self._coordinator.disable_experiment_arm)
        self._physiology_vm = PhysiologyViewModel(self._store, self)
        self._health_vm = HealthViewModel(self._store, self._health_model, self._alerts_model, self)
        self._health_vm.bind_repair_action(self._coordinator.repair_install)
        self._sessions_vm = SessionsViewModel(self._store, self._sessions_model, self)

        overview_view = OverviewView(self._overview_vm, self)
        overview_view.session_activated.connect(self._on_session_activated)
        self._pages[AppRoute.OVERVIEW] = overview_view

        self._pages[AppRoute.LIVE_SESSION] = LiveSessionView(self._live_session_vm, self)
        self._pages[AppRoute.EXPERIMENTS] = ExperimentsView(self._experiments_vm, self)
        self._pages[AppRoute.PHYSIOLOGY] = PhysiologyView(self._physiology_vm, self)
        self._pages[AppRoute.HEALTH] = HealthView(self._health_vm, self)

        sessions_view = SessionsView(self._sessions_vm, self)
        # Sessions page emits `session_selected(UUID)` on double-click;
        # route it through the same handler Overview's active-session
        # card uses so the shell keeps a single place that pushes
        # selection into the store + jumps to Live Session.
        sessions_view.session_selected.connect(self._on_session_activated)
        self._pages[AppRoute.SESSIONS] = sessions_view

    def _build_stack(self) -> QStackedWidget:
        stack = QStackedWidget(self)
        for entry in _NAV_SPEC:
            stack.addWidget(self._pages[entry.route])
        return stack

    def _build_action_bar(self) -> ActionBar:
        bar = ActionBar(self)
        bar.set_compact_mode(self.width() < _ACTION_BAR_COMPACT_WIDTH)
        # No session yet — bar starts disabled. `_update_action_bar_context`
        # re-enables it when the store emits `selected_session_changed`.
        bar.set_session_context(None, None, None)
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame(self)
        sidebar.setObjectName("SidebarNav")
        sidebar.setMinimumWidth(_SIDEBAR_MIN_WIDTH)
        sidebar.setMaximumWidth(_SIDEBAR_MAX_WIDTH)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(4)

        title = QLabel("LSIE-MLF", sidebar)
        title.setObjectName("SidebarTitle")
        subtitle = QLabel("Operator Console", sidebar)
        subtitle.setObjectName("SidebarSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._nav_group = QButtonGroup(sidebar)
        self._nav_group.setExclusive(True)
        for index, entry in enumerate(_NAV_SPEC):
            btn = QPushButton(entry.label, sidebar)
            btn.setObjectName("NavButton")
            btn.setAccessibleName(entry.label)
            btn.setAccessibleDescription(entry.description)
            btn.setToolTip(entry.description)
            btn.setCheckable(True)
            if entry.route is _INITIAL_ROUTE:
                btn.setChecked(True)
            self._nav_group.addButton(btn, index)
            self._nav_buttons[entry.route] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        self._sidebar = sidebar
        return sidebar

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _bind_navigation(self) -> None:
        for entry in _NAV_SPEC:
            btn = self._nav_buttons[entry.route]
            # `entry=entry` pins the default-arg so the loop variable
            # does not leak its last value into the lambda.
            btn.clicked.connect(
                lambda _checked=False, entry=entry: self._on_route_selected(entry.route)
            )
        self._action_bar.stimulus_requested.connect(self._on_stimulus_requested)

    def _bind_store(self) -> None:
        self._store.route_changed.connect(self._on_store_route_changed)
        self._store.selected_session_changed.connect(self._on_selected_session_changed)
        self._store.live_session_changed.connect(self._on_live_session_changed)
        self._store.stimulus_state_changed.connect(self._on_stimulus_state_changed)

    # ------------------------------------------------------------------
    # Route handling
    # ------------------------------------------------------------------

    def _on_route_selected(self, route: AppRoute) -> None:
        """User clicked a nav button — push to store, store re-emits
        for the coordinator's poll lifecycle."""
        self._store.set_route(route)

    @Slot(str)
    def _on_store_route_changed(self, route_value: str) -> None:
        """Store emitted route_changed — sync the stack + sidebar check."""
        route = AppRoute(route_value)
        page = self._pages.get(route)
        if page is not None:
            # Fire on_deactivated on the page we're leaving and
            # on_activated on the page we're entering. Pages that do
            # not define the hooks are ignored (duck-typed so the
            # remaining scaffold views continue to work).
            previous = self._stack.currentWidget()
            if previous is not None and previous is not page:
                deactivate = getattr(previous, "on_deactivated", None)
                if callable(deactivate):
                    deactivate()
            self._stack.setCurrentWidget(page)
            activate = getattr(page, "on_activated", None)
            if callable(activate):
                activate()
        btn = self._nav_buttons.get(route)
        if btn is not None and not btn.isChecked():
            # Block signals so we don't re-emit clicked and bounce back
            # into _on_route_selected.
            with _SignalBlocker(btn):
                btn.setChecked(True)

    @Slot(object)
    def _on_session_activated(self, session_id: object) -> None:
        """Overview's Active Session card was clicked — jump to Live Session."""
        if not isinstance(session_id, UUID):
            return
        self._store.set_selected_session_id(session_id)
        self._store.set_route(AppRoute.LIVE_SESSION)

    # ------------------------------------------------------------------
    # Session / ActionBar context
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_selected_session_changed(self, _session_id: UUID | None) -> None:
        self._update_action_bar_context()

    @Slot(object)
    def _on_live_session_changed(self, _live: SessionSummary | None) -> None:
        self._update_action_bar_context()

    def _update_action_bar_context(self) -> None:
        """Push session stimulus context and safe-submit readiness into the ActionBar.

        The bar is only populated with stimulus context/readiness when the
        live-session DTO we have on hand describes the selected session;
        readiness is the console-level threshold check, not the worker's
        authoritative ``is_calibrating`` lifecycle flag.
        """
        session_id = self._store.selected_session_id()
        live = self._store.live_session()
        active_arm: str | None = None
        expected_greeting: str | None = None
        operator_ready: bool | None = None
        if live is not None and session_id is not None and live.session_id == session_id:
            active_arm = live.active_arm
            expected_greeting = live.expected_greeting
            operator_ready = self._live_session_vm.operator_ready_for_submit()
        self._action_bar.set_session_context(
            session_id,
            active_arm,
            expected_greeting,
            operator_ready_for_submit=operator_ready,
        )
        self._update_action_bar_progress()

    def _update_action_bar_progress(self) -> None:
        state = self._live_session_vm.stimulus_ui_context().state
        if state is StimulusActionState.MEASURING:
            self._action_bar.set_countdown_remaining(
                self._live_session_vm.measurement_window_remaining_s(datetime.now(UTC))
            )
        else:
            self._action_bar.set_countdown_remaining(None)
        self._action_bar.set_last_message(self._live_session_vm.stimulus_progress_message())

    # ------------------------------------------------------------------
    # Stimulus submit path (§4.C)
    # ------------------------------------------------------------------

    def _on_stimulus_requested(self, operator_note: str) -> None:
        """ActionBar emitted — dispatch via the coordinator."""
        session_id = self._store.selected_session_id()
        if session_id is None:
            # Defensive: the ActionBar should be disabled in this state.
            return
        client_action_id = uuid4()
        request = StimulusRequest(
            operator_note=(operator_note or None),
            client_action_id=client_action_id,
        )
        # Optimistic transition to SUBMITTING — the action bar disables
        # itself so a second click is not possible; §4.C requires the
        # per-submission dedup key already, but the UX guard matters
        # because the operator can physically bang the Enter key.
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=StimulusActionState.SUBMITTING,
                client_action_id=client_action_id,
                operator_note=(operator_note or None),
                submitted_at_utc=datetime.now(UTC),
            )
        )

        signals = self._coordinator.submit_stimulus(session_id, request)
        key = str(client_action_id)
        self._inflight_stimulus[key] = signals
        signals.succeeded.connect(self._on_stimulus_succeeded)
        signals.failed.connect(self._on_stimulus_failed)
        # `finished` bookkeeping — free the signal-bus reference. Use
        # a per-submission key so concurrent submissions don't stomp
        # on each other's cleanup.
        signals.finished.connect(lambda _job, key=key: self._on_stimulus_finished(key))

    @Slot(str, object)
    def _on_stimulus_succeeded(self, _job: str, payload: object) -> None:
        if not isinstance(payload, StimulusAccepted):
            return
        if not payload.accepted:
            next_state = StimulusActionState.FAILED
        elif payload.stimulus_time_utc is not None:
            next_state = StimulusActionState.MEASURING
        else:
            next_state = StimulusActionState.ACCEPTED
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=next_state,
                client_action_id=payload.client_action_id,
                accepted_at_utc=payload.received_at_utc,
                authoritative_stimulus_time_utc=payload.stimulus_time_utc,
                message=payload.message,
            )
        )

    @Slot(str, object)
    def _on_stimulus_failed(self, _job: str, error: object) -> None:
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=StimulusActionState.FAILED,
                message=str(error),
            )
        )

    def _on_stimulus_finished(self, key: str) -> None:
        self._inflight_stimulus.pop(key, None)

    @Slot(object)
    def _on_stimulus_state_changed(self, ctx: object) -> None:
        if isinstance(ctx, StimulusUiContext):
            self._action_bar.set_action_state(ctx)
            self._update_action_bar_progress()

    @Slot(object)
    def _on_live_session_action_state_changed(self, _ctx: object) -> None:
        self._update_action_bar_progress()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        if hasattr(self, "_action_bar"):
            self._update_responsive_layout(event.size().width())

    def _update_responsive_layout(self, window_width: int) -> None:
        sidebar_width = min(
            _SIDEBAR_MAX_WIDTH,
            max(_SIDEBAR_MIN_WIDTH, int(window_width * _SIDEBAR_RATIO)),
        )
        if self._sidebar is not None:
            self._sidebar.setFixedWidth(sidebar_width)
        self._action_bar.set_compact_mode(window_width < _ACTION_BAR_COMPACT_WIDTH)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override
        """Hide immediately, then stop polling and join worker threads.

        Hiding before `coordinator.stop()` makes the X button feel
        instant: the operator sees the window disappear right away
        while the orphaned polling threads drain in the background.
        `processEvents()` forces the native hide to flush before we
        enter the short cooperative drain — otherwise the window can
        appear to freeze on screen during shutdown.
        """
        self.hide()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self._coordinator.stop()
        # Legacy scaffold pages exposed `shutdown()` — still call it on
        # any current page that implements it (belt and braces for
        # pages that may briefly re-appear while the
        # last scaffold view).
        for page in self._pages.values():
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        event.accept()


class _SignalBlocker:
    """Tiny context manager for ``QObject.blockSignals`` usage.

    PySide6's own `QSignalBlocker` works, but using a plain context
    manager keeps the override path explicit in this file and keeps
    the import surface small.
    """

    __slots__ = ("_obj", "_prev")

    def __init__(self, obj: QWidget) -> None:
        self._obj = obj
        self._prev = False

    def __enter__(self) -> None:
        self._prev = self._obj.blockSignals(True)

    def __exit__(self, *_: object) -> None:
        self._obj.blockSignals(self._prev)
