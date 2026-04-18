"""Operator Console shell — sidebar + stacked content + persistent ActionBar.

Phase 6 rewrites the scaffold window around the Phase-4 `OperatorStore`
and `PollingCoordinator`. The window is the composition root for the
six operator pages; views are eager-instantiated into a
`QStackedWidget` so switching between routes is a cheap index change
(no re-polling, no re-wiring). A single `ActionBar` mounts once below
the content area per SPEC-AMEND-008's "persistent stimulus rail" —
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
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
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
from services.operator_console.theme import PALETTE
from services.operator_console.views.placeholder_view import PlaceholderView
from services.operator_console.widgets.action_bar import ActionBar
from services.operator_console.workers import OneShotSignals


@dataclass(frozen=True)
class _NavEntry:
    """One sidebar nav entry — the route it selects and its button label."""

    route: AppRoute
    label: str


# Declaration order is also rendering order in the sidebar and the stack.
# The operator's eye sweeps top-to-bottom: Overview first (at-a-glance),
# Live Session (the active action surface), then drill-downs, then the
# historical Sessions page last.
_NAV_SPEC: tuple[_NavEntry, ...] = (
    _NavEntry(AppRoute.OVERVIEW, "Overview"),
    _NavEntry(AppRoute.LIVE_SESSION, "Live Session"),
    _NavEntry(AppRoute.EXPERIMENTS, "Experiments"),
    _NavEntry(AppRoute.PHYSIOLOGY, "Physiology"),
    _NavEntry(AppRoute.HEALTH, "Health"),
    _NavEntry(AppRoute.SESSIONS, "Sessions"),
)


# Placeholder copy for each route — Phases 9/10 replace these views with
# their real implementations. The placeholder text is operator-facing
# language (what will live there), not developer-facing TODO notes.
_PLACEHOLDER_COPY: dict[AppRoute, tuple[str, str]] = {
    AppRoute.OVERVIEW: (
        "Overview",
        "Active session, experiment, physiology, health, and the attention "
        "queue in a single glance. Ships in Phase 9.",
    ),
    AppRoute.LIVE_SESSION: (
        "Live Session",
        "Encounter timeline with per-segment reward explanation "
        "(p90_intensity, semantic_gate, gated_reward, n_frames_in_window, "
        "baseline_b_neutral) and physiology freshness. Ships in Phase 9.",
    ),
    AppRoute.EXPERIMENTS: (
        "Experiments",
        "Thompson Sampling posteriors per arm with evaluation variance and "
        "a plain-language update summary. Ships in Phase 10.",
    ),
    AppRoute.PHYSIOLOGY: (
        "Physiology",
        "Operator and streamer RMSSD, heart rate, freshness, and the §7C "
        "Co-Modulation Index (null-valid is a legitimate outcome). "
        "Ships in Phase 10.",
    ),
    AppRoute.HEALTH: (
        "Health",
        "Subsystem status rollup with degraded vs recovering vs error "
        "distinctions and operator-action hints. Ships in Phase 10.",
    ),
    AppRoute.SESSIONS: (
        "Sessions",
        "Recent sessions with status, active arm, latest reward, and duration. Ships in Phase 10.",
    ),
}


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
        self.resize(1280, 800)

        self._pages: dict[AppRoute, QWidget] = {}
        self._nav_buttons: dict[AppRoute, QPushButton] = {}

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

        # Align the store's initial route with the sidebar's checked button
        # so the coordinator's first sync sees the right active route.
        self._store.set_route(_NAV_SPEC[0].route)
        self._update_action_bar_context()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _register_pages(self) -> None:
        """Eager-instantiate every route's page widget.

        Phase 6 uses `PlaceholderView` for all six routes; Phases 9/10
        swap in the real views. Eager instantiation keeps route changes
        cheap (no rebuild, no re-wiring) and lets the coordinator's
        poll lifecycle be the single source of truth for "is this page
        active?" via `on_route_changed`.
        """
        for entry in _NAV_SPEC:
            title, subtitle = _PLACEHOLDER_COPY[entry.route]
            self._pages[entry.route] = PlaceholderView(title, subtitle)

    def _build_stack(self) -> QStackedWidget:
        stack = QStackedWidget(self)
        for entry in _NAV_SPEC:
            stack.addWidget(self._pages[entry.route])
        return stack

    def _build_action_bar(self) -> ActionBar:
        bar = ActionBar(self)
        # No session yet — bar starts disabled. `_update_action_bar_context`
        # re-enables it when the store emits `selected_session_changed`.
        bar.set_session_context(None, None, None)
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame(self)
        sidebar.setObjectName("SidebarNav")
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(4)

        title = QLabel("LSIE-MLF", sidebar)
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; "
            f"color: {PALETTE.text_primary}; padding: 0 6px 4px 6px;"
        )
        subtitle = QLabel("Operator Console", sidebar)
        subtitle.setStyleSheet(f"color: {PALETTE.text_muted}; padding: 0 6px 18px 6px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._nav_group = QButtonGroup(sidebar)
        self._nav_group.setExclusive(True)
        for index, entry in enumerate(_NAV_SPEC):
            btn = QPushButton(entry.label, sidebar)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            if index == 0:
                btn.setChecked(True)
            self._nav_group.addButton(btn, index)
            self._nav_buttons[entry.route] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
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
            self._stack.setCurrentWidget(page)
        btn = self._nav_buttons.get(route)
        if btn is not None and not btn.isChecked():
            # Block signals so we don't re-emit clicked and bounce back
            # into _on_route_selected.
            with _SignalBlocker(btn):
                btn.setChecked(True)

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
        """Push session / arm / greeting into the ActionBar.

        The bar is only enabled when a session is selected AND the
        live-session DTO we have on hand describes that same session —
        otherwise we'd surface an arm/greeting from a stale context.
        """
        session_id = self._store.selected_session_id()
        live = self._store.live_session()
        active_arm: str | None = None
        expected_greeting: str | None = None
        if live is not None and session_id is not None and live.session_id == session_id:
            active_arm = live.active_arm
            expected_greeting = live.expected_greeting
        self._action_bar.set_session_context(session_id, active_arm, expected_greeting)

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
        next_state = (
            StimulusActionState.ACCEPTED if payload.accepted else StimulusActionState.FAILED
        )
        self._store.set_stimulus_ui_context(
            StimulusUiContext(
                state=next_state,
                client_action_id=payload.client_action_id,
                accepted_at_utc=payload.received_at_utc,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt override
        """Stop polling and join worker threads before letting Qt close.

        `coordinator.stop()` quits every polling thread and waits up to
        2s for each; anything beyond that is a bug that would have leaked
        a thread anyway, so we accept the close after it returns.
        """
        self._coordinator.stop()
        # Legacy scaffold pages exposed `shutdown()` — still call it on
        # any current page that implements it (belt and braces for
        # pages that may briefly re-appear before Phase 10 rewrites the
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
