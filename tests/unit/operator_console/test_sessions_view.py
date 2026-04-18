"""Tests for the repurposed `SessionsView` + `SessionsViewModel` — Phase 10.

Phase 10 rewrites the scaffold Sessions page: no QThread, no ApiClient,
no local table model — the VM subscribes to `OperatorStore`, the view
binds to the shared `SessionsTableModel`, and double-click emits
`session_selected(UUID)` so the shell can jump into Live Session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from packages.schemas.operator_console import SessionSummary
from services.operator_console.state import OperatorStore
from services.operator_console.table_models.sessions_table_model import (
    SessionsTableModel,
)
from services.operator_console.viewmodels.sessions_vm import SessionsViewModel
from services.operator_console.views.sessions_view import SessionsView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _session(session_id: UUID | None = None) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v1",
        expected_greeting="hei rakas",
        duration_s=120.0,
    )


def _view() -> tuple[SessionsView, OperatorStore, SessionsViewModel]:
    store = OperatorStore()
    model = SessionsTableModel()
    vm = SessionsViewModel(store, model)
    return SessionsView(vm), store, vm


def test_sessions_view_empty_state_when_store_empty() -> None:
    view, _store, _vm = _view()
    # Offscreen QPA: parents aren't shown, so `isVisible()` is False.
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._body_container.isHidden() is True  # type: ignore[attr-defined]


def test_sessions_view_table_binds_to_shared_model() -> None:
    view, store, _vm = _view()
    sessions = [_session(), _session()]
    store.set_sessions(sessions)
    # The shared model drives the table — row count must match.
    assert view._table.model().rowCount() == len(sessions)  # type: ignore[attr-defined]
    assert view._empty_state.isHidden() is True  # type: ignore[attr-defined]


def test_sessions_vm_select_session_writes_store_and_emits() -> None:
    store = OperatorStore()
    model = SessionsTableModel()
    vm = SessionsViewModel(store, model)
    session_id = uuid4()
    received: list[object] = []
    vm.session_selected.connect(received.append)
    vm.select_session(session_id)
    assert store.selected_session_id() == session_id
    assert received == [session_id]


def test_sessions_view_forwards_vm_signal_to_shell() -> None:
    # The shell wires `session_selected` to its navigation handler; the
    # view must re-emit the VM's UUID payload so the shell doesn't need
    # to reach into the VM directly.
    view, _store, vm = _view()
    received: list[UUID] = []
    view.session_selected.connect(lambda sid: received.append(sid))
    target = uuid4()
    vm.select_session(target)
    assert received == [target]


def test_sessions_view_error_changed_shows_alert_banner() -> None:
    view, store, _vm = _view()
    store.set_error("sessions", "backend unreachable")
    assert view._error_banner.isHidden() is False  # type: ignore[attr-defined]


def test_sessions_view_no_longer_owns_qthread_or_apiclient() -> None:
    # Phase 10 invariant: the rewritten page must not carry the old
    # worker-thread machinery. Confirm the attributes are gone.
    view, _store, _vm = _view()
    assert not hasattr(view, "_worker")
    assert not hasattr(view, "_thread")
    assert not hasattr(view, "_api")
