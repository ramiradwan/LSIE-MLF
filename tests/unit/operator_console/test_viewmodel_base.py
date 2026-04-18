"""Regression tests for `ViewModelBase` — Phase 7.

The base VM's contract is three signals + a debounce on `error_changed`.
Every page VM in Phase 8 inherits from this, so the wiring here has to
be rock-solid.
"""

from __future__ import annotations

import pytest

from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.base import ViewModelBase

pytestmark = pytest.mark.usefixtures("qt_app")


def test_store_accessor_returns_injected_store() -> None:
    store = OperatorStore()
    vm = ViewModelBase(store)
    assert vm.store is store


def test_emit_changed_fires_signal() -> None:
    store = OperatorStore()
    vm = ViewModelBase(store)
    hits = 0

    def on_change() -> None:
        nonlocal hits
        hits += 1

    vm.changed.connect(on_change)
    vm.emit_changed()
    vm.emit_changed()
    assert hits == 2


def test_emit_toast_forwards_level_and_message() -> None:
    store = OperatorStore()
    vm = ViewModelBase(store)
    received: list[tuple[str, str]] = []

    vm.toast_requested.connect(lambda level, msg: received.append((level, msg)))
    vm.emit_toast("warn", "Stimulus retrying")
    assert received == [("warn", "Stimulus retrying")]


def test_set_error_emits_only_on_change() -> None:
    store = OperatorStore()
    vm = ViewModelBase(store)
    emissions: list[str] = []
    vm.error_changed.connect(emissions.append)

    vm.set_error("connection refused")
    vm.set_error("connection refused")  # duplicate — must be suppressed
    vm.set_error(None)
    vm.set_error("")  # already empty — no emit
    vm.set_error("timeout")

    assert emissions == ["connection refused", "", "timeout"]
    assert vm.error() == "timeout"


def test_set_error_none_normalizes_to_empty_string() -> None:
    store = OperatorStore()
    vm = ViewModelBase(store)
    vm.set_error("oops")
    vm.set_error(None)
    assert vm.error() == ""
