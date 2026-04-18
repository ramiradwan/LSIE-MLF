"""Qt + full-wiring scaffolding for operator-console integration tests.

Unit tests (`tests/unit/operator_console/`) hit one widget or viewmodel
at a time. These integration tests wire the full store → viewmodel →
table model → view chain to catch composition-level regressions that a
unit test would miss — e.g. a signal the store emits that the view
doesn't subscribe to, or a field the VM surfaces in the wrong card.

The helpers here build that chain and nothing else: no QMainWindow, no
coordinator, no real transport. Polling + navigation have their own
dedicated unit tests; the point of these is to assert that a seeded
`OperatorStore` lands real data in the right visible cell of the
rendered page.

Spec references:
  §4.E.1         — Operator Console is the production visualization surface
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app() -> Iterator[QCoreApplication]:
    # Headless CI / WSL / Git Bash all default to offscreen. Unit tests
    # use the same pattern; we duplicate the fixture here rather than
    # importing across the unit/integration boundary to keep the two
    # directories independently runnable.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance()
    created_here = False
    if app is None:
        app = QApplication([])
        created_here = True
    try:
        yield app
    finally:
        if created_here:
            app.processEvents()
