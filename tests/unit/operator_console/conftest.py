"""Qt test scaffolding for operator-console unit tests.

We do not rely on pytest-qt (not in the operator-host pin set). Instead
we create one session-scoped application instance — a full
``QApplication`` so widget tests can instantiate QWidget subclasses —
and rely on Qt's default direct-connection semantics to make signal
emission synchronous on the test thread.

``QT_QPA_PLATFORM=offscreen`` lets the widget layer bring up without
a display (headless CI, WSL, etc.).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app() -> Iterator[QCoreApplication]:
    # Default to the offscreen platform unless the host has set one
    # explicitly; this makes widget tests runnable on headless CI.
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
            # Let Qt drain queued deleteLater() before the process exits.
            app.processEvents()
