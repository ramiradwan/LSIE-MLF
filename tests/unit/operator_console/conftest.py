"""Qt test scaffolding for operator-console unit tests.

We do not rely on pytest-qt (not in the operator-host pin set). Instead
we create one session-scoped `QCoreApplication` — enough to let
`QObject` signals fire synchronously on the test thread — and rely on
Qt's default direct-connection semantics to make the tests blocking.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication


@pytest.fixture(scope="session")
def qt_app() -> Iterator[QCoreApplication]:
    app = QCoreApplication.instance()
    created_here = False
    if app is None:
        app = QCoreApplication([])
        created_here = True
    try:
        yield app
    finally:
        if created_here:
            # Let Qt drain queued deleteLater() before the process exits.
            app.processEvents()
