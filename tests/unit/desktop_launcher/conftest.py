"""Qt scaffolding for desktop launcher unit tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app() -> Iterator[QCoreApplication]:
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
