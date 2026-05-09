"""Tests for `MetricCard` — Phase 5.

Asserts the setters propagate and that `clicked` only fires after
`set_clickable(True)`.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from packages.schemas.operator_console import UiStatusKind
from services.operator_console.widgets.metric_card import MetricCard

pytestmark = pytest.mark.usefixtures("qt_app")


def _press(card: MetricCard) -> None:
    QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))


def test_setters_propagate() -> None:
    card = MetricCard("Active Session")
    card.set_primary_text("session abc")
    card.set_secondary_text("arm: greeting_v1")
    card.set_status(UiStatusKind.OK, "ok")
    assert card._secondary.isHidden() is False  # type: ignore[attr-defined]
    assert card._status.isHidden() is False  # type: ignore[attr-defined]
    assert card.minimumHeight() >= 132
    assert card.accessibleName() == "Active Session"
    assert "session abc" in card.accessibleDescription()
    assert "arm: greeting_v1" in card.accessibleDescription()


def test_set_status_none_hides_pill() -> None:
    card = MetricCard("Active Session")
    card.set_status(UiStatusKind.OK, "ok")
    card.set_status(UiStatusKind.OK, None)
    assert card._status.isHidden() is True  # type: ignore[attr-defined]


def test_clicked_signal_only_when_clickable() -> None:
    card = MetricCard("Overview")
    clicks: list[int] = []
    card.clicked.connect(lambda: clicks.append(1))

    _press(card)
    assert clicks == []

    card.set_clickable(True)
    _press(card)
    assert clicks == [1]
    assert card.focusPolicy() is Qt.FocusPolicy.StrongFocus
    assert card.toolTip() == "Open Overview"

    QTest.keyClick(card, Qt.Key.Key_Return)
    assert clicks == [1, 1]
