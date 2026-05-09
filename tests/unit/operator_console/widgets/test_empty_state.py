"""Tests for `EmptyStateWidget` — Phase 5."""

from __future__ import annotations

import pytest

from services.operator_console.widgets.empty_state import EmptyStateWidget

pytestmark = pytest.mark.usefixtures("qt_app")


def test_setters_apply() -> None:
    widget = EmptyStateWidget()
    widget.set_title("No session selected")
    widget.set_message("Select a session from the sidebar to begin.")
    assert widget._title.text() == "No session selected"  # type: ignore[attr-defined]
    assert widget._message.text() == (  # type: ignore[attr-defined]
        "Select a session from the sidebar to begin."
    )
    assert widget.accessibleName() == "No session selected"
    assert widget.accessibleDescription() == "Select a session from the sidebar to begin."
