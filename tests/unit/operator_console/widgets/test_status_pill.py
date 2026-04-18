"""Tests for `StatusPill` — Phase 5.

The pill is a leaf widget: tests only assert that text/kind setters
round-trip through the underlying label + internal dot without
crashing. Colour mapping itself is a theme concern, not a contract.
"""

from __future__ import annotations

import pytest

from packages.schemas.operator_console import UiStatusKind
from services.operator_console.widgets.status_pill import StatusPill

pytestmark = pytest.mark.usefixtures("qt_app")


def test_default_kind_is_neutral() -> None:
    pill = StatusPill()
    assert pill.kind() is UiStatusKind.NEUTRAL
    assert pill.text() == ""


def test_set_text_and_kind() -> None:
    pill = StatusPill()
    pill.set_text("Live · 3 session(s)")
    pill.set_kind(UiStatusKind.OK)
    assert pill.text() == "Live · 3 session(s)"
    assert pill.kind() is UiStatusKind.OK


def test_set_kind_is_idempotent() -> None:
    pill = StatusPill()
    pill.set_kind(UiStatusKind.WARN)
    first_kind = pill.kind()
    pill.set_kind(UiStatusKind.WARN)
    assert pill.kind() is first_kind
