"""Tests for `SectionHeader` — Phase 5."""

from __future__ import annotations

import pytest

from services.operator_console.widgets.section_header import SectionHeader

pytestmark = pytest.mark.usefixtures("qt_app")


def test_title_only_hides_subtitle() -> None:
    header = SectionHeader("Overview")
    assert header._title.text() == "Overview"  # type: ignore[attr-defined]
    assert header._subtitle.isHidden() is True  # type: ignore[attr-defined]


def test_with_subtitle_shows_subtitle() -> None:
    header = SectionHeader("Overview", "adaptive experiment snapshot")
    assert header._subtitle.isHidden() is False  # type: ignore[attr-defined]
    assert header._subtitle.text() == "adaptive experiment snapshot"  # type: ignore[attr-defined]


def test_setters_mutate_title_and_subtitle() -> None:
    header = SectionHeader("Overview")
    header.set_title("Live Session")
    header.set_subtitle("segment timeline")
    assert header._title.text() == "Live Session"  # type: ignore[attr-defined]
    assert header._subtitle.text() == "segment timeline"  # type: ignore[attr-defined]
    assert header._subtitle.isHidden() is False  # type: ignore[attr-defined]

    header.set_subtitle(None)
    assert header._subtitle.isHidden() is True  # type: ignore[attr-defined]
