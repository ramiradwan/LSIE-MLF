from __future__ import annotations

from services.operator_console.design_system.audit import (
    load_design_system_manifest,
    load_tokens_manifest,
    registered_object_names,
    stylesheet_object_names,
)
from services.operator_console.design_system.qss_builder import build_stylesheet
from services.operator_console.design_system.tokens import token_manifest


def test_registered_object_names_cover_sidebar_and_action_bar() -> None:
    names = registered_object_names(load_design_system_manifest())

    for name in (
        "SidebarNav",
        "SidebarTitle",
        "SidebarSubtitle",
        "ActionBar",
        "ActionBarSubmit",
        "StatusPill",
        "EventTimelineTable",
    ):
        assert name in names


def test_stylesheet_selectors_are_registered_in_manifest() -> None:
    manifest = load_design_system_manifest()
    registered = registered_object_names(manifest)
    selectors = stylesheet_object_names(build_stylesheet())

    assert selectors <= registered


def test_stylesheet_selector_extraction_ignores_hex_literals() -> None:
    selectors = stylesheet_object_names(build_stylesheet())

    for hex_fragment in ("e26a6a", "e7b34a", "ffffff"):
        assert hex_fragment not in selectors


def test_tokens_json_matches_token_manifest() -> None:
    assert load_tokens_manifest() == token_manifest()
