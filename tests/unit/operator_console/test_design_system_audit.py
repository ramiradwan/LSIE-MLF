from __future__ import annotations

from services.operator_console.design_system.audit import (
    load_design_system_manifest,
    load_tokens_manifest,
    registered_object_names,
    stylesheet_object_names,
)
from services.operator_console.design_system.qss_builder import (
    build_setup_stylesheet,
    build_stylesheet,
)
from services.operator_console.design_system.tokens import token_manifest


def test_registered_object_names_cover_sidebar_action_bar_and_launcher() -> None:
    names = registered_object_names(load_design_system_manifest())

    for name in (
        "SidebarNav",
        "SidebarTitle",
        "SidebarSubtitle",
        "ActionBar",
        "ActionBarSubmit",
        "StatusPill",
        "EventTimelineTable",
        "SetupRoot",
        "SetupPanel",
        "SetupProgress",
        "SetupLaunch",
    ):
        assert name in names


def test_stylesheet_selectors_are_registered_in_manifest() -> None:
    manifest = load_design_system_manifest()
    registered = registered_object_names(manifest)

    assert stylesheet_object_names(build_stylesheet()) <= registered
    assert stylesheet_object_names(build_setup_stylesheet()) <= registered


def test_stylesheet_selector_extraction_ignores_hex_literals() -> None:
    selectors = stylesheet_object_names(build_stylesheet())

    for hex_fragment in ("e26a6a", "e7b34a", "ffffff"):
        assert hex_fragment not in selectors


def test_tokens_json_matches_token_manifest() -> None:
    assert load_tokens_manifest() == token_manifest()
