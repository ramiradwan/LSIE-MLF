"""Regression tests for `services.operator_console.theme` — Phase 6.

The theme is a string-returning factory so tests can assert that every
Phase-5 widget's object name appears in the composed QSS (otherwise a
rename drops styling silently), and that the §12 health-row classes
are present so the Health view can key off them.
"""

from __future__ import annotations

from services.operator_console.theme import STYLESHEET, build_stylesheet


def test_build_stylesheet_returns_non_empty_string() -> None:
    qss = build_stylesheet()
    assert isinstance(qss, str)
    assert len(qss) > 500


def test_stylesheet_covers_phase5_widget_object_names() -> None:
    qss = build_stylesheet()
    for name in (
        "#MetricCard",
        "#MetricCardTitle",
        "#MetricCardPrimary",
        "#MetricCardSecondary",
        "#SectionHeader",
        "#ActionBar",
        "#ActionBarSubmit",
        "#ActionBarCountdown",
        "#AlertBannerInfo",
        "#AlertBannerWarning",
        "#AlertBannerCritical",
        "#EmptyState",
        "#EmptyStateTitle",
        "#EventTimelineTable",
    ):
        assert name in qss, f"missing object-name rule for {name}"


def test_stylesheet_includes_section12_health_row_states() -> None:
    qss = build_stylesheet()
    # §12 demands degraded and recovering read distinctly from bad.
    for name in (
        "#HealthRowOk",
        "#HealthRowWarn",
        "#HealthRowBad",
        "#HealthRowDegraded",
        "#HealthRowRecovering",
    ):
        assert name in qss, f"missing health-row rule for {name}"


def test_module_level_constant_matches_factory() -> None:
    # Back-compat: older imports use the constant.
    assert build_stylesheet() == STYLESHEET
