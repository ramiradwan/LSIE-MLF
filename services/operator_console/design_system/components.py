"""Current shared Operator Console component registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str
    object_names: tuple[str, ...]
    description: str


COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        name="MetricCard",
        kind="primitive",
        object_names=(
            "MetricCard",
            "MetricCardTitle",
            "MetricCardPrimary",
            "MetricCardSecondary",
        ),
        description=(
            "Summary card with title, primary value, optional secondary copy, "
            "and optional status pill."
        ),
    ),
    ComponentSpec(
        name="StatusPill",
        kind="primitive",
        object_names=("StatusPill", "StatusPillLabel"),
        description="Colored dot plus short text label keyed from UiStatusKind.",
    ),
    ComponentSpec(
        name="AlertBanner",
        kind="primitive",
        object_names=(
            "AlertBanner",
            "AlertBannerInfo",
            "AlertBannerWarning",
            "AlertBannerCritical",
            "AlertBannerGlyph",
            "AlertBannerMessage",
        ),
        description="Severity-coded banner for scoped page feedback.",
    ),
    ComponentSpec(
        name="EmptyStateWidget",
        kind="primitive",
        object_names=("EmptyState", "EmptyStateTitle", "EmptyStateMessage"),
        description="Centered empty-state placeholder used when no current data exists.",
    ),
    ComponentSpec(
        name="EventTimelineWidget",
        kind="primitive",
        object_names=("EventTimeline", "EventTimelineTable"),
        description="Responsive table wrapper for live timelines and alert history.",
    ),
    ComponentSpec(
        name="SectionHeader",
        kind="primitive",
        object_names=("SectionHeader", "SectionHeaderTitle", "SectionHeaderSubtitle"),
        description="Page or section title block with optional subtitle.",
    ),
    ComponentSpec(
        name="ResponsiveMetricGrid",
        kind="layout",
        object_names=(),
        description=(
            "Width-band-aware card grid that reflows between narrow, medium, "
            "and wide layouts."
        ),
    ),
    ComponentSpec(
        name="ActionBar",
        kind="compound",
        object_names=(
            "ActionBar",
            "ActionBarSession",
            "ActionBarGreeting",
            "ActionBarNote",
            "ActionBarSubmit",
            "ActionBarCountdown",
            "ActionBarMessage",
        ),
        description="Persistent shell-level stimulus rail mounted below stacked content.",
    ),
)


COMPONENT_REGISTRY = {spec.name: spec for spec in COMPONENT_SPECS}
