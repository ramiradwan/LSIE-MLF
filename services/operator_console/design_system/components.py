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
            "MetricCardChevron",
        ),
        description=(
            "Summary card with title, primary value, optional secondary copy, "
            "optional status pill, and an optional chevron affordance when "
            "the card is clickable."
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
        description=(
            "Page or section title block with optional subtitle. The title "
            "carries a `level` property (page/panel/sub) so the same widget "
            "renders at the page heading size, the panel heading size, or the "
            "muted sub-grouping size."
        ),
    ),
    ComponentSpec(
        name="ResponsiveMetricGrid",
        kind="layout",
        object_names=(),
        description=(
            "Width-band-aware card grid that reflows between narrow, medium, and wide layouts."
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
            "ActionBarNoteToggle",
            "ActionBarProgress",
        ),
        description=(
            "Persistent shell-level stimulus rail mounted below stacked "
            "content. Includes the optional countdown progress strip pinned "
            "to the bottom edge and the compact-mode note toggle that "
            "expands the operator note inline."
        ),
    ),
)


COMPONENT_REGISTRY = {spec.name: spec for spec in COMPONENT_SPECS}
