"""Current Operator Console shell catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShellSpec:
    name: str
    routes: tuple[str, ...]
    regions: tuple[str, ...]
    description: str


SHELL_SPECS: tuple[ShellSpec, ...] = (
    ShellSpec(
        name="SidebarStackShell",
        routes=(
            "overview",
            "live_session",
            "experiments",
            "physiology",
            "health",
            "sessions",
        ),
        regions=("sidebar", "stack", "action_bar", "status_bar"),
        description=(
            "Primary Operator Console shell: sidebar navigation, stacked page "
            "body, persistent ActionBar, and environment/API status bar."
        ),
    ),
    ShellSpec(
        name="MetricGridPlusTimelineShell",
        routes=("overview", "live_session", "physiology", "health"),
        regions=("header", "summary_grid", "timeline_or_followup"),
        description=(
            "Data-dense monitoring shell combining summary cards with a "
            "timeline, explanation panel, or follow-up readback surface."
        ),
    ),
    ShellSpec(
        name="TableWithDrillDownShell",
        routes=("live_session", "experiments", "sessions"),
        regions=("header", "table", "detail_or_manage"),
        description=(
            "Readback shell that keeps a table or list adjacent to a detail, "
            "update, or management surface."
        ),
    ),
)


SHELL_REGISTRY = {spec.name: spec for spec in SHELL_SPECS}
