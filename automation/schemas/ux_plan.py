"""Validated UX-plan schema for Operator Console UI work.

The repository does not currently carry a YAML parser dependency, so the CLI
accepts JSON payloads and prints normalized JSON for downstream work-item
creation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PageRoute = Literal[
    "overview",
    "live_session",
    "experiments",
    "physiology",
    "health",
    "sessions",
]
ShellName = Literal[
    "SidebarStackShell",
    "MetricGridPlusTimelineShell",
    "TableWithDrillDownShell",
]
ComponentName = Literal[
    "MetricCard",
    "StatusPill",
    "AlertBanner",
    "EmptyStateWidget",
    "EventTimelineWidget",
    "SectionHeader",
    "ResponsiveMetricGrid",
    "ActionBar",
]
RegionLayout = Literal["row", "column", "grid", "table", "stack"]
ResponsiveMode = Literal["hidden", "stacked", "collapsed", "visible"]
KeyboardKey = Literal["Return", "Space", "Escape", "Tab"]
ComponentPropValue = str | int | float | bool


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class A11ySpec(_StrictModel):
    accessible_name: str = Field(min_length=1)
    accessible_description: str | None = None
    focusable: bool = True
    keyboard_activation: list[KeyboardKey] = Field(default_factory=list)


class ResponsivePolicy(_StrictModel):
    narrow: ResponsiveMode = "visible"
    medium: ResponsiveMode = "visible"
    wide: ResponsiveMode = "visible"


class StateBindings(_StrictModel):
    empty: str | None = None
    loading: str | None = None
    error: str | None = None


class ComponentInstance(_StrictModel):
    component: ComponentName
    object_name: str = Field(pattern=r"^[A-Z][A-Za-z0-9_]*$")
    props: dict[str, ComponentPropValue] = Field(default_factory=dict)
    status_kind_source: str | None = None
    formatters: list[str] = Field(default_factory=list)
    a11y: A11ySpec
    states: StateBindings = Field(default_factory=StateBindings)
    responsive: ResponsivePolicy = Field(default_factory=ResponsivePolicy)


class UxRegion(_StrictModel):
    name: str = Field(min_length=1)
    layout: RegionLayout
    columns: int | None = Field(default=None, ge=1)
    components: list[ComponentInstance] = Field(min_length=1)


class UxPlan(_StrictModel):
    spec_refs: list[str] = Field(min_length=1)
    source_artifacts: list[str] = Field(min_length=1)
    page_route: PageRoute
    shell: ShellName
    viewmodel: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    regions: list[UxRegion] = Field(min_length=1)
    target_files: list[str] = Field(min_length=1)
    target_symbols: list[str] = Field(min_length=1)
    invariants: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_object_names(self) -> UxPlan:
        seen: set[str] = set()
        duplicates: list[str] = []
        for region in self.regions:
            for component in region.components:
                if component.object_name in seen:
                    duplicates.append(component.object_name)
                seen.add(component.object_name)
        if duplicates:
            unique_duplicates = sorted(set(duplicates))
            raise ValueError(
                "duplicate object_name values: " + ", ".join(unique_duplicates)
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_targets(self) -> UxPlan:
        duplicate_files = _duplicates(self.target_files)
        duplicate_symbols = _duplicates(self.target_symbols)
        if duplicate_files:
            raise ValueError(
                "duplicate target_files values: " + ", ".join(duplicate_files)
            )
        if duplicate_symbols:
            raise ValueError(
                "duplicate target_symbols values: " + ", ".join(duplicate_symbols)
            )
        return self


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_plan_payload(payload: object) -> UxPlan:
    return UxPlan.model_validate(payload)


def load_plan(path: Path) -> UxPlan:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON. UX plan validation is JSON-first in this repository."
        ) from exc
    return validate_plan_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Operator Console UX plan.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to a JSON UX plan. Reads stdin when omitted.",
    )
    args = parser.parse_args(argv)

    try:
        if args.path is None:
            raw_text = sys.stdin.read()
            payload = json.loads(raw_text)
            plan = validate_plan_payload(payload)
        else:
            plan = load_plan(args.path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(plan.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
