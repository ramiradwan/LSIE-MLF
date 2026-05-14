"""Validated generic spec-work-item schema for local agent handoffs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ACTIVE_WORK_ITEM_PREFIX = "automation/work-items/active/"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcceptanceCriteria(_StrictModel):
    invariants: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    forbidden_changes: list[str] = Field(default_factory=list)


class SpecWorkItem(_StrictModel):
    type: Literal["spec_work_item"]
    title: str = Field(min_length=1)
    description: str | None = Field(default=None, min_length=1)
    spec_refs: list[str] = Field(min_length=1)
    source_artifacts: list[str] = Field(min_length=1)
    target_files: list[str] = Field(min_length=1)
    target_symbols: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    guarded_activation_risks: list[str] = Field(default_factory=list)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    local_artifacts: list[str] = Field(default_factory=list)

    @field_validator(
        "spec_refs",
        "source_artifacts",
        "target_files",
        "target_symbols",
        "dependencies",
        "guarded_activation_risks",
        "local_artifacts",
    )
    @classmethod
    def _reject_blank_values(cls, values: list[str]) -> list[str]:
        blanks = [value for value in values if not value.strip()]
        if blanks:
            raise ValueError("list values must not be blank")
        return values

    @model_validator(mode="after")
    def _validate_unique_targets(self) -> SpecWorkItem:
        duplicate_files = _duplicates(self.target_files)
        duplicate_symbols = _duplicates(self.target_symbols)
        if duplicate_files:
            raise ValueError("duplicate target_files values: " + ", ".join(duplicate_files))
        if duplicate_symbols:
            raise ValueError("duplicate target_symbols values: " + ", ".join(duplicate_symbols))
        return self

    @field_validator("local_artifacts")
    @classmethod
    def _validate_local_artifact_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            normalized = _normalize_repo_path(value)
            if normalized.startswith(ACTIVE_WORK_ITEM_PREFIX):
                continue
            raise ValueError("local_artifacts entries must live under " + ACTIVE_WORK_ITEM_PREFIX)
        return values


def _normalize_repo_path(value: str) -> str:
    normalized = str(PurePosixPath(value.replace("\\", "/")))
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_work_item_payload(payload: object) -> SpecWorkItem:
    return SpecWorkItem.model_validate(payload)


def load_work_item(path: str) -> SpecWorkItem:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON. Spec work-item validation is JSON-first."
        ) from exc
    return validate_work_item_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a generic spec work item.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a JSON spec work item. Reads stdin when omitted.",
    )
    args = parser.parse_args(argv)

    try:
        if args.path is None:
            raw_text = sys.stdin.read()
            payload = json.loads(raw_text)
            work_item = validate_work_item_payload(payload)
        else:
            work_item = load_work_item(args.path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(work_item.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
