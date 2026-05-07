"""Helpers for validating Operator Console design-system artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SELECTOR_RE = re.compile(r"#([A-Za-z_][A-Za-z0-9_]*)(?=\s*(?:[:{,]))")


def package_root() -> Path:
    return Path(__file__).resolve().parent


def design_system_manifest_path() -> Path:
    return package_root() / "design_system.json"


def tokens_manifest_path() -> Path:
    return package_root() / "tokens.json"


def load_design_system_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or design_system_manifest_path()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {manifest_path}")
    return payload


def load_tokens_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or tokens_manifest_path()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {manifest_path}")
    return payload


def registered_object_names(manifest: Mapping[str, Any] | None = None) -> set[str]:
    resolved = manifest or load_design_system_manifest()
    names: set[str] = set()
    for key in ("primitives", "compounds", "selectors"):
        entries = resolved.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            object_name = entry.get("object_name")
            if isinstance(object_name, str):
                names.add(object_name)
            object_names = entry.get("object_names")
            if isinstance(object_names, list):
                names.update(name for name in object_names if isinstance(name, str))
    return names


def stylesheet_object_names(stylesheet: str) -> set[str]:
    return {match.group(1) for match in _SELECTOR_RE.finditer(stylesheet)}
