#!/usr/bin/env python3
"""
check_schema_consistency.py — Cross-source schema drift detector
=================================================================

The v4.0 desktop pivot retired the PostgreSQL DDL surface (``data/sql/``
+ ``services.api.db.schema.SCHEMA_SQL``). Local persistence is now
SQLite-backed via :mod:`services.desktop_app.state.sqlite_schema`, where
intentional storage-class compromises (UUID and TIMESTAMPTZ both stored
as ``TEXT``, JSONB as ``TEXT``) make a column-type cross-check
meaningless. The cloud Postgres surface lands in WS5 P1; until then the
two contract-bearing surfaces are:

  1. **Pydantic models** in ``packages/schemas/`` — runtime payload validation
  2. **JSON Schema blocks** under ``interface_contracts.schemas`` in the
     extracted ``content.json`` payload from the committed
     ``docs/tech-spec-v*.pdf`` (the spec contract surface that the
     review agent loads)

This script normalizes both onto a common type / nullability vocabulary
and compares them entity-by-entity using an explicit registry that maps
a logical entity name (e.g. ``PhysiologicalSnapshot``) to its
representation in each source. It reports every field that:

  - is present in one source but absent in another,
  - has disagreeing canonical types across sources, or
  - has disagreeing nullability across sources.

A non-zero exit code is returned whenever any divergence is detected.

Usage::

    python scripts/check_schema_consistency.py

The implementation is split into pure helpers (``pydantic_to_entity``,
``parse_json_schema``, ``compare_entity``, ``check_consistency``) so
unit tests can seed each source with synthetic known-good or known-bad
payloads — see ``tests/unit/scripts/test_schema_consistency.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin
from uuid import UUID

if TYPE_CHECKING:
    from pydantic import BaseModel

# =====================================================================
# Canonical type vocabulary
# =====================================================================

CANONICAL_TYPES: frozenset[str] = frozenset(
    {
        "integer",
        "number",
        "string",
        "boolean",
        "datetime",
        "uuid",
        "object",
        "array",
        "unknown",
    }
)

# JSON Schema (type, format) → canonical
JSON_SCHEMA_TYPE_MAP: dict[tuple[str, str | None], str] = {
    ("integer", None): "integer",
    ("number", None): "number",
    ("string", None): "string",
    ("string", "date-time"): "datetime",
    ("string", "date"): "datetime",
    ("string", "uuid"): "uuid",
    ("boolean", None): "boolean",
    ("object", None): "object",
    ("array", None): "array",
}


# =====================================================================
# Internal data model
# =====================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Normalized representation of one field in one source."""

    name: str
    type: str
    nullable: bool


@dataclass
class EntitySpec:
    """Normalized representation of one entity (table / model / schema)."""

    name: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityMapping:
    """Maps a logical entity name to its identifier in each source."""

    name: str
    pydantic_class: str | None
    json_schema_key: str | None
    ignore_fields: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Inconsistency:
    """A single drift finding to report."""

    entity: str
    field: str
    kind: str  # 'missing' | 'type_mismatch' | 'nullability_mismatch'
    detail: str


# =====================================================================
# Default registry — what to compare and how
# =====================================================================


DEFAULT_REGISTRY: tuple[EntityMapping, ...] = (
    EntityMapping(
        name="PhysiologicalSnapshot",
        pydantic_class="packages.schemas.physiology:PhysiologicalSnapshot",
        json_schema_key="PhysiologicalSnapshot",
    ),
    EntityMapping(
        name="PhysiologicalChunkEvent",
        pydantic_class="packages.schemas.physiology:PhysiologicalChunkEvent",
        json_schema_key="PhysiologicalChunkEvent",
    ),
    EntityMapping(
        name="SemanticEvaluationResult",
        pydantic_class="packages.schemas.evaluation:SemanticEvaluationResult",
        json_schema_key="SemanticEvaluationResult",
    ),
)


# =====================================================================
# Pydantic loader
# =====================================================================


def load_pydantic_model(dotted: str) -> EntitySpec:
    """Resolve ``module.path:ClassName`` into an EntitySpec."""
    module_path, _, class_name = dotted.partition(":")
    if not class_name:
        raise ValueError(f"Expected 'module:Class', got {dotted!r}")
    from pydantic import BaseModel

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(f"{dotted!r} is not a Pydantic BaseModel subclass")
    return pydantic_to_entity(cls, name=class_name)


def pydantic_to_entity(cls: type[BaseModel], name: str) -> EntitySpec:
    """Convert a Pydantic BaseModel subclass to an EntitySpec."""
    entity = EntitySpec(name=name)
    for fname, field_info in cls.model_fields.items():
        canonical, nullable = _normalize_pydantic_annotation(field_info.annotation)
        entity.fields[fname] = FieldSpec(name=fname, type=canonical, nullable=nullable)
    return entity


def _normalize_pydantic_annotation(annotation: Any) -> tuple[str, bool]:
    """Return ``(canonical_type, nullable)`` for a Pydantic field annotation."""
    nullable = False
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional / Union with None → strip None and mark nullable.
    if origin is Union or origin is UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != len(args):
            nullable = True
        if len(non_none) == 1:
            annotation = non_none[0]
            origin = get_origin(annotation)
            args = get_args(annotation)
        else:
            return "unknown", nullable

    # Primitive types
    if annotation is int:
        return "integer", nullable
    if annotation is float:
        return "number", nullable
    if annotation is str:
        return "string", nullable
    if annotation is bool:
        return "boolean", nullable
    if annotation is datetime or annotation is date:
        return "datetime", nullable
    if annotation is UUID:
        return "uuid", nullable

    # Containers
    if origin in (list, tuple, set, frozenset):
        return "array", nullable
    if origin is dict:
        return "object", nullable

    # Literal[...] — infer canonical from the literal value's runtime type.
    if origin is Literal:
        sample = args[0] if args else None
        if isinstance(sample, bool):
            return "boolean", nullable
        if isinstance(sample, int):
            return "integer", nullable
        if isinstance(sample, float):
            return "number", nullable
        if isinstance(sample, str):
            return "string", nullable

    # Nested BaseModel → object
    try:
        from pydantic import BaseModel

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return "object", nullable
    except Exception:  # pragma: no cover — pydantic is a hard dep
        pass

    return "unknown", nullable


# =====================================================================
# JSON Schema loader
# =====================================================================


def parse_json_schema(schema: dict[str, Any], name: str) -> EntitySpec:
    """Convert one JSON Schema object to an EntitySpec."""
    entity = EntitySpec(name=name)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    for fname, fdef in properties.items():
        if not isinstance(fdef, dict):
            continue
        type_field = fdef.get("type")
        nullable = False

        if isinstance(type_field, list):
            non_null = [t for t in type_field if t != "null"]
            nullable = len(non_null) != len(type_field)
            type_str = non_null[0] if len(non_null) == 1 else "unknown"
        elif isinstance(type_field, str):
            type_str = type_field
        else:
            type_str = "unknown"

        fmt = fdef.get("format")
        canonical = JSON_SCHEMA_TYPE_MAP.get(
            (type_str, fmt),
            JSON_SCHEMA_TYPE_MAP.get((type_str, None), "unknown"),
        )

        # JSON Schema "absent from required" is the JSON-document analogue
        # of nullability for our cross-source comparison.
        if fname not in required:
            nullable = True

        entity.fields[fname] = FieldSpec(name=fname, type=canonical, nullable=nullable)
    return entity


def extract_json_schemas(content: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull the schemas dict out of supported content.json layouts."""
    contracts = content.get("interface_contracts") or {}
    if not isinstance(contracts, dict):
        return {}
    direct = contracts.get("schemas")
    if isinstance(direct, dict):
        return direct
    nested = contracts.get("schema_definition")
    if isinstance(nested, dict):
        inner = nested.get("schemas")
        if isinstance(inner, dict):
            return inner
    return {}


# =====================================================================
# Comparison
# =====================================================================


def compare_entity(
    entity_name: str,
    sources: Mapping[str, EntitySpec | None],
    ignore_fields: frozenset[str],
) -> list[Inconsistency]:
    """Compare one entity across all sources where it is defined."""
    issues: list[Inconsistency] = []
    available = {k: v for k, v in sources.items() if v is not None}
    if len(available) < 2:
        # Nothing to compare — single-source entities are by construction
        # consistent with themselves.
        return issues

    all_fields: set[str] = set()
    for spec in available.values():
        all_fields.update(spec.fields.keys())
    all_fields -= ignore_fields

    for fname in sorted(all_fields):
        present_in: dict[str, FieldSpec] = {}
        absent_in: list[str] = []
        for src, spec in available.items():
            if fname in spec.fields:
                present_in[src] = spec.fields[fname]
            else:
                absent_in.append(src)

        if absent_in:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=fname,
                    kind="missing",
                    detail=(f"present in {sorted(present_in)}, absent in {sorted(absent_in)}"),
                )
            )
            continue  # Don't double-flag missing fields.

        # Type comparison
        types = {src: spec.type for src, spec in present_in.items()}
        if len(set(types.values())) > 1:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=fname,
                    kind="type_mismatch",
                    detail=", ".join(f"{k}={v}" for k, v in sorted(types.items())),
                )
            )

        # Nullability comparison
        nulls = {src: spec.nullable for src, spec in present_in.items()}
        if len(set(nulls.values())) > 1:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=fname,
                    kind="nullability_mismatch",
                    detail=", ".join(
                        f"{k}={'nullable' if v else 'required'}" for k, v in sorted(nulls.items())
                    ),
                )
            )
    return issues


def check_consistency(
    registry: tuple[EntityMapping, ...],
    pydantic_entities: dict[str, EntitySpec],
    json_schema_entities: dict[str, EntitySpec],
) -> list[Inconsistency]:
    """Run ``compare_entity`` for every mapping in ``registry``."""
    all_issues: list[Inconsistency] = []
    for mapping in registry:
        sources: dict[str, EntitySpec | None] = {
            "pydantic": (pydantic_entities.get(mapping.name) if mapping.pydantic_class else None),
            "json_schema": (
                json_schema_entities.get(mapping.name) if mapping.json_schema_key else None
            ),
        }
        all_issues.extend(compare_entity(mapping.name, sources, mapping.ignore_fields))
    return all_issues


# =====================================================================
# Default source loading
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow ``python scripts/check_schema_consistency.py`` to import packages.* /
# services.* without requiring an editable install or PYTHONPATH=. shim.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_default_sources(
    registry: tuple[EntityMapping, ...] = DEFAULT_REGISTRY,
    repo_root: Path = REPO_ROOT,
) -> tuple[
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    list[str],
]:
    """Load both sources from their canonical locations in the repo."""
    warnings: list[str] = []

    # 1. Pydantic
    pydantic_entities: dict[str, EntitySpec] = {}
    for mapping in registry:
        if mapping.pydantic_class is None:
            continue
        try:
            pydantic_entities[mapping.name] = load_pydantic_model(mapping.pydantic_class)
        except Exception as exc:
            warnings.append(f"Could not load pydantic model {mapping.pydantic_class}: {exc}")

    # 2. content.json (the active extracted index for the committed spec)
    json_schema_entities: dict[str, EntitySpec] = {}
    candidates = [
        repo_root / "docs" / "content.json",
        repo_root / "content.json",
    ]
    content_path = next((p for p in candidates if p.is_file()), None)
    if content_path is None:
        warnings.append(f"content.json not found (looked at: {[str(p) for p in candidates]})")
    else:
        try:
            raw = content_path.read_text(encoding="utf-8").strip()
            content = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            warnings.append(f"content.json is not valid JSON ({content_path}): {exc}")
            content = {}
        if not content:
            warnings.append(f"content.json is empty: {content_path}")
        schemas = extract_json_schemas(content)
        for mapping in registry:
            if mapping.json_schema_key is None:
                continue
            sch = schemas.get(mapping.json_schema_key)
            if isinstance(sch, dict):
                json_schema_entities[mapping.name] = parse_json_schema(sch, mapping.name)

    return (
        pydantic_entities,
        json_schema_entities,
        warnings,
    )


# =====================================================================
# Reporting
# =====================================================================


def format_report(issues: list[Inconsistency], warnings: list[str]) -> str:
    """Render warnings and issues as a single human-readable string."""
    lines: list[str] = []
    bar = "=" * 64
    lines.append(bar)
    lines.append(" Schema Consistency Check")
    lines.append(bar)

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  ! {w}")

    if not issues:
        lines.append("")
        lines.append("No drift detected across pydantic and JSON Schema sources.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Inconsistencies: {len(issues)}")

    by_entity: dict[str, list[Inconsistency]] = {}
    for issue in issues:
        by_entity.setdefault(issue.entity, []).append(issue)

    for entity, group in sorted(by_entity.items()):
        lines.append("")
        lines.append(f"Entity: {entity}")
        for issue in group:
            lines.append(f"  [X] {issue.field:<28} {issue.kind:<24} {issue.detail}")
    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect schema drift between the LSIE-MLF Pydantic models and the "
            "extracted spec JSON Schema (content.json)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: detected from script location).",
    )
    args = parser.parse_args(argv)

    (
        pydantic_entities,
        json_schema_entities,
        warnings,
    ) = load_default_sources(DEFAULT_REGISTRY, args.repo_root)

    issues = check_consistency(
        DEFAULT_REGISTRY,
        pydantic_entities,
        json_schema_entities,
    )

    print(format_report(issues, warnings))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
