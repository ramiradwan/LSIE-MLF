#!/usr/bin/env python3
"""
check_schema_consistency.py — Cross-source schema drift detector
=================================================================

The codebase carries four overlapping schema sources that are prone to
silent drift:

  1. **Pydantic models** in ``packages/schemas/`` (runtime payload validation)
  2. **PostgreSQL DDL** in ``data/sql/*.sql`` (mounted into the Persistent
     Store at container startup)
  3. **Python DDL string** ``services.api.db.schema.SCHEMA_SQL`` (used
     by the API Server's bootstrap path)
  4. **JSON Schema blocks** under ``interface_contracts.schemas`` in the
     ``content.json`` payload referenced by ``docs/tech-spec-v3.2.pdf``
     (the spec contract surface that the review agent loads)

This script loads all four, normalizes them onto a common type / nullability
vocabulary, and compares them entity-by-entity using an explicit registry
that maps a logical entity name (e.g. ``PhysiologicalSnapshot``) to its
representation in each source. It reports every field that:

  - is present in one source but absent in another,
  - has disagreeing canonical types across sources, or
  - has disagreeing nullability across sources.

A non-zero exit code is returned whenever any divergence is detected.

Usage::

    python scripts/check_schema_consistency.py

The implementation is split into pure helpers (``parse_sql_tables``,
``pydantic_to_entity``, ``parse_json_schema``, ``compare_entity``,
``check_consistency``) so unit tests can seed each source with synthetic
known-good or known-bad payloads — see
``tests/unit/scripts/test_schema_consistency.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
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

# PostgreSQL → canonical
SQL_TYPE_MAP: dict[str, str] = {
    "integer": "integer",
    "int": "integer",
    "int4": "integer",
    "int8": "integer",
    "bigint": "integer",
    "smallint": "integer",
    "bigserial": "integer",
    "serial": "integer",
    "double precision": "number",
    "real": "number",
    "numeric": "number",
    "decimal": "number",
    "float": "number",
    "float4": "number",
    "float8": "number",
    "text": "string",
    "varchar": "string",
    "character varying": "string",
    "char": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "timestamptz": "datetime",
    "timestamp": "datetime",
    "timestamp with time zone": "datetime",
    "timestamp without time zone": "datetime",
    "date": "datetime",
    "uuid": "uuid",
    "jsonb": "object",
    "json": "object",
}

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
    sql_table: str | None
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

# Auto-managed columns appear only in SQL: surrogate primary keys, FK
# routing columns the API code injects on insert, and DEFAULT NOW()
# audit timestamps. They are NOT part of the payload contract and must
# be excluded from the cross-source diff.
_AUTO_MANAGED_COLUMNS: frozenset[str] = frozenset(
    {"id", "session_id", "segment_id", "subject_role", "created_at"}
)


DEFAULT_REGISTRY: tuple[EntityMapping, ...] = (
    EntityMapping(
        name="PhysiologicalSnapshot",
        pydantic_class="packages.schemas.physiology:PhysiologicalSnapshot",
        sql_table="physiology_log",
        json_schema_key="PhysiologicalSnapshot",
        ignore_fields=_AUTO_MANAGED_COLUMNS,
    ),
    EntityMapping(
        name="PhysiologicalChunkEvent",
        pydantic_class="packages.schemas.physiology:PhysiologicalChunkEvent",
        sql_table=None,  # transit-only, never persisted (§5)
        json_schema_key="PhysiologicalChunkEvent",
        ignore_fields=frozenset(),
    ),
    EntityMapping(
        name="ComodulationLog",
        pydantic_class=None,  # no Pydantic mirror; analytics-only
        sql_table="comodulation_log",
        json_schema_key="ComodulationLog",
        ignore_fields=_AUTO_MANAGED_COLUMNS,
    ),
    EntityMapping(
        name="SemanticEvaluationResult",
        pydantic_class="packages.schemas.evaluation:SemanticEvaluationResult",
        sql_table=None,  # the 'evaluations' table re-shapes these fields
        json_schema_key="SemanticEvaluationResult",
        ignore_fields=frozenset(),
    ),
)


# =====================================================================
# SQL parser
# =====================================================================

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)


def _strip_sql_comments(sql: str) -> str:
    """Drop -- line comments and /* */ block comments before parsing."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def _split_top_level_commas(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are not inside parentheses."""
    chunks: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            chunks.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        chunks.append("".join(buf))
    return chunks


def _normalize_sql_type(type_part: str) -> str:
    """Map a raw SQL type fragment to a canonical type token."""
    type_lower = type_part.lower().strip()
    # Strip parenthesized modifiers: VARCHAR(255), NUMERIC(10,2), etc.
    type_lower = re.sub(r"\s*\([^)]*\)", "", type_lower).strip()
    return SQL_TYPE_MAP.get(type_lower, "unknown")


def parse_sql_tables(sql: str) -> dict[str, EntitySpec]:
    """Parse every CREATE TABLE in ``sql`` into ``{table_name: EntitySpec}``."""
    cleaned = _strip_sql_comments(sql)
    tables: dict[str, EntitySpec] = {}

    for table_match in _CREATE_TABLE_RE.finditer(cleaned):
        table_name = table_match.group(1).lower()
        body = table_match.group(2)
        entity = EntitySpec(name=table_name)

        for raw in _split_top_level_commas(body):
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            # Skip table-level constraints; only column definitions interest us.
            if upper.startswith(
                (
                    "CHECK",
                    "FOREIGN KEY",
                    "PRIMARY KEY",
                    "UNIQUE",
                    "CONSTRAINT",
                    "EXCLUDE",
                )
            ):
                continue

            m = re.match(r"(\w+)\s+(.+)", line, re.DOTALL)
            if not m:
                continue
            col_name = m.group(1).lower()
            rest = m.group(2)

            # Type runs from the column name to the first reserved suffix.
            type_part = re.split(
                r"\s+(?:NOT\s+NULL|NULL|DEFAULT|REFERENCES|CHECK|"
                r"PRIMARY\s+KEY|UNIQUE|GENERATED)",
                rest,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            canonical = _normalize_sql_type(type_part)
            nullable = not re.search(
                r"\bNOT\s+NULL\b|\bPRIMARY\s+KEY\b",
                rest,
                re.IGNORECASE,
            )
            entity.fields[col_name] = FieldSpec(name=col_name, type=canonical, nullable=nullable)

        tables[table_name] = entity
    return tables


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
    """Pull the schemas dict out of content.json under v3 or v3.1 layouts."""
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
    sql_file_tables: dict[str, EntitySpec],
    sql_string_tables: dict[str, EntitySpec],
    json_schema_entities: dict[str, EntitySpec],
) -> list[Inconsistency]:
    """Run ``compare_entity`` for every mapping in ``registry``."""
    all_issues: list[Inconsistency] = []
    for mapping in registry:
        sources: dict[str, EntitySpec | None] = {
            "pydantic": (pydantic_entities.get(mapping.name) if mapping.pydantic_class else None),
            "sql_file": (sql_file_tables.get(mapping.sql_table) if mapping.sql_table else None),
            "sql_string": (sql_string_tables.get(mapping.sql_table) if mapping.sql_table else None),
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
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    list[str],
]:
    """Load all four sources from their canonical locations in the repo."""
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

    # 2. SQL files on disk
    sql_dir = repo_root / "data" / "sql"
    sql_file_tables: dict[str, EntitySpec] = {}
    if sql_dir.is_dir():
        for sql_file in sorted(sql_dir.glob("*.sql")):
            sql_file_tables.update(parse_sql_tables(sql_file.read_text(encoding="utf-8")))
    else:
        warnings.append(f"SQL directory missing: {sql_dir}")

    # 3. Python DDL string
    sql_string_tables: dict[str, EntitySpec] = {}
    try:
        from services.api.db.schema import SCHEMA_SQL  # type: ignore

        sql_string_tables = parse_sql_tables(SCHEMA_SQL)
    except Exception as exc:
        warnings.append(f"Could not load services.api.db.schema.SCHEMA_SQL: {exc}")

    # 4. content.json (referenced by docs/tech-spec-v3.2.pdf)
    json_schema_entities: dict[str, EntitySpec] = {}
    candidates = [
        repo_root / "docs" / "artifacts" / "content.json",
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
        sql_file_tables,
        sql_string_tables,
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
        lines.append(
            "No drift detected across pydantic, SQL DDL, "
            "Python DDL string, and JSON Schema sources."
        )
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
            "Detect schema drift across the four LSIE-MLF schema sources "
            "(Pydantic, SQL files, Python DDL string, content.json JSON Schema)."
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
        sql_file_tables,
        sql_string_tables,
        json_schema_entities,
        warnings,
    ) = load_default_sources(DEFAULT_REGISTRY, args.repo_root)

    issues = check_consistency(
        DEFAULT_REGISTRY,
        pydantic_entities,
        sql_file_tables,
        sql_string_tables,
        json_schema_entities,
    )

    print(format_report(issues, warnings))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
