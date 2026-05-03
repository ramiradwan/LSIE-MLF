#!/usr/bin/env python3
"""
check_schema_consistency.py — Cross-source schema drift detector
==============================================================

The active contract-bearing surfaces are:

  1. **Pydantic models** in ``packages/schemas/`` — runtime payload validation
  2. **JSON Schema blocks** under ``interface_contracts`` in the extracted
     ``content.json`` payload from the committed ``docs/tech-spec-v*.pdf``
  3. **Cloud PostgreSQL tables** loaded from ``services.cloud_api.db.schema``

This script normalizes each source onto a common type / requiredness /
nullability vocabulary and compares them entity-by-entity using an
explicit registry that maps a logical entity name (for example
``PosteriorDelta``) to its representation in each source. It reports
configured source entities that disappear entirely, fields that are
present in one source but absent from another, disagreeing canonical
types, disagreeing required-vs-optional semantics for JSON payloads,
and disagreeing explicit-nullability semantics across payload and SQL
surfaces.

A non-zero exit code is returned whenever any divergence is detected.

Usage::

    python scripts/check_schema_consistency.py

The implementation is split into pure helpers so unit tests can seed
each source with synthetic known-good or known-bad payloads — see
``tests/unit/scripts/test_schema_consistency.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

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

JSON_SCHEMA_TYPE_MAP: dict[tuple[str, str | None], str] = {
    ("integer", None): "integer",
    ("number", None): "number",
    ("string", None): "string",
    ("string", "date-time"): "datetime",
    ("string", "date"): "datetime",
    ("string", "uuid"): "uuid",
    ("string", "uuid4"): "uuid",
    ("boolean", None): "boolean",
    ("object", None): "object",
    ("array", None): "array",
}


@dataclass(frozen=True)
class FieldSpec:
    """Normalized representation of one field in one source."""

    name: str
    type: str
    nullable: bool
    required: bool | None = None


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
    sql_table: str | None = None
    ignore_fields: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Inconsistency:
    """A single drift finding to report."""

    entity: str
    field: str
    kind: str
    detail: str


DEFAULT_REGISTRY: tuple[EntityMapping, ...] = (
    EntityMapping(
        name="InferenceHandoffPayload",
        pydantic_class="packages.schemas.inference_handoff:InferenceHandoffPayload",
        json_schema_key="InferenceHandoffPayload",
        ignore_fields=frozenset({"_audio_transport", "_desktop_ipc_mode", "_x_max"}),
    ),
    EntityMapping(
        name="PhysiologicalSnapshot",
        pydantic_class="packages.schemas.physiology:PhysiologicalSnapshot",
        json_schema_key="PhysiologicalSnapshot",
    ),
    EntityMapping(
        name="PhysiologicalContext",
        pydantic_class="packages.schemas.physiology:PhysiologicalContext",
        json_schema_key="PhysiologicalContext",
    ),
    EntityMapping(
        name="PhysiologicalChunkEvent",
        pydantic_class="packages.schemas.physiology:PhysiologicalChunkEvent",
        json_schema_key="PhysiologicalChunkEvent",
    ),
    EntityMapping(
        name="BanditDecisionSnapshot",
        pydantic_class="packages.schemas.attribution:BanditDecisionSnapshot",
        json_schema_key="BanditDecisionSnapshot",
    ),
    EntityMapping(
        name="AttributionEvent",
        pydantic_class="packages.schemas.attribution:AttributionEvent",
        json_schema_key="AttributionEvent",
        sql_table="attribution_event",
    ),
    EntityMapping(
        name="OutcomeEvent",
        pydantic_class="packages.schemas.attribution:OutcomeEvent",
        json_schema_key="OutcomeEvent",
        sql_table="outcome_event",
    ),
    EntityMapping(
        name="EventOutcomeLink",
        pydantic_class="packages.schemas.attribution:EventOutcomeLink",
        json_schema_key="EventOutcomeLink",
        sql_table="event_outcome_link",
    ),
    EntityMapping(
        name="AttributionScore",
        pydantic_class="packages.schemas.attribution:AttributionScore",
        json_schema_key="AttributionScore",
        sql_table="attribution_score",
    ),
    EntityMapping(
        name="PosteriorDelta",
        pydantic_class="packages.schemas.cloud:PosteriorDelta",
        json_schema_key="PosteriorDelta",
        sql_table="posterior_delta_log",
        ignore_fields=frozenset({"received_at"}),
    ),
    EntityMapping(
        name="ExperimentBundle",
        pydantic_class="packages.schemas.cloud:ExperimentBundle",
        json_schema_key="ExperimentBundle",
    ),
)


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
    return parse_json_schema(cls.model_json_schema(by_alias=True), name=name)


def parse_json_schema(
    schema: dict[str, Any],
    name: str,
    root_schema: dict[str, Any] | None = None,
) -> EntitySpec:
    """Convert one JSON Schema object to an EntitySpec."""
    entity = EntitySpec(name=name)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    root = root_schema or schema

    for field_name, field_def in properties.items():
        if not isinstance(field_def, dict):
            continue
        canonical, nullable = _normalize_json_property(field_def, root)
        entity.fields[field_name] = FieldSpec(
            name=field_name,
            type=canonical,
            nullable=nullable,
            required=field_name in required,
        )
    return entity


def _normalize_json_property(schema: dict[str, Any], root_schema: dict[str, Any]) -> tuple[str, bool]:
    if ref := schema.get("$ref"):
        resolved = _resolve_json_ref(root_schema, ref)
        if isinstance(resolved, dict):
            return _normalize_json_property(resolved, root_schema)
        return "unknown", False

    type_field = schema.get("type")
    if isinstance(type_field, list):
        non_null = [item for item in type_field if item != "null"]
        nullable = len(non_null) != len(type_field)
        if len(non_null) == 1 and isinstance(non_null[0], str):
            return _canonical_json_type(non_null[0], schema), nullable
        return "unknown", nullable

    if isinstance(type_field, str):
        return _canonical_json_type(type_field, schema), False

    if isinstance(schema.get("properties"), dict) or "additionalProperties" in schema:
        return "object", False
    if "items" in schema:
        return "array", False

    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            return _normalize_json_union(options, root_schema)

    return "unknown", False


def _normalize_json_union(options: list[Any], root_schema: dict[str, Any]) -> tuple[str, bool]:
    nullable = False
    canonical_types: set[str] = set()

    for option in options:
        if not isinstance(option, dict):
            canonical_types.add("unknown")
            continue
        if option.get("type") == "null":
            nullable = True
            continue
        canonical, _ = _normalize_json_property(option, root_schema)
        canonical_types.add(canonical)

    if len(canonical_types) == 1:
        return next(iter(canonical_types)), nullable
    if not canonical_types:
        return "unknown", nullable
    return "unknown", nullable


def _canonical_json_type(type_name: str, schema: dict[str, Any]) -> str:
    fmt = schema.get("format")
    if type_name == "string" and fmt is None and _looks_like_uuid_pattern(schema.get("pattern")):
        return "uuid"
    return JSON_SCHEMA_TYPE_MAP.get(
        (type_name, fmt),
        JSON_SCHEMA_TYPE_MAP.get((type_name, None), "unknown"),
    )


def _looks_like_uuid_pattern(pattern: Any) -> bool:
    if not isinstance(pattern, str):
        return False
    lowered = pattern.lower()
    return "[0-9a-f]{8}-[0-9a-f]{4}-" in lowered and "[0-9a-f]{12}" in lowered


def _resolve_json_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None

    current: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current if isinstance(current, dict) else None


def extract_json_schemas(content: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull JSON Schema objects out of supported content.json layouts."""
    contracts = content.get("interface_contracts") or {}
    if not isinstance(contracts, dict):
        return {}

    schemas: dict[str, dict[str, Any]] = {}

    def register_schema(key: str, schema: dict[str, Any], root_schema: dict[str, Any]) -> None:
        schema_key = schema.get("title") if isinstance(schema.get("title"), str) else key
        schema_payload = schema
        shared_defs = root_schema.get("$defs")
        schema_defs = schema_payload.get("$defs")
        if isinstance(shared_defs, dict):
            if isinstance(schema_defs, dict):
                schema_payload = {**schema_payload, "$defs": {**shared_defs, **schema_defs}}
            elif "$defs" not in schema_payload:
                schema_payload = {**schema_payload, "$defs": shared_defs}
        existing = schemas.get(schema_key)
        if existing is None or (
            schema_payload.get("title") == schema_key and existing.get("title") != schema_key
        ):
            schemas[schema_key] = schema_payload

        for defs_key in ("$defs", "definitions"):
            defs = root_schema.get(defs_key)
            if not isinstance(defs, dict):
                continue
            for nested_key, nested_schema in defs.items():
                if not isinstance(nested_schema, dict):
                    continue
                if nested_key not in schemas:
                    register_schema(nested_key, nested_schema, root_schema)
                    continue
                existing_nested = schemas[nested_key]
                if "$defs" not in existing_nested and isinstance(shared_defs, dict):
                    schemas[nested_key] = {**existing_nested, "$defs": shared_defs}

    direct = contracts.get("schemas")
    if isinstance(direct, dict):
        for key, schema in direct.items():
            if isinstance(schema, dict):
                register_schema(key, schema, schema)

    nested = contracts.get("schema_definition")
    if isinstance(nested, dict):
        inner = nested.get("schemas")
        if isinstance(inner, dict):
            for key, schema in inner.items():
                if isinstance(schema, dict):
                    register_schema(key, schema, schema)

    for block in contracts.values():
        if not isinstance(block, dict) or block.get("language") != "json":
            continue
        source = block.get("source")
        if not isinstance(source, str):
            continue
        try:
            schema = json.loads(source)
        except json.JSONDecodeError:
            continue
        if isinstance(schema, dict):
            title = schema.get("title") if isinstance(schema.get("title"), str) else "schema"
            register_schema(title, schema, schema)

    return schemas


POSTGRES_TYPE_MAP: dict[str, str] = {
    "integer": "integer",
    "bigint": "integer",
    "bigserial": "integer",
    "double precision": "number",
    "text": "string",
    "text[]": "array",
    "uuid": "uuid",
    "timestamptz": "datetime",
    "timestamp": "datetime",
    "boolean": "boolean",
    "jsonb": "object",
}


def parse_postgres_sql_tables(sql_text: str) -> dict[str, EntitySpec]:
    tables: dict[str, EntitySpec] = {}
    for statement in sql_text.split(";"):
        lowered = statement.lower()
        marker = "create table if not exists"
        if marker not in lowered:
            continue
        name_start = lowered.index(marker) + len(marker)
        table_name = _extract_table_name(statement, name_start)
        if table_name is None:
            continue
        body_start = statement.find("(", name_start)
        body_end = statement.rfind(")")
        if body_start < 0 or body_end <= body_start:
            continue
        tables[table_name] = _parse_sql_table_body(table_name, statement[body_start + 1 : body_end])
    return tables


def _extract_table_name(statement: str, start: int) -> str | None:
    remainder = statement[start:].strip()
    if not remainder:
        return None
    return remainder.split()[0].strip('"').lower()


def _parse_sql_table_body(table_name: str, body: str) -> EntitySpec:
    entity = EntitySpec(name=table_name)
    for item in _split_sql_columns(body):
        line = " ".join(
            raw_line.strip()
            for raw_line in item.splitlines()
            if raw_line.strip() and not raw_line.strip().startswith("--")
        )
        if not line:
            continue
        first = line.split()[0].lower()
        if first in {"constraint", "primary", "unique", "foreign", "check", "alter"}:
            continue
        field = _parse_sql_column(line)
        if field is not None:
            entity.fields[field.name] = field
    return entity


def _split_sql_columns(body: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    in_single_quote = False
    index = 0
    while index < len(body):
        char = body[index]
        if char == "'":
            if in_single_quote and index + 1 < len(body) and body[index + 1] == "'":
                index += 2
                continue
            in_single_quote = not in_single_quote
        elif not in_single_quote:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            elif char == "," and depth == 0:
                items.append(body[start:index].strip())
                start = index + 1
        index += 1
    tail = body[start:].strip()
    if tail:
        items.append(tail)
    return items


def _parse_sql_column(line: str) -> FieldSpec | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    name = parts[0].strip('"').lower()
    type_token = _sql_type_token(parts[1:])
    canonical = POSTGRES_TYPE_MAP.get(type_token, "unknown")
    normalized = line.lower()
    nullable = "not null" not in normalized and "primary key" not in normalized
    return FieldSpec(name=name, type=canonical, nullable=nullable)


def _sql_type_token(parts: list[str]) -> str:
    first = parts[0].lower()
    if len(parts) >= 2 and f"{first} {parts[1].lower()}" in POSTGRES_TYPE_MAP:
        return f"{first} {parts[1].lower()}"
    return first


def compare_entity(
    entity_name: str,
    sources: Mapping[str, EntitySpec | None],
    ignore_fields: frozenset[str],
) -> list[Inconsistency]:
    """Compare one entity across all sources where it is defined."""
    issues: list[Inconsistency] = []
    available = {key: value for key, value in sources.items() if value is not None}
    if len(available) < 2:
        return issues

    all_fields: set[str] = set()
    for spec in available.values():
        all_fields.update(spec.fields.keys())
    all_fields -= ignore_fields

    for field_name in sorted(all_fields):
        present_in: dict[str, FieldSpec] = {}
        absent_in: list[str] = []
        for source_name, spec in available.items():
            if field_name in spec.fields:
                present_in[source_name] = spec.fields[field_name]
            else:
                absent_in.append(source_name)

        if absent_in:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=field_name,
                    kind="missing",
                    detail=f"present in {sorted(present_in)}, absent in {sorted(absent_in)}",
                )
            )
            continue

        types = {source_name: spec.type for source_name, spec in present_in.items()}
        if len(set(types.values())) > 1:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=field_name,
                    kind="type_mismatch",
                    detail=", ".join(
                        f"{source_name}={source_type}"
                        for source_name, source_type in sorted(types.items())
                    ),
                )
            )

        nulls = {source_name: spec.nullable for source_name, spec in present_in.items()}
        if len(set(nulls.values())) > 1:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=field_name,
                    kind="nullability_mismatch",
                    detail=", ".join(
                        f"{source_name}={'nullable' if nullable else 'non-null'}"
                        for source_name, nullable in sorted(nulls.items())
                    ),
                )
            )

        requiredness = {
            source_name: spec.required
            for source_name, spec in present_in.items()
            if spec.required is not None
        }
        if len(requiredness) >= 2 and len(set(requiredness.values())) > 1:
            issues.append(
                Inconsistency(
                    entity=entity_name,
                    field=field_name,
                    kind="requiredness_mismatch",
                    detail=", ".join(
                        f"{source_name}={'required' if required else 'optional'}"
                        for source_name, required in sorted(requiredness.items())
                    ),
                )
            )
    return issues


def check_consistency(
    registry: tuple[EntityMapping, ...],
    pydantic_entities: dict[str, EntitySpec],
    json_schema_entities: dict[str, EntitySpec],
    sql_entities: dict[str, EntitySpec] | None = None,
) -> list[Inconsistency]:
    """Run ``compare_entity`` for every mapping in ``registry``."""
    all_issues: list[Inconsistency] = []
    sql_entities = sql_entities or {}

    for mapping in registry:
        sources: dict[str, EntitySpec | None] = {
            "pydantic": (pydantic_entities.get(mapping.name) if mapping.pydantic_class else None),
            "json_schema": (
                json_schema_entities.get(mapping.name) if mapping.json_schema_key else None
            ),
            "sql": (sql_entities.get(mapping.sql_table) if mapping.sql_table else None),
        }

        missing_sources: list[str] = []
        if mapping.pydantic_class and sources["pydantic"] is None:
            missing_sources.append("pydantic")
        if mapping.json_schema_key and sources["json_schema"] is None:
            missing_sources.append("json_schema")
        if mapping.sql_table and sources["sql"] is None:
            missing_sources.append("sql")
        if missing_sources:
            all_issues.append(
                Inconsistency(
                    entity=mapping.name,
                    field="(entity)",
                    kind="missing_source",
                    detail=f"configured source missing: {sorted(missing_sources)}",
                )
            )

        all_issues.extend(compare_entity(mapping.name, sources, mapping.ignore_fields))

    return all_issues


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_default_sources(
    registry: tuple[EntityMapping, ...] = DEFAULT_REGISTRY,
    repo_root: Path = REPO_ROOT,
) -> tuple[
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    list[str],
]:
    """Load schema sources from their canonical locations in the repo."""
    warnings: list[str] = []

    pydantic_entities: dict[str, EntitySpec] = {}
    for mapping in registry:
        if mapping.pydantic_class is None:
            continue
        try:
            pydantic_entities[mapping.name] = load_pydantic_model(mapping.pydantic_class)
        except Exception as exc:
            warnings.append(f"Could not load pydantic model {mapping.pydantic_class}: {exc}")

    json_schema_entities: dict[str, EntitySpec] = {}
    candidates = [
        repo_root / "docs" / "content.json",
        repo_root / "content.json",
    ]
    content_path = next((path for path in candidates if path.is_file()), None)
    if content_path is None:
        warnings.append(f"content.json not found (looked at: {[str(path) for path in candidates]})")
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
            schema = schemas.get(mapping.json_schema_key)
            if isinstance(schema, dict):
                json_schema_entities[mapping.name] = parse_json_schema(schema, mapping.name, schema)
            else:
                warnings.append(
                    f"JSON schema {mapping.json_schema_key!r} not found in {content_path}"
                )

    sql_entities: dict[str, EntitySpec] = {}
    sql_path = repo_root / "services" / "cloud_api" / "db" / "schema.py"
    if any(mapping.sql_table for mapping in registry) and sql_path.is_file():
        try:
            from services.cloud_api.db.schema import SCHEMA_SQL

            sql_entities = parse_postgres_sql_tables(SCHEMA_SQL)
        except Exception as exc:
            warnings.append(f"Could not load cloud SQL schema {sql_path}: {exc}")

        for mapping in registry:
            if mapping.sql_table and mapping.sql_table not in sql_entities:
                warnings.append(f"SQL table {mapping.sql_table!r} not found in {sql_path}")

    return (
        pydantic_entities,
        json_schema_entities,
        sql_entities,
        warnings,
    )


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
        for warning in warnings:
            lines.append(f"  ! {warning}")

    if not issues:
        lines.append("")
        lines.append("No drift detected across configured schema sources.")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect schema drift between LSIE-MLF Pydantic models, the extracted "
            "spec JSON Schema, and cloud PostgreSQL tables."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: detected from script location).",
    )
    args = parser.parse_args(argv)

    pydantic_entities, json_schema_entities, sql_entities, warnings = load_default_sources(
        DEFAULT_REGISTRY,
        args.repo_root,
    )

    issues = check_consistency(
        DEFAULT_REGISTRY,
        pydantic_entities,
        json_schema_entities,
        sql_entities,
    )

    print(format_report(issues, warnings))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
