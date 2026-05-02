"""Tests for ``scripts.check_schema_consistency``.

These tests seed the schema sources (Pydantic models, JSON Schema
blocks from the extracted spec content, and optional cloud SQL tables)
with synthetic known-good and known-bad combinations to verify the detector catches the specific
drift cases it is designed to catch:

  * Field present in one source but missing from another
  * Field whose canonical type disagrees across sources
  * Field whose nullability disagrees across sources
  * Single-source entities are not falsely flagged
  * The all-aligned case yields zero issues and exit code 0

The v4.0 desktop pivot retired the local PostgreSQL DDL surface
(``data/sql/`` + ``services.api.db.schema.SCHEMA_SQL``); see the
docstring of ``scripts/check_schema_consistency.py`` for context. The
SQLite local store ports those columns as ``TEXT`` for UUID/datetime/
JSONB by design, so the SQL source here is limited to the restored WS5
cloud PostgreSQL bootstrap.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from packages.schemas.attribution import (
    AttributionEvent,
    AttributionScore,
    EventOutcomeLink,
    OutcomeEvent,
)
from packages.schemas.inference_handoff import physiological_sample_event_schema
from packages.schemas.physiology import PhysiologicalChunkEvent
from scripts.check_schema_consistency import (
    DEFAULT_REGISTRY,
    EntityMapping,
    EntitySpec,
    FieldSpec,
    Inconsistency,
    _normalize_pydantic_annotation,
    check_consistency,
    compare_entity,
    extract_json_schemas,
    format_report,
    main,
    parse_json_schema,
    parse_postgres_sql_tables,
    pydantic_to_entity,
)

# =====================================================================
# Pydantic loader
# =====================================================================


class _ChildModel(BaseModel):
    inner: int


class _Sample(BaseModel):
    rmssd_ms: float | None = Field(None, ge=0.0)
    freshness_s: float = Field(..., ge=0.0)
    is_stale: bool
    provider: Literal["oura"] = "oura"
    when: datetime
    sub_id: UUID
    tags: list[str] = Field(default_factory=list)
    child: _ChildModel | None = None


class TestPydanticLoader:
    def test_round_trip_through_pydantic_to_entity(self) -> None:
        entity = pydantic_to_entity(_Sample, name="Sample")

        assert entity.fields["rmssd_ms"] == FieldSpec("rmssd_ms", "number", True)
        assert entity.fields["freshness_s"] == FieldSpec("freshness_s", "number", False)
        assert entity.fields["is_stale"] == FieldSpec("is_stale", "boolean", False)
        assert entity.fields["provider"] == FieldSpec("provider", "string", False)
        assert entity.fields["when"] == FieldSpec("when", "datetime", False)
        assert entity.fields["sub_id"] == FieldSpec("sub_id", "uuid", False)
        assert entity.fields["tags"] == FieldSpec("tags", "array", False)
        assert entity.fields["child"] == FieldSpec("child", "object", True)

    def test_normalize_optional_union(self) -> None:
        assert _normalize_pydantic_annotation(int | None) == ("integer", True)
        assert _normalize_pydantic_annotation(str) == ("string", False)
        assert _normalize_pydantic_annotation(float) == ("number", False)
        assert _normalize_pydantic_annotation(bool) == ("boolean", False)

    def test_normalize_unsupported_returns_unknown(self) -> None:
        canonical, nullable = _normalize_pydantic_annotation(complex)
        assert canonical == "unknown"
        assert nullable is False


# =====================================================================
# JSON Schema loader
# =====================================================================


class TestJsonSchemaLoader:
    def test_required_vs_optional_maps_to_nullability(self) -> None:
        schema = {
            "type": "object",
            "required": ["freshness_s", "is_stale"],
            "properties": {
                "rmssd_ms": {"type": "number"},
                "freshness_s": {"type": "number"},
                "is_stale": {"type": "boolean"},
                "when": {"type": "string", "format": "date-time"},
                "sub_id": {"type": "string", "format": "uuid"},
                "score": {"type": ["number", "null"]},
            },
        }
        entity = parse_json_schema(schema, name="Sample")

        assert entity.fields["rmssd_ms"] == FieldSpec("rmssd_ms", "number", True)
        assert entity.fields["freshness_s"] == FieldSpec("freshness_s", "number", False)
        assert entity.fields["is_stale"] == FieldSpec("is_stale", "boolean", False)
        assert entity.fields["when"] == FieldSpec("when", "datetime", True)
        assert entity.fields["sub_id"] == FieldSpec("sub_id", "uuid", True)
        # score: not in required → nullable; type list also signals nullable
        assert entity.fields["score"].type == "number"
        assert entity.fields["score"].nullable is True

    def test_extract_schemas_handles_supported_layouts(self) -> None:
        direct = {"interface_contracts": {"schemas": {"Foo": {"type": "object"}}}}
        nested = {
            "interface_contracts": {"schema_definition": {"schemas": {"Bar": {"type": "object"}}}}
        }
        extracted_block = {
            "interface_contracts": {
                "posterior_delta_schema": {
                    "language": "json",
                    "source": '{"title":"PosteriorDelta","type":"object"}',
                }
            }
        }
        assert extract_json_schemas(direct) == {"Foo": {"type": "object"}}
        assert extract_json_schemas(nested) == {"Bar": {"type": "object"}}
        assert extract_json_schemas(extracted_block) == {
            "PosteriorDelta": {"title": "PosteriorDelta", "type": "object"}
        }
        assert extract_json_schemas({}) == {}
        assert extract_json_schemas({"interface_contracts": "not-a-dict"}) == {}

    def test_load_default_sources_prefers_active_content_index(self, tmp_path: Path) -> None:
        import json

        from scripts.check_schema_consistency import load_default_sources

        active_content = {
            "interface_contracts": {
                "schemas": {
                    "ActiveSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                }
            }
        }
        frozen_content = {
            "interface_contracts": {
                "schemas": {
                    "FrozenSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    }
                }
            }
        }

        docs_dir = tmp_path / "docs"
        artifacts_dir = docs_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (docs_dir / "content.json").write_text(json.dumps(active_content), encoding="utf-8")
        (artifacts_dir / "content.json").write_text(json.dumps(frozen_content), encoding="utf-8")

        registry = (
            EntityMapping(
                name="ActiveSchema",
                pydantic_class=None,
                json_schema_key="ActiveSchema",
            ),
            EntityMapping(
                name="FrozenSchema",
                pydantic_class=None,
                json_schema_key="FrozenSchema",
            ),
        )

        _, json_schema_entities, sql_entities, warnings = load_default_sources(registry, tmp_path)

        assert "ActiveSchema" in json_schema_entities
        assert sql_entities == {}
        assert "FrozenSchema" not in json_schema_entities
        assert not any("docs/artifacts/content.json" in warning for warning in warnings)


# =====================================================================
# PostgreSQL SQL loader
# =====================================================================


class TestPostgresSqlLoader:
    def test_parse_posterior_delta_log_table(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS posterior_delta_log (
            event_id UUID PRIMARY KEY,
            experiment_id INTEGER NOT NULL,
            arm_id TEXT NOT NULL,
            delta_alpha DOUBLE PRECISION NOT NULL CHECK (delta_alpha >= 0.0),
            delta_beta DOUBLE PRECISION NOT NULL CHECK (delta_beta >= 0.0),
            segment_id TEXT NOT NULL CHECK (segment_id ~ '^[a-f0-9]{64}$'),
            client_id TEXT NOT NULL,
            applied_at_utc TIMESTAMPTZ NOT NULL,
            decision_context_hash TEXT CHECK (decision_context_hash IS NULL),
            received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (segment_id, client_id, arm_id)
        );
        """

        tables = parse_postgres_sql_tables(sql)
        entity = tables["posterior_delta_log"]

        assert entity.fields["event_id"] == FieldSpec("event_id", "uuid", False)
        assert entity.fields["experiment_id"] == FieldSpec("experiment_id", "integer", False)
        assert entity.fields["delta_alpha"] == FieldSpec("delta_alpha", "number", False)
        assert entity.fields["delta_beta"] == FieldSpec("delta_beta", "number", False)
        assert entity.fields["applied_at_utc"] == FieldSpec("applied_at_utc", "datetime", False)
        assert entity.fields["decision_context_hash"] == FieldSpec(
            "decision_context_hash", "string", True
        )
        assert entity.fields["received_at"] == FieldSpec("received_at", "datetime", False)
        assert "unique" not in entity.fields

    def test_sql_source_participates_when_mapping_names_table(self) -> None:
        pyd = {
            "PosteriorDelta": _spec(
                ("event_id", "uuid", False),
                ("delta_alpha", "number", False),
            )
        }
        jsn = {
            "PosteriorDelta": _spec(
                ("event_id", "uuid", False),
                ("delta_alpha", "number", False),
            )
        }
        sql = {
            "posterior_delta_log": _spec(
                ("event_id", "uuid", False),
                ("delta_alpha", "integer", False),
            )
        }
        registry = (
            EntityMapping(
                name="PosteriorDelta",
                pydantic_class="packages.schemas.cloud:PosteriorDelta",
                json_schema_key="PosteriorDelta",
                sql_table="posterior_delta_log",
            ),
        )

        issues = check_consistency(registry, pyd, jsn, sql)

        assert len(issues) == 1
        assert issues[0].field == "delta_alpha"
        assert issues[0].kind == "type_mismatch"
        assert "sql=integer" in issues[0].detail


# =====================================================================
# Comparison primitive
# =====================================================================


def _spec(*pairs: tuple[str, str, bool]) -> EntitySpec:
    """Helper: build an EntitySpec from (name, type, nullable) tuples."""
    e = EntitySpec(name="x")
    for name, typ, nullable in pairs:
        e.fields[name] = FieldSpec(name, typ, nullable)
    return e


class TestCompareEntity:
    def test_all_aligned_yields_no_issues(self) -> None:
        spec = _spec(
            ("rmssd_ms", "number", True),
            ("freshness_s", "number", False),
        )
        sources = {
            "pydantic": spec,
            "json_schema": spec,
        }
        assert compare_entity("E", sources, frozenset()) == []

    def test_field_missing_in_one_source(self) -> None:
        full = _spec(
            ("rmssd_ms", "number", True),
            ("provider", "string", False),
        )
        partial = _spec(("rmssd_ms", "number", True))
        issues = compare_entity(
            "E",
            {"pydantic": full, "json_schema": partial},
            frozenset(),
        )
        assert len(issues) == 1
        assert issues[0].field == "provider"
        assert issues[0].kind == "missing"
        assert "json_schema" in issues[0].detail

    def test_type_mismatch_is_caught(self) -> None:
        as_int = _spec(("heart_rate_bpm", "integer", False))
        as_num = _spec(("heart_rate_bpm", "number", False))
        issues = compare_entity(
            "E",
            {"pydantic": as_int, "json_schema": as_num},
            frozenset(),
        )
        kinds = [i.kind for i in issues]
        assert "type_mismatch" in kinds
        type_issue = next(i for i in issues if i.kind == "type_mismatch")
        assert "json_schema=number" in type_issue.detail
        assert "pydantic=integer" in type_issue.detail

    def test_nullability_mismatch_is_caught(self) -> None:
        nullable = _spec(("rmssd_ms", "number", True))
        required = _spec(("rmssd_ms", "number", False))
        issues = compare_entity(
            "E",
            {"pydantic": nullable, "json_schema": required},
            frozenset(),
        )
        kinds = [i.kind for i in issues]
        assert "nullability_mismatch" in kinds
        null_issue = next(i for i in issues if i.kind == "nullability_mismatch")
        assert "pydantic=nullable" in null_issue.detail
        assert "json_schema=required" in null_issue.detail

    def test_missing_field_does_not_also_flag_type_or_nullability(self) -> None:
        full = _spec(("x", "integer", False))
        empty = EntitySpec(name="x")
        issues = compare_entity(
            "E",
            {"pydantic": full, "json_schema": empty},
            frozenset(),
        )
        # Exactly one finding (the missing field), not three.
        assert len(issues) == 1
        assert issues[0].kind == "missing"

    def test_ignore_fields_are_skipped(self) -> None:
        with_extra = _spec(
            ("internal_marker", "integer", False),
            ("rmssd_ms", "number", True),
        )
        without_extra = _spec(("rmssd_ms", "number", True))
        issues = compare_entity(
            "E",
            {"pydantic": with_extra, "json_schema": without_extra},
            frozenset({"internal_marker"}),
        )
        assert issues == []

    def test_single_source_entity_not_flagged(self) -> None:
        """An entity defined in only one source has nothing to drift against."""
        only_pydantic = _spec(("x", "integer", False))
        issues = compare_entity(
            "E",
            {"pydantic": only_pydantic, "json_schema": None},
            frozenset(),
        )
        assert issues == []


# =====================================================================
# Top-level check_consistency
# =====================================================================


class _SnapPydantic(BaseModel):
    rmssd_ms: float | None = None
    freshness_s: float
    is_stale: bool
    provider: str


def _snap_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["freshness_s", "is_stale", "provider"],
        "properties": {
            "rmssd_ms": {"type": "number"},
            "freshness_s": {"type": "number"},
            "is_stale": {"type": "boolean"},
            "provider": {"type": "string"},
        },
    }


_TEST_REGISTRY = (
    EntityMapping(
        name="PhysiologicalSnapshot",
        pydantic_class="tests.unit.scripts.test_schema_consistency:_SnapPydantic",
        json_schema_key="PhysiologicalSnapshot",
    ),
)


def _build_sources(
    pydantic_cls: type[BaseModel] = _SnapPydantic,
    json_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, EntitySpec], dict[str, EntitySpec]]:
    pyd = {"PhysiologicalSnapshot": pydantic_to_entity(pydantic_cls, "PhysiologicalSnapshot")}
    schema_block = json_schema if json_schema is not None else _snap_json_schema()
    json_entities = {
        "PhysiologicalSnapshot": parse_json_schema(schema_block, "PhysiologicalSnapshot"),
    }
    return pyd, json_entities


class TestCheckConsistencyEndToEnd:
    def test_known_good_aligned_sources_yield_zero_issues(self) -> None:
        pyd, jsn = _build_sources()
        assert check_consistency(_TEST_REGISTRY, pyd, jsn) == []

    def test_pydantic_drift_field_added_is_detected(self) -> None:
        class Drifted(BaseModel):
            rmssd_ms: float | None = None
            freshness_s: float
            is_stale: bool
            provider: str
            extra_field: int  # added in Pydantic but absent from JSON Schema

        pyd, jsn = _build_sources(pydantic_cls=Drifted)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)

        missing = [i for i in issues if i.field == "extra_field"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "pydantic" in missing[0].detail
        assert "json_schema" in missing[0].detail

    def test_json_schema_drift_required_to_optional_is_detected(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["required"] = ["is_stale", "provider"]  # drop freshness_s

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        nulls = [i for i in issues if i.kind == "nullability_mismatch" and i.field == "freshness_s"]
        assert len(nulls) == 1
        assert "json_schema=nullable" in nulls[0].detail
        assert "pydantic=required" in nulls[0].detail

    def test_json_schema_field_missing_from_pydantic(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["properties"]["coverage_ratio"] = {"type": "number"}
        bad_schema["required"].append("coverage_ratio")

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        missing = [i for i in issues if i.field == "coverage_ratio"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "json_schema" in missing[0].detail
        assert "pydantic" in missing[0].detail

    def test_json_schema_drift_type_change_is_detected(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["properties"]["rmssd_ms"] = {"type": "integer"}

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        types = [i for i in issues if i.kind == "type_mismatch" and i.field == "rmssd_ms"]
        assert len(types) == 1
        assert "json_schema=integer" in types[0].detail
        assert "pydantic=number" in types[0].detail


# =====================================================================
# Reporting & main()
# =====================================================================


class TestReporting:
    def test_clean_report_has_no_inconsistencies_block(self) -> None:
        out = format_report(issues=[], warnings=[])
        assert "No drift detected" in out
        assert "Inconsistencies" not in out

    def test_warnings_are_rendered(self) -> None:
        out = format_report(issues=[], warnings=["content.json is empty"])
        assert "Warnings:" in out
        assert "content.json is empty" in out

    def test_issues_are_grouped_by_entity(self) -> None:
        issues = [
            Inconsistency("A", "x", "missing", "absent in pydantic"),
            Inconsistency("B", "y", "type_mismatch", "pydantic=integer, json_schema=number"),
            Inconsistency("A", "z", "type_mismatch", "pydantic=integer, json_schema=string"),
        ]
        out = format_report(issues=issues, warnings=[])
        assert "Entity: A" in out
        assert "Entity: B" in out
        assert "Inconsistencies: 3" in out


class TestMain:
    def test_clean_repo_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from scripts import check_schema_consistency as mod

        def fake_loader(
            registry: tuple[EntityMapping, ...], repo_root: Path
        ) -> tuple[
            dict[str, EntitySpec],
            dict[str, EntitySpec],
            dict[str, EntitySpec],
            list[str],
        ]:
            pyd, jsn = _build_sources()
            return pyd, jsn, {}, []

        monkeypatch.setattr(mod, "load_default_sources", fake_loader)
        monkeypatch.setattr(mod, "DEFAULT_REGISTRY", _TEST_REGISTRY)

        rc = main([])
        captured = capsys.readouterr().out
        assert rc == 0
        assert "No drift detected" in captured

    def test_drift_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from scripts import check_schema_consistency as mod

        def fake_loader(
            registry: tuple[EntityMapping, ...], repo_root: Path
        ) -> tuple[
            dict[str, EntitySpec],
            dict[str, EntitySpec],
            dict[str, EntitySpec],
            list[str],
        ]:
            bad_schema = _snap_json_schema()
            bad_schema["properties"]["rmssd_ms"] = {"type": "integer"}
            pyd, jsn = _build_sources(json_schema=bad_schema)
            return pyd, jsn, {}, []

        monkeypatch.setattr(mod, "load_default_sources", fake_loader)
        monkeypatch.setattr(mod, "DEFAULT_REGISTRY", _TEST_REGISTRY)

        rc = main([])
        captured = capsys.readouterr().out
        assert rc == 1
        assert "type_mismatch" in captured
        assert "rmssd_ms" in captured


# =====================================================================
# Registry sanity (smoke test on the real DEFAULT_REGISTRY)
# =====================================================================


class TestDefaultRegistry:
    def test_default_registry_is_non_empty_and_well_formed(self) -> None:
        assert len(DEFAULT_REGISTRY) > 0
        for mapping in DEFAULT_REGISTRY:
            assert mapping.name
            # Each entity must be defined in both sources to be worth
            # comparing — single-source entities are silently consistent.
            assert mapping.pydantic_class is not None, (
                f"Entity {mapping.name!r} has no Pydantic class — drop it "
                f"from the registry or wire one in."
            )
            assert mapping.json_schema_key is not None, (
                f"Entity {mapping.name!r} has no JSON Schema key — drop it "
                f"from the registry or add the schema block."
            )

    def test_physiological_chunk_event_schema_requires_event_type(self) -> None:
        schema = PhysiologicalChunkEvent.model_json_schema()
        exported_schema = physiological_sample_event_schema

        assert "event_type" in schema["required"]
        assert "event_type" in exported_schema["required"]
        event_entity = parse_json_schema(schema, name="PhysiologicalChunkEvent")
        assert event_entity.fields["event_type"] == FieldSpec(
            "event_type",
            "string",
            False,
        )

    def test_attribution_models_are_loadable_for_schema_consistency(self) -> None:
        attribution_models = (
            AttributionEvent,
            OutcomeEvent,
            EventOutcomeLink,
            AttributionScore,
        )

        for model in attribution_models:
            entity = pydantic_to_entity(model, name=model.__name__)
            assert "schema_version" in entity.fields
            assert "finality" in entity.fields
            assert "created_at" in entity.fields

        event_entity = pydantic_to_entity(AttributionEvent, name="AttributionEvent")
        assert event_entity.fields["event_id"] == FieldSpec("event_id", "uuid", False)
        assert event_entity.fields["bandit_decision_snapshot"] == FieldSpec(
            "bandit_decision_snapshot", "object", False
        )

    def test_attribution_models_are_exported_from_schema_package(self) -> None:
        import packages.schemas as schema_exports

        assert schema_exports.AttributionEvent is AttributionEvent
        assert schema_exports.OutcomeEvent is OutcomeEvent
        assert schema_exports.EventOutcomeLink is EventOutcomeLink
        assert schema_exports.AttributionScore is AttributionScore
