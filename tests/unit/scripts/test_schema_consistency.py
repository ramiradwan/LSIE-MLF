"""Tests for ``scripts.check_schema_consistency``.

These tests seed the schema sources (Pydantic models, JSON Schema
blocks from the extracted spec content, and optional cloud SQL tables)
with synthetic known-good and known-bad combinations to verify the
checker catches the specific drift cases it is designed to catch:

  * configured entities disappearing from a source
  * fields present in one source but missing from another
  * fields whose canonical type disagrees across sources
  * fields whose required-vs-optional semantics disagree across JSON payload sources
  * fields whose explicit-nullability semantics disagree across sources
  * the all-aligned case yields zero issues and exit code 0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field

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
    check_consistency,
    compare_entity,
    extract_json_schemas,
    format_report,
    load_default_sources,
    main,
    parse_json_schema,
    parse_postgres_sql_tables,
    pydantic_to_entity,
)


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
    active_arm: str = Field(..., alias="_active_arm")

    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class TestPydanticLoader:
    def test_round_trip_through_pydantic_to_entity(self) -> None:
        entity = pydantic_to_entity(_Sample, name="Sample")

        assert entity.fields["rmssd_ms"] == FieldSpec("rmssd_ms", "number", True, False)
        assert entity.fields["freshness_s"] == FieldSpec("freshness_s", "number", False, True)
        assert entity.fields["is_stale"] == FieldSpec("is_stale", "boolean", False, True)
        assert entity.fields["provider"] == FieldSpec("provider", "string", False, False)
        assert entity.fields["when"] == FieldSpec("when", "datetime", False, True)
        assert entity.fields["sub_id"] == FieldSpec("sub_id", "uuid", False, True)
        assert entity.fields["tags"] == FieldSpec("tags", "array", False, False)
        assert entity.fields["child"] == FieldSpec("child", "object", True, False)
        assert entity.fields["_active_arm"] == FieldSpec("_active_arm", "string", False, True)


class TestJsonSchemaLoader:
    def test_required_optional_and_nullable_are_distinct(self) -> None:
        schema = {
            "type": "object",
            "required": ["freshness_s", "is_stale"],
            "properties": {
                "rmssd_ms": {"type": ["number", "null"]},
                "freshness_s": {"type": "number"},
                "is_stale": {"type": "boolean"},
                "when": {"type": "string", "format": "date-time"},
                "sub_id": {
                    "type": "string",
                    "pattern": (
                        "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
                    ),
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        entity = parse_json_schema(schema, name="Sample")

        assert entity.fields["rmssd_ms"] == FieldSpec("rmssd_ms", "number", True, False)
        assert entity.fields["freshness_s"] == FieldSpec("freshness_s", "number", False, True)
        assert entity.fields["is_stale"] == FieldSpec("is_stale", "boolean", False, True)
        assert entity.fields["when"] == FieldSpec("when", "datetime", False, False)
        assert entity.fields["sub_id"] == FieldSpec("sub_id", "uuid", False, False)
        assert entity.fields["tags"] == FieldSpec("tags", "array", False, False)

    def test_parse_json_schema_resolves_refs_and_nullable_unions(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "oneOf": [
                        {"type": "null"},
                        {"$ref": "#/$defs/Child"},
                    ]
                }
            },
            "$defs": {
                "Child": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                }
            },
        }

        entity = parse_json_schema(schema, name="Sample")

        assert entity.fields["child"] == FieldSpec("child", "object", True, False)

    def test_extract_schemas_handles_supported_layouts_and_nested_defs(self) -> None:
        direct = {
            "interface_contracts": {
                "schemas": {
                    "Foo": {
                        "title": "Foo",
                        "type": "object",
                        "$defs": {
                            "Bar": {
                                "title": "Bar",
                                "type": "object",
                            }
                        },
                    }
                }
            }
        }
        nested = {
            "interface_contracts": {"schema_definition": {"schemas": {"Baz": {"type": "object"}}}}
        }
        extracted_block = {
            "interface_contracts": {
                "posterior_delta_schema": {
                    "language": "json",
                    "source": '{"title":"PosteriorDelta","type":"object"}',
                }
            }
        }
        assert set(extract_json_schemas(direct)) >= {"Foo", "Bar"}
        assert extract_json_schemas(nested)["Baz"] == {"type": "object"}
        assert extract_json_schemas(extracted_block) == {
            "PosteriorDelta": {"title": "PosteriorDelta", "type": "object"}
        }
        assert extract_json_schemas({}) == {}
        assert extract_json_schemas({"interface_contracts": "not-a-dict"}) == {}

    def test_load_default_sources_prefers_active_content_index(self, tmp_path: Path) -> None:
        import json

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

    def test_parse_comment_preamble_and_array_columns(self) -> None:
        sql = """
        -- Bootstrap heading (v4.0)
        CREATE TABLE IF NOT EXISTS attribution_event (
            evidence_flags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """

        tables = parse_postgres_sql_tables(sql)
        entity = tables["attribution_event"]

        assert "create" not in entity.fields
        assert entity.fields["evidence_flags"] == FieldSpec("evidence_flags", "array", False)
        assert entity.fields["created_at"] == FieldSpec("created_at", "datetime", False)

    def test_sql_source_participates_when_mapping_names_table(self) -> None:
        pyd = {
            "PosteriorDelta": _spec(
                ("event_id", "uuid", False, True),
                ("delta_alpha", "number", False, True),
            )
        }
        jsn = {
            "PosteriorDelta": _spec(
                ("event_id", "uuid", False, True),
                ("delta_alpha", "number", False, True),
            )
        }
        sql = {
            "posterior_delta_log": _spec(
                ("event_id", "uuid", False, None),
                ("delta_alpha", "integer", False, None),
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


def _spec(*pairs: tuple[str, str, bool, bool | None]) -> EntitySpec:
    entity = EntitySpec(name="x")
    for name, typ, nullable, required in pairs:
        entity.fields[name] = FieldSpec(name, typ, nullable, required)
    return entity


class TestCompareEntity:
    def test_all_aligned_yields_no_issues(self) -> None:
        spec = _spec(
            ("rmssd_ms", "number", True, False),
            ("freshness_s", "number", False, True),
        )
        sources = {
            "pydantic": spec,
            "json_schema": spec,
        }
        assert compare_entity("E", sources, frozenset()) == []

    def test_field_missing_in_one_source(self) -> None:
        full = _spec(
            ("rmssd_ms", "number", True, False),
            ("provider", "string", False, True),
        )
        partial = _spec(("rmssd_ms", "number", True, False))
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
        as_int = _spec(("heart_rate_bpm", "integer", False, True))
        as_num = _spec(("heart_rate_bpm", "number", False, True))
        issues = compare_entity(
            "E",
            {"pydantic": as_int, "json_schema": as_num},
            frozenset(),
        )
        kinds = [issue.kind for issue in issues]
        assert "type_mismatch" in kinds
        type_issue = next(issue for issue in issues if issue.kind == "type_mismatch")
        assert "json_schema=number" in type_issue.detail
        assert "pydantic=integer" in type_issue.detail

    def test_nullability_mismatch_is_caught(self) -> None:
        nullable = _spec(("rmssd_ms", "number", True, True))
        non_null = _spec(("rmssd_ms", "number", False, True))
        issues = compare_entity(
            "E",
            {"pydantic": nullable, "json_schema": non_null},
            frozenset(),
        )
        kinds = [issue.kind for issue in issues]
        assert "nullability_mismatch" in kinds
        null_issue = next(issue for issue in issues if issue.kind == "nullability_mismatch")
        assert "pydantic=nullable" in null_issue.detail
        assert "json_schema=non-null" in null_issue.detail

    def test_requiredness_mismatch_is_caught(self) -> None:
        optional = _spec(("evidence_flags", "array", False, False))
        required = _spec(("evidence_flags", "array", False, True))
        issues = compare_entity(
            "E",
            {"pydantic": optional, "json_schema": required},
            frozenset(),
        )
        required_issue = next(issue for issue in issues if issue.kind == "requiredness_mismatch")
        assert "pydantic=optional" in required_issue.detail
        assert "json_schema=required" in required_issue.detail

    def test_missing_field_does_not_also_flag_type_or_nullability(self) -> None:
        full = _spec(("x", "integer", False, True))
        empty = EntitySpec(name="x")
        issues = compare_entity(
            "E",
            {"pydantic": full, "json_schema": empty},
            frozenset(),
        )
        assert len(issues) == 1
        assert issues[0].kind == "missing"

    def test_ignore_fields_are_skipped(self) -> None:
        with_extra = _spec(
            ("internal_marker", "integer", False, True),
            ("rmssd_ms", "number", True, False),
        )
        without_extra = _spec(("rmssd_ms", "number", True, False))
        issues = compare_entity(
            "E",
            {"pydantic": with_extra, "json_schema": without_extra},
            frozenset({"internal_marker"}),
        )
        assert issues == []

    def test_single_source_entity_not_flagged(self) -> None:
        only_pydantic = _spec(("x", "integer", False, True))
        issues = compare_entity(
            "E",
            {"pydantic": only_pydantic, "json_schema": None},
            frozenset(),
        )
        assert issues == []


class _SnapPydantic(BaseModel):
    rmssd_ms: float | None = None
    freshness_s: float
    is_stale: bool
    provider: str
    evidence_flags: list[str] = Field(default_factory=list)


def _snap_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["freshness_s", "is_stale", "provider"],
        "properties": {
            "rmssd_ms": {"type": ["number", "null"]},
            "freshness_s": {"type": "number"},
            "is_stale": {"type": "boolean"},
            "provider": {"type": "string"},
            "evidence_flags": {"type": "array", "items": {"type": "string"}},
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
    pydantic_entities = {
        "PhysiologicalSnapshot": pydantic_to_entity(pydantic_cls, "PhysiologicalSnapshot")
    }
    schema_block = json_schema if json_schema is not None else _snap_json_schema()
    json_schema_entities = {
        "PhysiologicalSnapshot": parse_json_schema(schema_block, "PhysiologicalSnapshot")
    }
    return pydantic_entities, json_schema_entities


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
            evidence_flags: list[str] = Field(default_factory=list)
            extra_field: int

        pyd, jsn = _build_sources(pydantic_cls=Drifted)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)

        missing = [issue for issue in issues if issue.field == "extra_field"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "pydantic" in missing[0].detail
        assert "json_schema" in missing[0].detail

    def test_json_schema_drift_required_to_optional_is_detected(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["required"] = ["is_stale", "provider"]

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        requiredness = [
            issue
            for issue in issues
            if issue.kind == "requiredness_mismatch" and issue.field == "freshness_s"
        ]
        assert len(requiredness) == 1
        assert "json_schema=optional" in requiredness[0].detail
        assert "pydantic=required" in requiredness[0].detail

    def test_json_schema_field_missing_from_pydantic(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["properties"]["coverage_ratio"] = {"type": "number"}
        bad_schema["required"].append("coverage_ratio")

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        missing = [issue for issue in issues if issue.field == "coverage_ratio"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "json_schema" in missing[0].detail
        assert "pydantic" in missing[0].detail

    def test_json_schema_drift_type_change_is_detected(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["properties"]["rmssd_ms"] = {"type": ["integer", "null"]}

        pyd, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, jsn)
        types = [
            issue for issue in issues if issue.kind == "type_mismatch" and issue.field == "rmssd_ms"
        ]
        assert len(types) == 1
        assert "json_schema=integer" in types[0].detail
        assert "pydantic=number" in types[0].detail

    def test_configured_missing_source_is_detected(self) -> None:
        pyd, jsn = _build_sources()
        registry = (
            EntityMapping(
                name="MissingSchema",
                pydantic_class=None,
                json_schema_key="MissingSchema",
            ),
        )

        issues = check_consistency(registry, pyd, jsn)

        assert issues == [
            Inconsistency(
                entity="MissingSchema",
                field="(entity)",
                kind="missing_source",
                detail="configured source missing: ['json_schema']",
            )
        ]


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
            bad_schema["properties"]["rmssd_ms"] = {"type": ["integer", "null"]}
            pyd, jsn = _build_sources(json_schema=bad_schema)
            return pyd, jsn, {}, []

        monkeypatch.setattr(mod, "load_default_sources", fake_loader)
        monkeypatch.setattr(mod, "DEFAULT_REGISTRY", _TEST_REGISTRY)

        rc = main([])
        captured = capsys.readouterr().out
        assert rc == 1
        assert "type_mismatch" in captured
        assert "rmssd_ms" in captured


class TestDefaultRegistry:
    def test_default_registry_is_non_empty_and_well_formed(self) -> None:
        assert len(DEFAULT_REGISTRY) > 0
        for mapping in DEFAULT_REGISTRY:
            assert mapping.name
            assert mapping.pydantic_class is not None, (
                f"Entity {mapping.name!r} has no Pydantic class — drop it "
                f"from the registry or wire one in."
            )
            assert mapping.json_schema_key is not None, (
                f"Entity {mapping.name!r} has no JSON Schema key — drop it "
                f"from the registry or add the schema block."
            )

    def test_default_registry_resolves_against_repo_sources(self) -> None:
        pydantic_entities, json_schema_entities, sql_entities, _warnings = load_default_sources(
            DEFAULT_REGISTRY
        )

        missing_pydantic = [
            mapping.name for mapping in DEFAULT_REGISTRY if mapping.name not in pydantic_entities
        ]
        missing_json_schema = [
            mapping.name for mapping in DEFAULT_REGISTRY if mapping.name not in json_schema_entities
        ]
        missing_sql = [
            mapping.sql_table
            for mapping in DEFAULT_REGISTRY
            if mapping.sql_table is not None and mapping.sql_table not in sql_entities
        ]

        assert missing_pydantic == []
        assert missing_json_schema == []
        assert missing_sql == []

    def test_default_registry_sources_are_currently_consistent(self) -> None:
        pydantic_entities, json_schema_entities, sql_entities, _warnings = load_default_sources(
            DEFAULT_REGISTRY
        )

        issues = check_consistency(
            DEFAULT_REGISTRY,
            pydantic_entities,
            json_schema_entities,
            sql_entities,
        )

        assert issues == []

    def test_physiological_chunk_event_schema_requires_event_type(self) -> None:
        schema = PhysiologicalChunkEvent.model_json_schema()
        exported_schema = physiological_sample_event_schema

        assert "event_type" in schema["required"]
        assert "event_type" in exported_schema["required"]
        event_entity = parse_json_schema(schema, name="PhysiologicalChunkEvent")
        assert event_entity.fields["event_type"] == FieldSpec("event_type", "string", False, True)

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
        assert event_entity.fields["event_id"] == FieldSpec("event_id", "uuid", False, True)
        assert event_entity.fields["bandit_decision_snapshot"] == FieldSpec(
            "bandit_decision_snapshot", "object", False, True
        )

    def test_attribution_models_are_exported_from_schema_package(self) -> None:
        import packages.schemas as schema_exports

        assert schema_exports.AttributionEvent is AttributionEvent
        assert schema_exports.OutcomeEvent is OutcomeEvent
        assert schema_exports.EventOutcomeLink is EventOutcomeLink
        assert schema_exports.AttributionScore is AttributionScore
