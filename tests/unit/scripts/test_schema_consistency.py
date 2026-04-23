"""Tests for ``scripts.check_schema_consistency``.

These tests seed the four schema sources with synthetic known-good and
known-bad combinations to verify the detector catches the specific drift
cases it is designed to catch:

  * Field present in one source but missing from another
  * Field whose canonical type disagrees across sources
  * Field whose nullability disagrees across sources
  * Auto-managed columns (``id``, ``created_at``, etc.) are correctly ignored
  * Single-source entities are not falsely flagged
  * The all-aligned case yields zero issues and exit code 0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from scripts.check_schema_consistency import (
    DEFAULT_REGISTRY,
    EntityMapping,
    EntitySpec,
    FieldSpec,
    Inconsistency,
    _normalize_pydantic_annotation,
    _normalize_sql_type,
    check_consistency,
    compare_entity,
    extract_json_schemas,
    format_report,
    main,
    parse_json_schema,
    parse_sql_tables,
    pydantic_to_entity,
)

# =====================================================================
# SQL parser
# =====================================================================


class TestSqlParser:
    def test_parses_single_table_with_mixed_nullability(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS physiology_log (
            id              BIGSERIAL PRIMARY KEY,
            session_id      UUID NOT NULL REFERENCES sessions(session_id),
            rmssd_ms        DOUBLE PRECISION,
            freshness_s     DOUBLE PRECISION NOT NULL,
            is_stale        BOOLEAN NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        tables = parse_sql_tables(sql)
        assert "physiology_log" in tables
        cols = tables["physiology_log"].fields

        assert cols["id"] == FieldSpec("id", "integer", nullable=False)
        assert cols["session_id"] == FieldSpec("session_id", "uuid", nullable=False)
        assert cols["rmssd_ms"] == FieldSpec("rmssd_ms", "number", nullable=True)
        assert cols["freshness_s"] == FieldSpec("freshness_s", "number", nullable=False)
        assert cols["is_stale"] == FieldSpec("is_stale", "boolean", nullable=False)
        assert cols["created_at"] == FieldSpec("created_at", "datetime", nullable=False)

    def test_skips_check_constraints(self) -> None:
        sql = """
        CREATE TABLE x (
            subject_role TEXT NOT NULL CHECK (subject_role IN ('a', 'b')),
            count INTEGER
        );
        """
        cols = parse_sql_tables(sql)["x"].fields
        assert "subject_role" in cols
        assert cols["subject_role"].type == "string"
        assert cols["subject_role"].nullable is False
        assert "count" in cols and cols["count"].nullable is True

    def test_parses_multiple_tables(self) -> None:
        sql = """
        CREATE TABLE a (id BIGSERIAL PRIMARY KEY, x TEXT);
        CREATE TABLE b (id BIGSERIAL PRIMARY KEY, y INTEGER NOT NULL);
        """
        tables = parse_sql_tables(sql)
        assert set(tables) == {"a", "b"}
        assert tables["a"].fields["x"].type == "string"
        assert tables["b"].fields["y"].type == "integer"

    def test_strips_line_and_block_comments(self) -> None:
        sql = """
        -- top-level comment
        CREATE TABLE z (
            /* inline block */
            value DOUBLE PRECISION NOT NULL  -- trailing
        );
        """
        cols = parse_sql_tables(sql)["z"].fields
        assert cols == {"value": FieldSpec("value", "number", nullable=False)}

    def test_normalize_strips_modifiers(self) -> None:
        assert _normalize_sql_type("VARCHAR(255)") == "string"
        assert _normalize_sql_type("NUMERIC(10, 2)") == "number"
        assert _normalize_sql_type("TIMESTAMP WITH TIME ZONE") == "datetime"

    def test_unknown_sql_type_is_unknown(self) -> None:
        assert _normalize_sql_type("CIDR") == "unknown"


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

    def test_extract_schemas_handles_v3_and_v31_layouts(self) -> None:
        v31 = {"interface_contracts": {"schemas": {"Foo": {"type": "object"}}}}
        v3 = {
            "interface_contracts": {"schema_definition": {"schemas": {"Bar": {"type": "object"}}}}
        }
        assert extract_json_schemas(v31) == {"Foo": {"type": "object"}}
        assert extract_json_schemas(v3) == {"Bar": {"type": "object"}}
        assert extract_json_schemas({}) == {}
        assert extract_json_schemas({"interface_contracts": "not-a-dict"}) == {}


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
            "sql_file": spec,
            "sql_string": spec,
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
            {
                "pydantic": full,
                "sql_file": full,
                "sql_string": full,
                "json_schema": partial,
            },
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
            {
                "pydantic": as_int,
                "sql_file": as_int,
                "sql_string": as_int,
                "json_schema": as_num,
            },
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
            {
                "pydantic": nullable,
                "sql_file": required,
                "sql_string": required,
                "json_schema": nullable,
            },
            frozenset(),
        )
        kinds = [i.kind for i in issues]
        assert "nullability_mismatch" in kinds
        null_issue = next(i for i in issues if i.kind == "nullability_mismatch")
        assert "pydantic=nullable" in null_issue.detail
        assert "sql_file=required" in null_issue.detail

    def test_missing_field_does_not_also_flag_type_or_nullability(self) -> None:
        full = _spec(("x", "integer", False))
        empty = EntitySpec(name="x")
        issues = compare_entity(
            "E",
            {"pydantic": full, "sql_file": empty, "sql_string": full, "json_schema": full},
            frozenset(),
        )
        # Exactly one finding (the missing field), not three.
        assert len(issues) == 1
        assert issues[0].kind == "missing"

    def test_ignore_fields_are_skipped(self) -> None:
        sql_only = _spec(
            ("id", "integer", False),
            ("created_at", "datetime", False),
            ("rmssd_ms", "number", True),
        )
        py_only = _spec(("rmssd_ms", "number", True))
        issues = compare_entity(
            "E",
            {
                "pydantic": py_only,
                "sql_file": sql_only,
                "sql_string": sql_only,
                "json_schema": py_only,
            },
            frozenset({"id", "created_at"}),
        )
        assert issues == []

    def test_single_source_entity_not_flagged(self) -> None:
        """An entity defined in only one source has nothing to drift against."""
        only_pydantic = _spec(("x", "integer", False))
        issues = compare_entity(
            "E",
            {
                "pydantic": only_pydantic,
                "sql_file": None,
                "sql_string": None,
                "json_schema": None,
            },
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


def _snap_sql() -> str:
    # NOTE: id/session_id/segment_id/subject_role/created_at are auto-managed
    # and ignored by the registry — they intentionally have no Pydantic mirror.
    return """
    CREATE TABLE physiology_log (
        id BIGSERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        segment_id TEXT NOT NULL,
        subject_role TEXT NOT NULL,
        rmssd_ms DOUBLE PRECISION,
        freshness_s DOUBLE PRECISION NOT NULL,
        is_stale BOOLEAN NOT NULL,
        provider TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """


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
        sql_table="physiology_log",
        json_schema_key="PhysiologicalSnapshot",
        ignore_fields=frozenset({"id", "session_id", "segment_id", "subject_role", "created_at"}),
    ),
)


def _build_sources(
    pydantic_cls: type[BaseModel] = _SnapPydantic,
    sql_text: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> tuple[
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    dict[str, EntitySpec],
    dict[str, EntitySpec],
]:
    pyd = {"PhysiologicalSnapshot": pydantic_to_entity(pydantic_cls, "PhysiologicalSnapshot")}
    sql_tables = parse_sql_tables(sql_text or _snap_sql())
    schema_block = json_schema if json_schema is not None else _snap_json_schema()
    json_entities = {
        "PhysiologicalSnapshot": parse_json_schema(schema_block, "PhysiologicalSnapshot"),
    }
    return pyd, sql_tables, dict(sql_tables), json_entities


class TestCheckConsistencyEndToEnd:
    def test_known_good_aligned_sources_yield_zero_issues(self) -> None:
        pyd, sql_file, sql_str, jsn = _build_sources()
        assert check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn) == []

    def test_pydantic_drift_field_added_is_detected(self) -> None:
        class Drifted(BaseModel):
            rmssd_ms: float | None = None
            freshness_s: float
            is_stale: bool
            provider: str
            extra_field: int  # added in Pydantic but absent from SQL & JSON Schema

        pyd, sql_file, sql_str, jsn = _build_sources(pydantic_cls=Drifted)
        issues = check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn)

        missing = [i for i in issues if i.field == "extra_field"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "pydantic" in missing[0].detail
        assert "sql_file" in missing[0].detail
        assert "json_schema" in missing[0].detail

    def test_sql_file_drift_type_change_is_detected(self) -> None:
        # rmssd_ms changed from DOUBLE PRECISION to INTEGER on disk — disagrees
        # with the Pydantic float and the JSON Schema number.
        bad_sql = _snap_sql().replace("rmssd_ms DOUBLE PRECISION", "rmssd_ms INTEGER")
        pyd, sql_file, sql_str, jsn = _build_sources(sql_text=bad_sql)
        # sql_string (Python DDL) keeps the correct DOUBLE PRECISION
        sql_str = parse_sql_tables(_snap_sql())

        issues = check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn)
        types = [i for i in issues if i.kind == "type_mismatch" and i.field == "rmssd_ms"]
        assert len(types) == 1
        assert "sql_file=integer" in types[0].detail
        assert "sql_string=number" in types[0].detail

    def test_python_ddl_string_drift_is_detected(self) -> None:
        # Python DDL string has freshness_s as nullable; SQL file & Pydantic
        # & JSON Schema all have it required.
        bad_py_ddl = _snap_sql().replace(
            "freshness_s DOUBLE PRECISION NOT NULL",
            "freshness_s DOUBLE PRECISION",
        )
        pyd, sql_file, _sql_str, jsn = _build_sources()
        sql_str = parse_sql_tables(bad_py_ddl)

        issues = check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn)
        nulls = [i for i in issues if i.kind == "nullability_mismatch" and i.field == "freshness_s"]
        assert len(nulls) == 1
        assert "sql_string=nullable" in nulls[0].detail
        assert "sql_file=required" in nulls[0].detail
        assert "pydantic=required" in nulls[0].detail

    def test_json_schema_drift_required_to_optional_is_detected(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["required"] = ["is_stale", "provider"]  # drop freshness_s

        pyd, sql_file, sql_str, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn)
        nulls = [i for i in issues if i.kind == "nullability_mismatch" and i.field == "freshness_s"]
        assert len(nulls) == 1
        assert "json_schema=nullable" in nulls[0].detail

    def test_json_schema_field_missing_from_pydantic(self) -> None:
        bad_schema = _snap_json_schema()
        bad_schema["properties"]["coverage_ratio"] = {"type": "number"}
        bad_schema["required"].append("coverage_ratio")

        pyd, sql_file, sql_str, jsn = _build_sources(json_schema=bad_schema)
        issues = check_consistency(_TEST_REGISTRY, pyd, sql_file, sql_str, jsn)
        missing = [i for i in issues if i.field == "coverage_ratio"]
        assert len(missing) == 1
        assert missing[0].kind == "missing"
        assert "json_schema" in missing[0].detail
        assert "pydantic" in missing[0].detail

    def test_repo_sql_surfaces_include_enriched_physiology_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        sql_file_tables = parse_sql_tables(
            (repo_root / "data" / "sql" / "03-physiology.sql").read_text(encoding="utf-8")
        )

        from services.api.db.schema import SCHEMA_SQL

        sql_string_tables = parse_sql_tables(SCHEMA_SQL)

        expected_fields = {
            "source_kind": FieldSpec("source_kind", "string", nullable=True),
            "derivation_method": FieldSpec("derivation_method", "string", nullable=True),
            "window_s": FieldSpec("window_s", "integer", nullable=True),
            "validity_ratio": FieldSpec("validity_ratio", "number", nullable=True),
            "is_valid": FieldSpec("is_valid", "boolean", nullable=True),
        }

        for field_name, expected in expected_fields.items():
            assert sql_file_tables["physiology_log"].fields[field_name] == expected
            assert sql_string_tables["physiology_log"].fields[field_name] == expected

        sql_file_entity = EntitySpec(
            name="physiology_log",
            fields={
                field_name: sql_file_tables["physiology_log"].fields[field_name]
                for field_name in expected_fields
            },
        )
        sql_string_entity = EntitySpec(
            name="physiology_log",
            fields={
                field_name: sql_string_tables["physiology_log"].fields[field_name]
                for field_name in expected_fields
            },
        )

        assert (
            compare_entity(
                "PhysiologicalSnapshot",
                {
                    "pydantic": None,
                    "sql_file": sql_file_entity,
                    "sql_string": sql_string_entity,
                    "json_schema": None,
                },
                _TEST_REGISTRY[0].ignore_fields,
            )
            == []
        )

    def test_repo_schema_sql_contains_additive_physiology_migration_statements(self) -> None:
        from services.api.db.schema import SCHEMA_SQL

        expected_fragments = (
            "ALTER TABLE physiology_log\n    ADD COLUMN IF NOT EXISTS source_kind TEXT",
            "ALTER TABLE physiology_log\n    ADD COLUMN IF NOT EXISTS derivation_method TEXT",
            "ALTER TABLE physiology_log\n    ADD COLUMN IF NOT EXISTS window_s INTEGER",
            (
                "ALTER TABLE physiology_log\n"
                "    ADD COLUMN IF NOT EXISTS validity_ratio DOUBLE PRECISION"
            ),
            "ALTER TABLE physiology_log\n    ADD COLUMN IF NOT EXISTS is_valid BOOLEAN",
        )

        for fragment in expected_fragments:
            assert fragment in SCHEMA_SQL


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
            Inconsistency("B", "y", "type_mismatch", "pydantic=integer, sql=number"),
            Inconsistency("A", "z", "type_mismatch", "pydantic=integer, sql=string"),
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
            dict[str, EntitySpec],
            list[str],
        ]:
            pyd, sql_file, sql_str, jsn = _build_sources()
            return pyd, sql_file, sql_str, jsn, []

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
            dict[str, EntitySpec],
            list[str],
        ]:
            bad_sql = _snap_sql().replace("rmssd_ms DOUBLE PRECISION", "rmssd_ms INTEGER")
            pyd, sql_file, _, jsn = _build_sources(sql_text=bad_sql)
            sql_str = parse_sql_tables(_snap_sql())
            return pyd, sql_file, sql_str, jsn, []

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
            # Each entity must be defined in at least two sources to be worth
            # comparing. Otherwise it should not be in the registry.
            sources_present = sum(
                1
                for v in (
                    mapping.pydantic_class,
                    mapping.sql_table,
                    mapping.json_schema_key,
                )
                if v is not None
            )
            assert sources_present >= 2, (
                f"Entity {mapping.name!r} has only one source — drop it "
                f"from the registry or add a second source."
            )
