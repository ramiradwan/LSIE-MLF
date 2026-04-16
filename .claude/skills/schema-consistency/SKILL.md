---
name: schema-consistency
description: Run scripts/check_schema_consistency.py and interpret its output whenever schema-affecting work is in flight. Use when the user says "update schema", "add Pydantic model", "modify DDL", "change a payload field", "edit data/sql", "edit packages/schemas", "edit services/api/db/schema.py", or after any merge that touches one of the four schema sources.
---

# Schema Consistency Gate (`scripts/check_schema_consistency.py`)

## Why this skill exists

The codebase carries four overlapping schema sources that drift independently:

1. **Pydantic models** in `packages/schemas/` — runtime payload validation (`InferenceHandoffPayload`, `PhysiologicalSnapshot`, `SemanticEvaluationResult`, etc.).
2. **PostgreSQL DDL** in `data/sql/*.sql` — mounted into the Persistent Store via `docker-entrypoint-initdb.d/`.
3. **Python DDL string** `services.api.db.schema.SCHEMA_SQL` — the API Server's bootstrap path.
4. **JSON Schema blocks** under `interface_contracts.schemas` in `docs/artifacts/content.json` — the spec contract surface that the review agent loads.

A field rename in one source without the other three causes silent drift. The script normalizes all four onto a common `{name, canonical_type, nullable}` representation and reports any divergence.

## When to invoke

Run the script:

- Before committing any change under `packages/schemas/`, `data/sql/`, or `services/api/db/schema.py`.
- Before merging a PR that touches any of those paths.
- After regenerating `docs/artifacts/content.json` from the signed PDF (`scripts/spec_ref_check.py --from-pdf docs/tech-spec-v3.1.pdf --extract`).
- As Standing Post-Merge Chore #3 in the playbook (Schema-Code Consistency Verification).
- As CI gate step 7 in `scripts/check.sh` (already wired — non-zero exit fails the local check suite).

User-language triggers: "update schema", "add Pydantic model", "modify DDL", "edit data/sql", "schema drift check", "drift in physiology_log", "field rename", "add column".

## How to run

```bash
python scripts/check_schema_consistency.py
```

Exit code `0` means no drift. Exit code `1` means at least one inconsistency was detected — the report is written to stdout and the process exits non-zero so CI fails.

To inspect just one entity, edit `DEFAULT_REGISTRY` at the top of the script (or call `check_consistency()` directly with a one-element registry).

## Output format

The report has three sections:

1. **Header** — banner.
2. **Warnings** (optional) — sources that could not be loaded. A missing source is reported but does not by itself cause a failure (a source that doesn't exist can't drift). Common warnings:
   - `Could not load pydantic model …: No module named 'packages'` — the script's `sys.path` shim failed; run from the repo root.
   - `content.json is empty: …` — the spec content payload has not been extracted from the PDF.
3. **Inconsistencies** — grouped by entity, one line per finding:
   ```
   [X] <field>  <kind>  <detail>
   ```
   where `<kind>` is one of:
   - `missing` — field present in some sources but absent from at least one. Detail lists which sources have it and which don't.
   - `type_mismatch` — canonical types disagree across sources. Detail lists `source=type` for each source.
   - `nullability_mismatch` — required-vs-nullable disagrees. Detail lists `source=nullable|required` for each source.

## Interpreting findings

For each finding, decide which source is correct and align the other three to match:

| Finding pattern | Most likely root cause | Fix location |
|---|---|---|
| Field present in `pydantic` only | New Pydantic field added without DDL or JSON Schema mirror | Add column to `data/sql/*.sql` AND `services/api/db/schema.py` AND `interface_contracts.schemas` block |
| Field present in `sql_file` and `sql_string` only | DB column added without payload schema | Add field to the corresponding Pydantic model AND the JSON Schema block |
| Field present in `sql_file` only (not `sql_string`) | DDL files diverged — `data/sql/` updated, Python string forgot | Mirror the change into `services.api.db.schema.SCHEMA_SQL` |
| Field present in `json_schema` only | Spec defined a field that was never implemented | Either implement (Pydantic + SQL × 2) or remove from spec via amendment |
| `type_mismatch: pydantic=integer, sql_file=number` | `int` vs `DOUBLE PRECISION` — Pydantic field is `int` but DDL is float | Decide which is intended; usually the SQL side is wrong (downcast risk) |
| `type_mismatch` involving `unknown` | The script's type normalizer didn't recognize a SQL type or Pydantic annotation | Either fix the source to use a known type, or extend `SQL_TYPE_MAP` / `_normalize_pydantic_annotation` if the type is genuinely common |
| `nullability_mismatch: pydantic=required, sql_file=nullable` | DDL allows NULL but Pydantic requires the field | Add `NOT NULL` to the column OR change Pydantic to `field: T \| None = None` — the choice depends on whether `None` is semantically valid |

## Auto-managed columns are excluded

The registry in `scripts/check_schema_consistency.py` ignores `id`, `session_id`, `segment_id`, `subject_role`, and `created_at` for SQL-backed entities. These are surrogate keys, FK routing columns the API code injects on insert, or `DEFAULT NOW()` audit timestamps. They are not part of the cross-source payload contract and would otherwise produce noise.

If you add a new auto-managed column (e.g., another `*_at` timestamp with `DEFAULT NOW()`), add it to `_AUTO_MANAGED_COLUMNS` in the script.

## Adding a new entity to the registry

When a new persistence table or new payload schema lands, add a `EntityMapping` entry to `DEFAULT_REGISTRY`:

```python
EntityMapping(
    name="MyNewEntity",
    pydantic_class="packages.schemas.my_new:MyNewEntity",  # or None
    sql_table="my_new_table",                               # or None
    json_schema_key="MyNewEntity",                          # or None
    ignore_fields=_AUTO_MANAGED_COLUMNS,
)
```

At least two of the three source identifiers must be non-`None` — otherwise there is nothing to compare and the entity should not be in the registry. The registry sanity test in `tests/unit/scripts/test_schema_consistency.py::TestDefaultRegistry` enforces this.

## Hard rules

- A schema-touching PR that fails this gate must NOT be merged. Fix the drift first.
- Do NOT silence a drift finding by adding the field to `_AUTO_MANAGED_COLUMNS` unless it is genuinely auto-managed by the database.
- Do NOT widen a Pydantic field to `Any` to make `unknown` warnings disappear — the spec forbids `Any` outside of explicitly-flexible dicts (per `CLAUDE.md` hard rules). Fix the type instead.
- The script must remain pure-Python with no third-party dependencies beyond what `packages/schemas/` already requires (Pydantic). Adding a SQL parser dependency would make the gate harder to run.

## Cross-references

- Script: `scripts/check_schema_consistency.py`
- Tests: `tests/unit/scripts/test_schema_consistency.py`
- CI gate: `scripts/check.sh` step 7
- Source-of-truth schemas: `packages/schemas/`, `data/sql/`, `services/api/db/schema.py`, `docs/artifacts/content.json`
- Spec contract: `docs/SPEC_REFERENCE.md` §6 (Interface Contracts)
- Post-merge chore: see `post-merge-playbook` skill, Standing Chore #3
