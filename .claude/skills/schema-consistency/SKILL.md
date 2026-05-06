---
name: schema-consistency
description: Run scripts/check_schema_consistency.py and interpret its output whenever schema-affecting work is in flight. Use when the user says "update schema", "add Pydantic model", "change a payload field", "edit packages/schemas", or after any merge that touches one of the two schema sources.
---

# Schema Consistency Gate (`scripts/check_schema_consistency.py`)

## Why this skill exists

The codebase carries three contract-bearing schema sources that can drift independently:

1. **Pydantic models** in `packages/schemas/` — runtime payload validation (`InferenceHandoffPayload`, `PhysiologicalSnapshot`, `SemanticEvaluationResult`, etc.).
2. **JSON Schema blocks** under `interface_contracts` in the signed spec content payload — loaded from the committed `docs/tech-spec-v*.pdf` and optionally inspected via generated `docs/content.json`.
3. **Cloud PostgreSQL tables** from `services.cloud_api.db.schema` — the cloud persistence surface for contract-bearing records.

A field rename in one source without the others causes silent drift. The script normalizes every source onto a common `{name, canonical_type, nullable}` representation and reports any divergence.

Local SQLite state in `services/desktop_app/state/sqlite_schema.py` is intentionally out of scope for direct type parity checks. The desktop mirror collapses several cloud types (for example UUID, TIMESTAMPTZ, and JSONB) to SQLite `TEXT`, so a strict storage-class comparison would produce noise instead of signal.

## When to invoke

Run the script:

- Before committing any change under `packages/schemas/`.
- Before merging a PR that touches Pydantic schemas or the signed spec payload.
- After regenerating ignored `docs/content.json` from the signed PDF for local inspection (`python scripts/spec_ref_check.py --extract > docs/content.json`).
- As Standing Post-Merge Chore #3 in the playbook (Schema-Code Consistency Verification).
- As CI gate step in `scripts/check.sh` (already wired — non-zero exit fails the local check suite).

User-language triggers: "update schema", "add Pydantic model", "schema drift check", "drift in physiology_log", "field rename", "add column".

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

For each finding, decide which source is correct and align the other to match:

| Finding pattern | Most likely root cause | Fix location |
|---|---|---|
| Field present in `pydantic` only | New Pydantic field added without a JSON Schema mirror in the spec | Add the field to the matching block under `interface_contracts.schemas` in the signed spec payload, then regenerate the PDF/payload output |
| Field present in `json_schema` only | Spec defined a field that was never implemented | Either implement the Pydantic field or update the signed spec payload to remove the JSON Schema property |
| `type_mismatch: pydantic=integer, json_schema=number` | Pydantic field is `int` but JSON Schema declares `number` | Decide which is intended; fix in the source that's wrong |
| `type_mismatch` involving `unknown` | The script's type normalizer didn't recognize a Pydantic annotation or an unsupported JSON Schema type/format combo | Either fix the source to use a known type, or extend `_normalize_pydantic_annotation` / `JSON_SCHEMA_TYPE_MAP` if the type is genuinely common |
| `nullability_mismatch: pydantic=required, json_schema=nullable` | JSON Schema's `required` array dropped the field; Pydantic still requires it | Add the field name to the JSON Schema `required` list OR change Pydantic to `field: T \| None = None` — the choice depends on whether `None` is semantically valid |

## Adding a new entity to the registry

When a new payload schema lands, add an `EntityMapping` entry to `DEFAULT_REGISTRY`:

```python
EntityMapping(
    name="MyNewEntity",
    pydantic_class="packages.schemas.my_new:MyNewEntity",
    json_schema_key="MyNewEntity",
)
```

Both source identifiers must be non-`None` — the registry sanity test in `tests/unit/scripts/test_schema_consistency.py::TestDefaultRegistry` enforces this. Single-source entities have nothing to drift against and produce noise rather than signal.

## Hard rules

- A schema-touching PR that fails this gate must NOT be merged. Fix the drift first.
- Do NOT widen a Pydantic field to `Any` to make `unknown` warnings disappear — the spec forbids `Any` outside of explicitly-flexible dicts (per `CLAUDE.md` hard rules). Fix the type instead.
- The script must remain pure-Python with no third-party dependencies beyond what `packages/schemas/` already requires (Pydantic). Adding a parser dependency would make the gate harder to run.
- If another contract-bearing source is added later, extend `DEFAULT_REGISTRY` deliberately and keep the comparison scoped to surfaces where type parity is semantically meaningful.

## Cross-references

- Script: `scripts/check_schema_consistency.py`
- Tests: `tests/unit/scripts/test_schema_consistency.py`
- CI gate: `scripts/check.sh`
- Source-of-truth schemas: `packages/schemas/`, signed spec content payload
- Spec contract: signed spec §6 Interface Contracts via `scripts/spec_ref_check.py --resolve 6`
- Post-merge chore: see `post-merge-playbook` skill, Standing Chore #3
