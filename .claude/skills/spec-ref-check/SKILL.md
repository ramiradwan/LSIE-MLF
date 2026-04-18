---
name: spec-ref-check
description: Validate and resolve spec references (§N.N patterns) against the LSIE-MLF content.json. Use when creating or editing ADO work items, writing code docstrings that reference spec sections, reviewing PRs for spec compliance, or after any content.json or tech-spec PDF update to verify all refs still resolve. Also use when unsure which spec section a ref like "§7A.4" or "§4.A.1" points to. Can extract content.json directly from the authoritative PDF.
---

# Spec Reference Checker (v3 Schema)

Resolves spec_refs (§7A, §7A.4, §4.A.1, §12.3.2, etc.) to actual content.json paths. Can load content.json from the authoritative PDF, a standalone file, or auto-detected in the repo.

## When to Run

Run `--validate` after editing content.json or after any spec-related content change. Run `--resolve` before using a spec_ref in ADO work items, docstrings, or review comments to confirm it points to real content. Run `--extract` to pull content.json out of the PDF when you need the machine-readable payload.

## Commands

```bash
# Validate all refs in the project against the authoritative PDF
python scripts/spec_ref_check.py --from-pdf docs/tech-spec-v3.2.pdf --validate

# Validate against standalone content.json
python scripts/spec_ref_check.py --validate

# Show the full generated index
python scripts/spec_ref_check.py --index

# Resolve a single ref
python scripts/spec_ref_check.py --resolve "7A.4"
python scripts/spec_ref_check.py --resolve "4.A.1"
python scripts/spec_ref_check.py --resolve "12.3.2"

# Extract content.json from the PDF to stdout
python scripts/spec_ref_check.py --from-pdf docs/tech-spec-v3.2.pdf --extract > docs/content.json

# JSON output for programmatic use
python scripts/spec_ref_check.py --validate --json
```

## Ref Conventions (v3)

The v3 content.json uses these ref patterns:

- `§7A`, `§7B` — Math topics (uppercase letter appended to section number, no dot).
- `§7A.1`, `§7B.4` — Math topic subsections (variable dict, derivation steps, ref impl).
- `§4.A.1`, `§4.C.3` — Core module subsections (dot-separated).
- `§9.1`, `§10.2.14` — Conventional subsections and their children.
- `§12.1`, `§12.3.2` — Error handling by module (12.1=Module A, 12.2=Module B, ...).
- `§9`, `§12` — Bare section numbers.

## Content Source Priority

The script loads content.json from the first available source: `--from-pdf` (extracts the embedded PDF/A-3 attachment), then `--content` (explicit file path), then auto-detected at `docs/content.json` or `content.json` in the repo root. The PDF is the authoritative source because it carries the exact content.json that was used to generate the document.

## How the Index Works

Layer 1 (structural) auto-generates refs from content.json structure: section numbers, module IDs (§4.A–§4.F), subsection numbers, math topic letters (§7A, §7B), data flow stages (§2.1–§2.N), error matrix by module (§12.1–§12.6), and conventional field mappings. Layer 2 (explicit) overlays any `spec_refs` arrays in the content.json. Explicit entries take precedence and are marked with `*` in the index output.

## Maintenance

The structural index rebuilds from content.json on every run. Two hardcoded mappings exist at the top of the script: root payload keys to section numbers (14 entries), and math topic_id to letter (au12→A, thompson_sampling→B). Update these when the schema adds sections or math topics.

PyMuPDF (`pip install pymupdf`) is required only for `--from-pdf` and `--extract`. All other functionality has zero dependencies.

## Integration

The `build_index` and `load_content` functions are importable. `automation/platform/spec_grounding.py` in {REDACTED}-platform can use them to replace its hardcoded Rosetta Stone mappings.