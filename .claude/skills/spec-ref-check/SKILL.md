---
name: spec-ref-check
description: Validate and resolve spec references (§N.N patterns) against the embedded LSIE-MLF spec content payload. Use when creating or editing ADO work items, writing code docstrings that reference spec sections, reviewing PRs for spec compliance, or after any signed spec PDF/content-payload update to verify all refs still resolve. Also use when unsure which spec section a ref like "§7A.4" or "§4.A.1" points to. Can extract the embedded payload from the authoritative PDF.
---

# Spec Reference Checker

Resolves spec_refs (§7A, §7A.4, §4.A.1, §12.3.2, etc.) to actual embedded spec-content paths. By default, it loads the content payload from the single `docs/tech-spec-v*.pdf` file committed in the repo; it can also load an explicit PDF or standalone generated content file for diagnostics.

## When to Run

Run `--validate` after any spec-related content change. Run `--resolve` before using a spec_ref in ADO work items, docstrings, or review comments to confirm it points to real content. Run `--extract` when you need to inspect or regenerate the machine-readable payload produced by the signed PDF workflow.

## Commands

```bash
# Validate all refs in the project against the committed authoritative PDF
python scripts/spec_ref_check.py --validate

# Validate against an explicit standalone generated payload
python scripts/spec_ref_check.py --content docs/content.json --validate

# Show the full generated index
python scripts/spec_ref_check.py --index

# Resolve a single ref
python scripts/spec_ref_check.py --resolve "7A.4"
python scripts/spec_ref_check.py --resolve "4.A.1"
python scripts/spec_ref_check.py --resolve "12.3.2"

# Extract the embedded content payload from the committed PDF to stdout
python scripts/spec_ref_check.py --extract

# Extract from an explicit PDF path when working offline
python scripts/spec_ref_check.py --from-pdf <path-to-spec-pdf> --extract

# JSON output for programmatic use
python scripts/spec_ref_check.py --validate --json
```

## Ref Conventions

The current embedded content payload uses these ref patterns:

- `§7A`, `§7B` — Math topics (uppercase letter appended to section number, no dot).
- `§7A.1`, `§7B.4` — Math topic subsections (variable dict, derivation steps, ref impl).
- `§4.A.1`, `§4.C.3` — Core module subsections (dot-separated).
- `§9.1`, `§10.2.14` — Conventional subsections and their children.
- `§12.1`, `§12.3.2` — Legacy error-handling aliases retained for historical module-era references.
- `§9`, `§12` — Bare section numbers.

## Content Source Priority

The script loads spec content from the first available source: `--from-pdf` (explicit PDF extraction), then `--content` (explicit generated file path), then the single `docs/tech-spec-v*.pdf` match in the repo root. The default path intentionally asserts exactly one committed PDF so future spec rolls are a PDF replacement plus optional payload extraction for inspection, not a documentation sweep.

## How the Index Works

Layer 1 (structural) auto-generates refs from the embedded content structure: section numbers, module IDs (§4.A–§4.F), subsection numbers, math topic letters (§7A, §7B), data flow stages (§2.1–§2.N), runtime-topology subsections, and conventional field mappings. Layer 2 (explicit) overlays any `spec_refs` arrays in the payload. The checker also adds a small legacy-alias layer for historical refs that still appear in active docs and tests. Explicit entries take precedence and are marked with `*` in the index output.

## Maintenance

The structural index rebuilds from the loaded payload on every run. Two hardcoded mappings exist at the top of the script: root payload keys to section numbers (14 entries), and math topic_id to letter (au12→A, thompson_sampling→B). Update these when the payload schema adds sections or math topics.

PyMuPDF (`pip install pymupdf`) is used for PDF extraction when available; the checker also has a built-in fallback for the committed PDF's compressed JSON attachment. Non-PDF operations have zero dependencies.

## Integration

The `build_index` and `load_content` functions are importable. `automation/platform/spec_grounding.py` in {REDACTED}-platform can use them to replace its hardcoded Rosetta Stone mappings.
