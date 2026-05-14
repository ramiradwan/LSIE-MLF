# Operator Console Designer Export

This branch contains immutable designer-reference exports for the LSIE-MLF Operator Console. It is not the PySide6 implementation source of truth and must not be mutated from pull-request workflows.

## Exports

- `exports/bootstrap-2026-05-14/` — bootstrap export copied from the external HTML/CSS reference folder on 2026-05-14.

## Contract

Each export has:

- `reference/` — designer-facing HTML/CSS reference pages and approved synthetic reference images.
- `contract/tokens.json` — machine-readable token contract consumed by `main`.
- `contract/reference_capture_manifest.json` — HTML capture selectors for advisory designer-reference screenshots.
- `contract/reference_to_qt_mapping.json` — mapping from designer reference IDs/routes to PySide route/objectName concepts.
- `export_manifest.json` — SHA-256 hashes for export files.

`main` consumes only verified manifests and generated PySide artifacts. Screenshot baselines remain outside `main`.