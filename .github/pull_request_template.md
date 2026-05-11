## Spec Reference

Which sections of the current LSIE-MLF Technical Specification (`docs/tech-spec-v*.pdf`) govern this change? Resolve §N refs with `python scripts/spec_ref_check.py --resolve "<ref>"` (the command uses the single committed spec PDF by default).

- §

## Spec Work Item

If this PR implements a local spec-work-item packet, name the packet title or local path. Do not commit files under `automation/work-items/active/`.

- Work item:

## Changes

Describe what was implemented or modified.

## Verification

- [ ] `uv run mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases` passes
- [ ] `uv run pytest tests/ -x -q --tb=short` passes
- [ ] No retired synonyms from §0.3 in changed files
- [ ] All new functions have full type annotations
- [ ] All new functions have `from __future__ import annotations`
- [ ] No raw biometric data persisted to PostgreSQL
- [ ] Dependency versions match §10.2 matrix
