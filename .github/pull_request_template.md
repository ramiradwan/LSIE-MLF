## Spec Reference

Which sections of the LSIE-MLF Technical Specification v3.1 (`docs/tech-spec-v3.1.pdf`) govern this change? Resolve §N refs with `python scripts/spec_ref_check.py --resolve "<ref>"`.

- §

## Implementation Phase

If an active plan is in flight, which phase of `IMPLEMENTATION_PLAN.md` does this PR belong to? Otherwise leave blank.

- Phase:

## Changes

Describe what was implemented or modified.

## Verification

- [ ] `mypy packages/ services/ --python-version 3.11 --strict` passes
- [ ] `pytest tests/ -x -q` passes
- [ ] `docker compose config --quiet` passes
- [ ] No retired synonyms from §0.3 in changed files
- [ ] All new functions have full type annotations
- [ ] All new functions have `from __future__ import annotations`
- [ ] No raw biometric data persisted to PostgreSQL
- [ ] Dependency versions match §10.2 matrix
