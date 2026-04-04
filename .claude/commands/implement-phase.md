Implement phase $ARGUMENTS from IMPLEMENTATION_PLAN.md.

Steps:
1. TRUST GATE: Run `python scripts/verify_spec_signature.py docs/tech-spec-v3.0.pdf`. If FAIL, STOP and report. Do not implement anything against an unverified spec.
2. Read IMPLEMENTATION_PLAN.md and find the exact phase requested.
2. For each file in that phase, read the current stub to understand the scaffold.
3. Implement each file fully, replacing all `raise NotImplementedError` with working code.
4. Every implementation decision MUST trace to a specific spec section. Add inline comments citing the section.
5. After implementing each file, run `mypy <file> --python-version 3.11 --strict` and fix any type errors.
6. Write or update unit tests in `tests/` for the implemented code.
7. Run `pytest tests/ -x -q` to verify all tests pass.
8. Summarize what was implemented and which spec sections governed each decision.
