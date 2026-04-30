Implement the file at path $ARGUMENTS.

Steps:
1. TRUST GATE: Resolve the single `docs/tech-spec-v*.pdf` match and run `python scripts/verify_spec_signature.py <resolved-pdf>`. If FAIL, STOP and report.
2. Read the current stub at the given path.
2. Read IMPLEMENTATION_PLAN.md to understand which phase this file belongs to and what dependencies it has.
3. Check that all upstream dependencies (files from earlier phases) are already implemented (no NotImplementedError).
4. Implement the file fully, tracing every decision to a spec section with inline comments.
5. Run `mypy <file> --python-version 3.11 --strict` and fix type errors.
6. Write or update the corresponding test file in `tests/`.
7. Run `pytest tests/ -x -q` and confirm all tests pass.
8. Show a brief summary of what was implemented.
