Verify the implementation of Module $ARGUMENTS against its formal contract from the spec.

Steps:
1. Identify which module letter (A–F) was requested.
2. Read the relevant source files for that module.
3. Check every contract item: inputs match, outputs match, dependencies present, side effects handled, all failure modes implemented.
4. Cross-reference with §12 error handling matrix for that module across all 4 failure categories.
5. Check that only canonical names from §0.3 are used.
6. Run any existing tests for that module.
7. Report a checklist: each contract item as PASS/FAIL with file:line evidence.
