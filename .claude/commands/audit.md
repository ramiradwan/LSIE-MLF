Run the executable autonomous implementation audit checklist from §13 of the current spec.

Use the audit harness as the source of truth for item discovery, verifier dispatch, strict pass/fail status, and the Markdown report:

```bash
python scripts/run_audit.py --strict
```

The harness discovers the single committed `docs/tech-spec-v*.pdf`, extracts its embedded content, enumerates the current §13 checklist at runtime, dispatches registered verifiers, prints a deterministic Markdown table to stdout, and exits nonzero when any dispatched item fails. Do not copy a fixed item list, duplicate verifier grep recipes, or assume an item count; the committed PDF and registered harness verifiers are authoritative.

For post-merge cycle logs, capture stdout verbatim and append it to the log so adjacent runs can be diffed against the harness-rendered table:

```bash
python scripts/run_audit.py --strict > /tmp/section13-audit.md
cat /tmp/section13-audit.md >> docs/artifacts/<cycle-log>.md
```

Resolve individual spec references when investigating a harness failure with:

```bash
python scripts/spec_ref_check.py --resolve "13.<n>" --json
```
