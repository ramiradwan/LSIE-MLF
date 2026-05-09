---
name: dependabot-triage
description: Process Dependabot pull requests against the LSIE-MLF dependency matrix on a weekly cadence. Apply the auto-merge eligibility rules, file the impact analysis for PRs that need human review, and keep the queue empty between cycles. Triggers include "dependabot", "dep updates", "triage dep PRs", "auto-merge eligible", or "weekly dep sweep".
---

# Dependabot Triage (`pyproject.toml` + `uv.lock`, §10.2)

## When to invoke

- Weekly dependency sweep (default cadence: every Monday).
- When the user says "triage dependabot", "process dep PRs", "auto-merge what's safe", or asks about the dep update queue.
- Immediately if a Dependabot PR appears for a critical-severity CVE — out-of-band, do not wait for the weekly cadence.

## Required inputs

1. The list of open Dependabot PRs: `gh pr list --author "app/dependabot" --state open`.
2. For each PR: title (package + version bump), changed files (`pyproject.toml`, `uv.lock`, `.github/workflows/*.yml`, etc.), and CI status (`gh pr checks <number>`).
3. Signed spec dependency matrix (§10.2 / `dependency_matrix.pinned_packages`, resolved with `scripts/spec_ref_check.py`) — anything pinned there is spec-governed and cannot be auto-merged.

## Auto-merge eligibility rules — ALL must be true

A Dependabot PR is eligible for auto-merge ONLY when every condition holds:

1. **Version delta is patch or minor.** Major version bumps (`X.Y.Z` → `(X+1).Y.Z`) always require human review. Inspect the title or run `gh pr view <number>`.
2. **The package is not pinned in the signed spec dependency matrix (§10.2 / `dependency_matrix.pinned_packages`).** Resolve the current matrix with `scripts/spec_ref_check.py` or the embedded content payload. Any bump to a pinned package requires an updated signed spec/content payload before merge.
3. **CI is fully green.** All workflow runs on the PR head must pass — ruff lint, ruff format, mypy, pytest, schema-consistency check, canonical terminology audit, and dependency-pin check.
4. **The PR does NOT modify `packages/ml_core/` or `packages/schemas/`.** These directories are the ML inference core and the inter-module type contracts. A dep bump that touches them via transitive type-stub or generated-code changes is a contract change, not a maintenance update.
5. **The package is not Python itself, CUDA, cuDNN, scrcpy, or any base image.** Runtime base changes are governed by the signed runtime/dependency matrix and require an updated signed spec/content payload.
6. **No co-pending Dependabot PR for the same package** (avoids race-condition merges where two transitive bumps land in the wrong order).

If all six conditions hold, approve and enable auto-merge:

```bash
gh pr review <number> --approve --body "Auto-merge: patch/minor bump, non-pinned package, CI green, no ml_core/schemas changes."
gh pr merge <number> --auto --squash
```

## Impact analysis template — for PRs that require human review

Any PR failing one or more eligibility rules gets a comment with this template before being left for human review. Fill every field; an empty field is a TODO, not an OK.

```markdown
## Dependabot Impact Analysis

**Package:** <name>
**Bump:** <old version> → <new version> (<patch | minor | major>)
**Pinned in §10.2:** <yes / no — if yes, name the pin>
**Touches ml_core or schemas:** <yes / no — if yes, list files>
**CI status:** <green / failing — if failing, name the job>

### Why this needs human review
<one or more of: major-version bump, pinned package, ml_core/schemas change, CI failure, base-image change>

### Risk summary
- **Breaking changes in upstream changelog:** <link to release notes; list any breaking entries>
- **Transitive dependency shifts:** <summarize the `uv.lock` delta or equivalent; list any other pinned packages whose resolved version would change>
- **API surface changes affecting our usage:** <grep results for the public symbols we import; list any signatures we depend on that changed>
- **Runtime behavior changes:** <perf regressions, deprecation warnings, default config changes>

### Required actions before merge
- [ ] Signed spec/content payload updated (if pinned package)
- [ ] Targeted tests added or run for the changed surface (if ml_core/schemas)
- [ ] Manual smoke test of <module>
- [ ] Verify performance baseline still within tolerance (if `services/worker/` touched — see Standing Chore #7)

### Recommendation
<merge / hold / close — and one-sentence reason>
```

## Weekly cadence

Once per week (default Monday), execute the full sweep:

1. **Inventory.** `gh pr list --author "app/dependabot" --state open --json number,title,headRefName,createdAt`. Note any PR older than 14 days — these are stale and need either escalation or close-with-reason.
2. **Classify.** For each PR, walk the six eligibility rules in order. The first failed rule decides the disposition (auto-merge vs human-review). Do not silently skip rules — record which rule was the disqualifier.
3. **Auto-merge sweep.** Approve and enable auto-merge on every eligible PR in a single batch. Wait for CI to land them sequentially; do not force-merge.
4. **Impact analysis.** For every ineligible PR, post the template comment as a single PR comment (`gh pr comment <number> --body-file <path>`). Tag the human reviewer responsible for the affected area (ML core → ML lead, schemas → API lead, base images → infra lead).
5. **Queue health report.** At the end of the sweep, produce a short summary: total open PRs, auto-merged this cycle, awaiting human review, stale (> 14 days), and any package that has had three or more Dependabot PRs in the past month (signal of upstream instability — consider pinning).

If the queue is empty, the sweep takes seconds. The point of the cadence is that the queue **stays** empty; large backlogs hide critical CVEs behind noise.

## Hard rules

- NEVER bypass CI with `--admin` or merge a red PR. If CI is wrong, fix CI first.
- NEVER bump a §10.2-pinned package without an updated signed spec/content payload — even if the bump is a patch and CI is green. Pinned versions are pinned for a reason.
- NEVER squash a Dependabot PR's commit message into something that loses the package name and version delta — the audit trail depends on `git log --grep` finding bumps later.
- NEVER auto-merge a PR that touches `packages/ml_core/` or `packages/schemas/`, even transitively via stub regeneration.
- NEVER process Dependabot PRs in parallel with a feature merge — the auto-merge queue can interleave with the post-merge playbook and produce a confusing main-branch state.

## Cross-references

- Pin list: signed spec §10.2 via `scripts/spec_ref_check.py --resolve 10.2` / `dependency_matrix.pinned_packages`
- CI gates: `scripts/check.sh`, `.github/workflows/ci.yml`
- Post-merge follow-up: see the `post-merge-playbook` skill
