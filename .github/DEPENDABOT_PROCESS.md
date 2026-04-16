# Dependabot Handling Process

Operational rules for handling Dependabot pull requests against the LSIE-MLF dependency matrix. These rules are authoritative — Claude Code, the ADO agent, and human reviewers all follow this document. The companion `.claude/skills/dependabot-triage/SKILL.md` describes how Claude Code invokes this process; this file defines the rules.

The Dependabot configuration itself lives in `.github/dependabot.yml` (pip, github-actions, and three docker ecosystems on weekly or monthly cadence with `open-pull-requests-limit: 5` on pip to prevent queue overflow).

---

## Category 1 — Auto-merge

**Claude Code is authorized to review and merge these automatically without human intervention.**

A Dependabot PR is auto-mergeable when **every** condition below holds:

1. **Version delta is patch or minor** — `X.Y.Z` → `X.Y.(Z+1)` or `X.Y.Z` → `X.(Y+1).0`. Major version bumps (`X.Y.Z` → `(X+1).Y.Z`) are never auto-merged.
2. **The package is NOT listed in `docs/SPEC_REFERENCE.md` §10.2** (the dependency matrix). The §10.2 list is spec-governed; any bump there requires a registered SPEC-AMEND entry.
3. **The full CI suite passes**, including:
   - `ruff check` and `ruff format --check` on `packages/`, `services/`, `tests/`
   - `mypy` on `packages/` and `services/` with `--python-version 3.11`
   - `pytest tests/ -x -q` — and explicitly the **v3.0 math recipe regression tests in `tests/unit/test_v3_math_recipe.py`** must be green. These guard the Thompson Sampling / reward / AU12 mathematical contract; a failure there is never a flake.
   - `scripts/check_schema_consistency.py` (the four-source schema gate)
   - `docker compose config --quiet`
   - The §0.3 canonical-terminology audit
   - The dependency pin check in `scripts/check.sh`
4. **The PR diff does NOT touch `packages/ml_core/` or `packages/schemas/`**, including via transitive type-stub regeneration. These directories define the ML inference core and the inter-module type contracts — any change there is a contract change, not a maintenance update.
5. **No co-pending Dependabot PR exists for the same package** (avoids race-condition merges where two transitive bumps land in the wrong order).
6. **The package is not Python itself, CUDA, cuDNN, scrcpy, or any container base image** — those are governed by SPEC-AMEND-001 / SPEC-AMEND-002 / SPEC-AMEND-004.

When all six conditions hold, Claude Code approves and enables auto-merge:

```bash
gh pr review <number> --approve --body "Auto-merge: patch/minor bump, non-§10.2 package, CI green (incl. v3 math recipe), no ml_core/schemas changes."
gh pr merge <number> --auto --squash
```

The squash commit message must preserve the package name and version delta so that `git log --grep` finds it later.

---

## Category 2 — Human review

The following PRs are **never auto-merged** and require a human reviewer (ML lead for ml_core, API lead for schemas, infra lead for base images):

- **Major version bumps** — `(X+1).Y.Z`.
- **Updates to packages pinned in §10.2** — `faster-whisper`, `mediapipe`, `parselmouth`, `spacy`, `psycopg2-binary`, `pandas`, `celery`, `redis`, `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pycryptodome`, `patchright`, `TikTokLive`.
- **Updates to `faster-whisper`, `CTranslate2`, `mediapipe`, or `parselmouth`** — these four are explicitly called out because they govern the ML inference path's accuracy and latency contract. Even a patch bump can shift the Whisper transcription output, the MediaPipe landmark indexing, or the Praat acoustic feature extraction in ways that invalidate prior session data. CTranslate2 in particular sits beneath `faster-whisper` and is sensitive to compute_type compatibility (SPEC-AMEND-001 locks `int8`).
- **Any PR where CI fails**, regardless of category. A failing CI gate is never bypassed with `--admin`.

For every PR in this category, **Claude Code produces a short impact analysis** as a single PR comment using `gh pr comment <number> --body-file <path>`. The analysis covers three sections:

```markdown
## Dependabot Impact Analysis

**Package:** <name>  **Bump:** <old> → <new> (<patch|minor|major>)

### What changed upstream
<Summarize the upstream changelog or release notes between the two
 versions. Quote any breaking-change entries verbatim. Link to the
 release page. Note deprecations even when not breaking.>

### Which parts of the codebase use the changed functionality
<List every file in services/ and packages/ that imports the package
 (grep results). For each call site, name the public symbol used and
 whether its signature or behavior changed in the upstream diff. Pay
 particular attention to packages/ml_core/ and packages/schemas/.>

### Whether the update implies a spec amendment
<One of:
 - "No — the package is not in §10.2 and behavior is unchanged in the
    surfaces we use."
 - "Yes — package is pinned in §10.2; merging requires a SPEC-AMEND
    entry. Drafted entry: [text]."
 - "Conditional — the upstream changelog notes a behavior change in
    [surface]; if we adopt the new behavior we should register an
    amendment, otherwise we must add a regression test that pins the
    old behavior.">
```

The reviewer uses the analysis to decide merge vs hold vs close. Claude Code does NOT decide merge for Category 2 PRs — the analysis is decision support, not a rubber stamp.

---

## Category 3 — Cadence

Dependabot processing runs on a **fixed weekly day** (default: Monday), independent of feature merge activity. The cadence rules:

1. **Weekly sweep is non-negotiable.** Even an empty queue gets a sweep — the point is that the queue *stays* empty so that a critical-CVE PR is not buried under noise when it eventually arrives.
2. **Dependabot PRs are not bundled into feature merge reviews.** A feature PR review focuses on the feature; a Dependabot PR review focuses on the dependency. Mixing them produces confusing reviews where a reviewer cannot tell which change introduced a behavior shift. The two queues are processed on separate days and never share a merge window.
3. **A feature merge in flight pauses the Dependabot sweep.** If the post-merge playbook (`docs/POST_MERGE_PLAYBOOK.md`) is mid-execution when the weekly sweep day arrives, the sweep slips by one day rather than interleaving auto-merges with the playbook's chores.
4. **Out-of-band processing is allowed only for critical-severity CVE PRs.** Dependabot tags these via the GitHub Security Advisory feed. They jump the queue and are processed within 24 hours of arrival, regardless of cadence day.
5. **Backlog watchdog.** Any Category-2 PR open for more than 14 days is escalated: re-ping the assigned reviewer, and if still unaddressed at 21 days, close-with-reason and let Dependabot reopen on the next cycle. The point is to surface stalled reviews, not to silently let PRs rot.
6. **Queue health report.** At the end of every weekly sweep, Claude Code emits a one-paragraph report: total open, auto-merged this cycle, awaiting human review, stale (> 14 days), and any package that has had 3+ Dependabot PRs in the past month (signal of upstream instability — consider tightening the pin or excluding the package from Dependabot).

---

## Hard rules (apply across all three categories)

- Never bypass CI with `--admin` or merge a red PR. If CI is wrong, fix CI first.
- Never bump a §10.2-pinned package without a registered SPEC-AMEND entry — even if the bump is a patch and CI is green.
- Never auto-merge a PR that touches `packages/ml_core/` or `packages/schemas/`, even transitively.
- Never run a Dependabot sweep in parallel with a feature post-merge playbook execution.
- Never rewrite a Dependabot squash commit in a way that loses the package name and version delta.

---

## Cross-references

- Dependabot config: `.github/dependabot.yml`
- Pin list (§10.2): `docs/SPEC_REFERENCE.md`
- Spec deviation registry: `docs/SPEC_AMENDMENTS.md`
- CI gates: `scripts/check.sh`, `.github/workflows/ci.yml`
- Math recipe regression tests: `tests/unit/test_v3_math_recipe.py`
- Schema consistency gate: `scripts/check_schema_consistency.py`
- Operational skill: `.claude/skills/dependabot-triage/SKILL.md`
- Post-merge interaction: `docs/POST_MERGE_PLAYBOOK.md`
