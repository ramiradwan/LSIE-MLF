# LSIE-MLF

Real-time multimodal inference engine. All code decisions are governed by the single signed spec PDF at `docs/tech-spec-v*.pdf`; use `scripts/spec_ref_check.py` to extract, index, resolve, and validate its embedded content payload.

Build order and phase dependencies: @IMPLEMENTATION_PLAN.md

Deferred integrations (implemented but not wired — do not activate without satisfying the documented gating dependency): @docs/DEFERRED_INTEGRATIONS.md

Dependabot handling rules (auto-merge eligibility, human-review impact analysis, weekly cadence): @.github/DEPENDABOT_PROCESS.md — Dependabot PRs are processed on their own weekly cadence and are NEVER bundled into feature merge reviews. Do not include Dependabot bumps in a feature PR's review scope, and do not run the Dependabot sweep in parallel with the post-merge playbook.

## Canonical names (§0.3) — MUST use only these identifiers in all code and config

The v4 desktop runtime uses `services.desktop_app` terminology: ProcessGraph, IPC queues/shared-memory blocks, SQLite local state, and cloud outbox uploads. Container-era canonical names remain valid only when discussing retained legacy/server/cloud architecture, historical spec context, or deferred integrations; do not use them to describe the active desktop runtime.

- API Server — FastAPI application process for retained server/cloud routes. In the desktop runtime, `ui_api_shell` hosts a loopback FastAPI surface backed by SQLite.
- ML Worker — Legacy Celery consumer process for retained server/cloud inference tasks. In the desktop runtime, use `gpu_ml_worker` for the spawned local ML child process.
- Message Broker — Legacy Redis-backed Celery broker between the API Server and ML Worker. The v4 desktop runtime is Redis-free; use IPC queues/shared memory for local process transport and cloud outbox for upload durability.
- Persistent Store — PostgreSQL relational store for retained server/cloud analytical metrics and experiment state. In the desktop runtime, use SQLite local state.
- Capture Container — Legacy privileged Docker container for physical USB I/O and raw audio-visual stream extraction. In the desktop runtime, use `capture_supervisor`.
- Orchestrator Container — Legacy Module C orchestration-loop container that produces tasks consumed by the ML Worker. In the desktop runtime, use `module_c_orchestrator`.
- IPC Pipe — Legacy POSIX filesystem IPC transport under `/tmp/ipc/` for Capture Container to Orchestrator Container media transfer. In the desktop runtime, use IPC queues/shared-memory blocks.
- Ephemeral Vault — AES-256-GCM short-lived raw-media buffer regime with mandatory 24-hour secure deletion.
- InferenceHandoffPayload — JSON Schema/Pydantic contract for Module C→D and Module D→E exchange.
- PhysiologicalChunkEvent — normalized wearable telemetry chunk record emitted by hydration and drained by the Orchestrator Container.
- Physiological Context — per-subject physiological snapshot attached to each eligible `InferenceHandoffPayload` segment.
- Physiological State Buffer — in-memory rolling buffer keyed by `subject_role` for deriving physiological snapshots.
- subject_role — role discriminator for physiological data, currently `streamer` or `operator`.
- Co-Modulation Index — rolling Pearson correlation between time-aligned valid streamer/operator RMSSD sequences.
- segment_id — deterministic SHA-256 identifier derived from stable segment identity fields.
- BanditDecisionSnapshot — deterministic pre-update Thompson Sampling decision record attached for downstream attribution and evaluation.
- AttributionEvent — canonical Module E record for a stimulus-linked interaction eligible for event→outcome attribution.
- OutcomeEvent — canonical Module E record for a delayed downstream outcome such as `creator_follow`.
- EventOutcomeLink — canonical Module E record linking an `AttributionEvent` to an eligible `OutcomeEvent` under a versioned rule.
- AttributionScore — canonical Module E record for method-specific attribution or observational score values.
- semantic_method — canonical field naming the deterministic semantic scoring method used for an attribution record.
- semantic_method_version — canonical field naming the version of the active deterministic semantic method.
- bounded_reason_code — bounded semantic reason-code value; never persist unbounded semantic rationales.
- soft_reward_candidate — observational §7E soft semantic reward candidate persisted only for attribution scoring.

## Stack

Python 3.11.x only (3.12+ breaks CTranslate2). The v4 desktop runtime launches through `services.desktop_app`, using a spawned ProcessGraph, local SQLite state, IPC queues/shared memory, and cloud outbox durability without Docker Compose or Redis. Retained server/cloud surfaces still use FastAPI, Celery, Redis, and PostgreSQL where explicitly scoped. CUDA ≥12 + cuDNN ≥8 remains required for GPU-backed ML paths. All Pydantic models live in `packages/schemas/`. All ML utilities live in `packages/ml_core/`.

## Hard rules

- NEVER persist raw biometric media (facial images, voiceprints, raw audio) in PostgreSQL. Only anonymized metrics.
- All inter-module payloads validated via Pydantic before dispatch. No untyped dicts crossing module boundaries.
- No active Docker Compose or Dockerfile manifests are tracked for the v4 desktop runtime; do not add Docker launch or validation instructions unless a future spec change reintroduces manifests.
- Keep `pyproject.toml` base dependencies and the `ml_backend` extra split so the desktop/API runtime surface does not hydrate ML-heavy packages unless explicitly requested.
- Pinned versions in `pyproject.toml` and `uv.lock` are authoritative. Do not upgrade without spec justification.
- Use `from __future__ import annotations` in every Python file.
- Full type annotations on all function signatures. No `Any` unless the spec explicitly defines a flexible dict.

## Verify changes

```bash
# Full local gate mirror
bash scripts/check.sh

# Individual enforced gates
uv run ruff check packages/ services/ tests/
uv run ruff format --check packages/ services/ tests/
uv run mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases
uv run pytest tests/ -x -q --tb=short
uv run python scripts/check_schema_consistency.py
uv run python scripts/run_audit.py --strict
# No Docker Compose/Dockerfile gate exists for the active v4 desktop runtime.
# Canonical terminology audit — must return 0 results
# Run the canonical-name grep recipe in .claude/commands/audit.md exactly:
# use its grep -rnE pattern against services/ packages/ scripts/ and do not filter comments or docstrings.
# Spec-reference validation when regenerating ignored docs/content.json or touching §-refs/spec payload
python scripts/spec_ref_check.py --validate
```
