# LSIE-MLF

Real-time multimodal inference engine. All code decisions governed by the spec: @docs/SPEC_REFERENCE.md

Build order and phase dependencies: @IMPLEMENTATION_PLAN.md

Deferred integrations (implemented but not wired — do not activate without satisfying the documented gating dependency): @docs/DEFERRED_INTEGRATIONS.md

Dependabot handling rules (auto-merge eligibility, human-review impact analysis, weekly cadence): @.github/DEPENDABOT_PROCESS.md — Dependabot PRs are processed on their own weekly cadence and are NEVER bundled into feature merge reviews. Do not include Dependabot bumps in a feature PR's review scope, and do not run the Dependabot sweep in parallel with the post-merge playbook.

## Canonical names (§0.3) — MUST use only these identifiers in all code and config

- API Server — FastAPI application process that serves REST endpoints on port 8000.
- ML Worker — Celery consumer process that executes GPU-bound ML inference tasks.
- Message Broker — Redis in-memory store that brokers Celery dispatch between the API Server and ML Worker.
- Persistent Store — PostgreSQL relational store for analytical metrics and experiment state.
- Capture Container — privileged Docker container for physical USB I/O and raw audio-visual stream extraction.
- Orchestrator Container — Module C orchestration-loop container that produces tasks consumed by the ML Worker.
- IPC Pipe — POSIX filesystem IPC transport under `/tmp/ipc/` for Capture Container to Orchestrator Container media transfer.
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

Python 3.11.x only (3.12+ breaks CTranslate2). API Server (FastAPI), ML Worker (Celery consumer), Message Broker (Redis), and Persistent Store (PostgreSQL). CUDA ≥12 + cuDNN ≥8 in the ML Worker (SPEC-AMEND-001). All Pydantic models in `packages/schemas/`. All ML utilities in `packages/ml_core/`.

## Hard rules

- NEVER persist raw biometric media (facial images, voiceprints, raw audio) in PostgreSQL. Only anonymized metrics.
- All inter-module payloads validated via Pydantic before dispatch. No untyped dicts crossing module boundaries.
- API Server image MUST exclude ML dependencies. Worker image MUST exclude web assets. Enforced via separate requirements files.
- Dockerfile build context is always monorepo root so `packages/` is accessible.
- Pinned versions in `requirements/` are authoritative. Do not upgrade without spec justification.
- Use `from __future__ import annotations` in every Python file.
- Full type annotations on all function signatures. No `Any` unless the spec explicitly defines a flexible dict.

## Verify changes

```bash
# Type check
mypy packages/ services/ --python-version 3.11 --strict
# Tests
pytest tests/ -x -q
# Docker topology
docker compose config --quiet
# Canonical name audit — must return 0 results
# Run the canonical-name grep recipe in .claude/commands/audit.md exactly:
# use its grep -rnE pattern against services/ packages/ scripts/ and do not filter comments or docstrings.
```
