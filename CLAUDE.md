# LSIE-MLF

Real-time multimodal inference engine. All code decisions governed by the spec: @docs/SPEC_REFERENCE.md

Build order and phase dependencies: @IMPLEMENTATION_PLAN.md

## Canonical names (§0.3) — MUST use only these identifiers in all code and config

api, worker, redis, postgres, stream_scrcpy, IPC Pipe, Ephemeral Vault, InferenceHandoffPayload, ML Worker, API Server, Message Broker, Persistent Store, Capture Container

## Stack

Python 3.11.x only (3.12+ breaks CTranslate2). FastAPI (api container), Celery (worker container), Redis (broker), PostgreSQL (store). CUDA ≥12 + cuDNN ≥9 in worker. All Pydantic models in `packages/schemas/`. All ML utilities in `packages/ml_core/`.

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
grep -rn "Celery node\|GPU worker\|inference worker\|task queue\|FIFO\|named pipe\|24-hour vault" services/ packages/ | grep -vP ':\s*#|"""' || true
```
