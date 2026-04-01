# LSIE-MLF Hardening Summary

**Date:** 2026-04-01
**Baseline commit:** `4f9ee56` (Code clean-up chores with Claude)
**Purpose:** Clean up technical debt from the E2E experiment sprint so the baseline is safe for the ADO agent (MS Foundry GPT-5.4) to work against for feature updates.

---

## Phase 1 — Documentation Alignment

Updated all agent-facing documentation to match the actual 6-container topology.

| File | Changes |
|---|---|
| `.claude/skills/docker-topology/SKILL.md` | 5 → 6 containers, cuDNN 9 → 8 (SPEC-AMEND-001), scrcpy v2.x → v3.3.4 (SPEC-AMEND-004), added orchestrator service (SPEC-AMEND-003), added env vars, ubuntu 22.04 → 24.04 (SPEC-AMEND-002) |
| `.claude/skills/module-contracts/SKILL.md` | Module A: dual-instance scrcpy flags, video pipe. Module C: orchestrator container, stimulus injection, video capture self-healing. Module B: noted as implemented but not wired (EulerStream dependency). |
| `.claude/skills/ipc-pipeline/SKILL.md` | Replaced dd piping with direct `--record` to pipe, added video pipe, dual scrcpy architecture, staggered startup, PyAV consumer |
| `README.md` | 6-container topology table, cuDNN 8, scrcpy v3.3.4, orchestrator row, Thompson Sampling overview, dual-path data flow diagram, `data/sql/` in directory tree, Quick Start with `.env` copy and stimulus trigger |
| `.gitignore` | Removed `scripts/debug_e2e.sh` and `scripts/verify_e2e.sh` exclusions |

## Phase 2 — Spec Amendment Registry

| File | Changes |
|---|---|
| `docs/SPEC_AMENDMENTS.md` | **New file.** Formal registry of 6 spec deviations: SPEC-AMEND-001 (cuDNN 8), 002 (Ubuntu 24.04), 003 (orchestrator container), 004 (scrcpy v3.3.4 dual-instance), 005 (audio chunk 1/30s), 006 (INT8 locked). Each entry: ID, spec section, original text, new behavior, rationale, affected files. |
| `.env.example` | Added `ADB_SERVER_SOCKET` with §4.A documentation. Expanded `AUTO_STIMULUS_DELAY_S` comment. |

## Phase 3 — Test Coverage for Gap Fix Code

63 new tests across 6 new test files covering all code added during the E2E sprint.

| File | Tests | Coverage |
|---|---|---|
| `tests/unit/worker/pipeline/test_serialization.py` | 17 | encode/decode bytes fields, round-trip, idempotency, None handling |
| `tests/unit/worker/pipeline/test_stimulus.py` | 12 | Auto-trigger (delay, calibration, skip), Redis listener (subscribe, trigger, skip), publish |
| `tests/unit/worker/pipeline/test_video_capture.py` | 17 | VideoCapture init, get_latest_frame, is_running, stop, deque eviction |
| `tests/unit/worker/test_run_orchestrator.py` | 7 | main() env vars, signal handling, stimulus setup, graceful shutdown |
| `tests/unit/worker/tasks/test_inference_v3.py` | 6 | _FORWARD_FIELDS forwarding (all/missing/partial), base64 decode (encoded/None/raw) |
| `tests/unit/api/routes/test_stimulus.py` | 4 | POST /stimulus triggered, no subscribers warning, Redis 503, response schema |

**Total test count:** 247 → 310 (+63)

## Phase 4 — Code Quality Cleanup

| File | Changes |
|---|---|
| `services/worker/tasks/inference.py` | Removed 4 emoji log lines, replaced with 2 standard logger.info() calls |
| `services/worker/pipeline/orchestrator.py` | Video revival: bare `except: pass` → `logger.debug(exc_info=True)`. Assemble segment frame: bare `except: pass` → `logger.warning(exc_info=True)`. Added hasattr guard comments. AU12 frame failure: `logger.error()` → `logger.debug()` (expected per §12). Added §12 annotation to video revival block. |
| `packages/ml_core/face_mesh.py` | Verified close() method present (lines 75-86) |

## Phase 5 — Workspace-Server Configuration

| File | Changes |
|---|---|
| `workspace.json` | Updated from 2-step stub to 4-step CI config: ruff-lint, ruff-format, mypy, pytest. Schema adapted to workspace-server's actual `_get_ci_commands()` format (`ci_commands` with `step`/`cmd`). |
| `tests/unit/worker/pipeline/test_stimulus.py` | Fixed unused `typing.Any` import (ruff F401) |
| `tests/unit/worker/test_run_orchestrator.py` | Fixed unused `asyncio` import (ruff F401), reformatted |

## Phase 6 — Canonical Terminology Audit

Fixed 5 §0.3 violations introduced during E2E sprint:

| File | Line | Old term | Canonical term |
|---|---|---|---|
| `orchestrator.py` | 593 | "named pipe" | "IPC Pipe" |
| `video_capture.py` | 8 | "named FIFO pipe" | "IPC Pipe" |
| `video_capture.py` | 18 | "FIFO pipes" | "IPC Pipes" |
| `video_capture.py` | 55 | "named FIFO" | "mkfifo pipe" |
| `video_capture.py` | 108 | "named FIFO" | "IPC Pipe" |

## Phase 7 — Final Validation

### §13 Audit Checklist Results

| # | Item | Result | Notes |
|---|---|---|---|
| 1 | Directory structure | PASS | All §3.1 directories present |
| 2 | Docker topology | PASS | 6 containers (SPEC-AMEND-003), correct images/networks/volumes/restart |
| 3 | IPC lifecycle | PASS | mkfifo, exec 3<>, --record, wait -n crash recovery |
| 4 | Audio pipeline | PASS | FFMPEG_RESAMPLE_CMD matches §4.C.2 exactly |
| 5 | Drift correction | PASS | 30s poll, host_utc - android_epoch, freeze/3, reset/5min |
| 6 | AU12 implementation | PASS | Indices [61,291,33,133,362,263], epsilon 1e-6, clamp 5.0 |
| 7 | LLM determinism | PASS | temperature=0, top_p=1.0, seed=42, json_object format |
| 8 | Ephemeral Vault | PASS | AES-256-GCM, os.urandom(32), shred -vfz -n 3, 24h cycle |
| 9 | Schema validation | PASS | InferenceHandoffPayload Pydantic model matches §6.1 |
| 10 | Module contracts | PASS | All 6 modules implemented with inputs/outputs/failures |
| 11 | Error handling | PASS | All §12.1-12.4 categories covered across modules |
| 12 | Dependency versions | PASS | Pinned versions match §10.2 (==X.* is pip equivalent of X.x) |
| 13 | Variable traceability | PASS | All §11 output variables produced by correct modules |
| 14 | Data classification | PASS | Transient/debug/permanent tiers enforced per §5.2 |
| 15 | Canonical terminology | PASS | Zero retired synonyms in services/ packages/ docker-compose.yml |

**Result: 15/15 PASS**

---

## Files Modified (10)

- `.claude/skills/docker-topology/SKILL.md`
- `.claude/skills/ipc-pipeline/SKILL.md`
- `.claude/skills/module-contracts/SKILL.md`
- `.env.example`
- `.gitignore`
- `README.md`
- `services/worker/pipeline/orchestrator.py`
- `services/worker/pipeline/video_capture.py`
- `services/worker/tasks/inference.py`
- `workspace.json`

## Files Created (8)

- `docs/SPEC_AMENDMENTS.md`
- `tests/unit/api/routes/test_stimulus.py`
- `tests/unit/worker/pipeline/test_serialization.py`
- `tests/unit/worker/pipeline/test_stimulus.py`
- `tests/unit/worker/pipeline/test_video_capture.py`
- `tests/unit/worker/tasks/test_inference_v3.py`
- `tests/unit/worker/test_run_orchestrator.py`
- `HARDENING_SUMMARY.md`
