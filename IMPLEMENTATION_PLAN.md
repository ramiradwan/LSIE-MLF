# LSIE-MLF Implementation Plan — Hardening & Cleanup
# Replaces the original 8-phase build plan (all phases complete)

**Purpose:** Clean up the technical debt accumulated during the E2E experiment sprint so the baseline is safe for the ADO agent (MS Foundry GPT-5.4) to work against for feature updates.

**Context:** The system passed all E2E verification checks. The Thompson Sampling loop fires, posteriors update, and the full USB → PostgreSQL data flow works. But the rush to get E2E working left documentation drift, missing test coverage, inconsistent error handling, informal logging, and untracked spec amendments. If the ADO agent starts feature work against this baseline, it will inherit and amplify these inconsistencies.

**Execution:** Claude Code, phased, with spec trust gate enforced. Each phase is self-contained and commits cleanly. Run `scripts/check.sh` after each phase to confirm no regressions.

**Module B clarification:** The GroundTruthIngester (§4.B) is fully implemented with the SignatureProvider protocol, EulerStreamSigner, WebSocket reconnection, and Action_Combo constraint. However, it is not wired into the orchestrator loop and was not exercised during the E2E run. Module B requires the EulerStream third-party signature API for TikTok WebSocket authentication. This is an external dependency that is not available in the current test environment. The code is correct and tested in isolation — it simply has no integration point yet. Wiring Module B into the orchestrator is a feature update task for the ADO agent, not a hardening task. It should be planned as an ADO work item with a clear dependency on EulerStream API access.

---

## Phase 1 — Documentation Alignment

Fix all documentation that the ADO agent or the workspace-server reads as context. If these are wrong, the agent makes wrong decisions.

**1.1 `.claude/skills/docker-topology/SKILL.md`**
- Change container count from 5 to 6 (add orchestrator)
- Change worker image from `cudnn9` to `cudnn8` (SPEC-AMEND-001)
- Change scrcpy version from `v2.x` to `v3.3.4`
- Add orchestrator service: same worker image, command `python3.11 -m services.worker.run_orchestrator`, depends on redis + postgres + stream_scrcpy
- Add stream_scrcpy environment variables: SDL_VIDEODRIVER=dummy, XDG_RUNTIME_DIR=/tmp, ADB_SERVER_SOCKET
- Add worker environment variables: CELERYD_CONCURRENCY=1, OMP_NUM_THREADS=1, HF_HOME
- Update Capture Container base from ubuntu:22.04 to ubuntu:24.04 (SPEC-AMEND-002)

**1.2 `.claude/skills/module-contracts/SKILL.md`**
- Module C contract: add orchestrator container as separate process, stimulus injection trigger mechanism (Redis pub/sub + auto-trigger), video capture thread self-healing
- Module A contract: update scrcpy flags (--record to pipe, --no-playback, --port ranges, --tunnel-host), remove dd piping reference, add dual-instance architecture
- Add note that Module B is implemented but not wired (EulerStream dependency)

**1.3 `.claude/skills/ipc-pipeline/SKILL.md`**
- Replace dd piping description with direct --record to named pipe
- Add video pipe: `/tmp/ipc/video_stream.mkv` (MKV streaming container)
- Add dual scrcpy instance architecture: audio on port 27100:27199, video on port 27200:27299
- Update scrcpy flags to match current entrypoint.sh
- Add staggered startup note (4s delay between instances to prevent ADB collision)

**1.4 `README.md`**
- Container topology table: 5 → 6 containers, add orchestrator row
- cuDNN: ≥ 9.0 → ≥ 8.0 (with SPEC-AMEND-001 note)
- scrcpy: v2.x → v3.3.4
- Data flow diagram: add orchestrator as separate process, add video path, add reward pipeline
- Directory structure: add data/sql/ directory
- Add Thompson Sampling experiment description to System Overview
- Quick Start: add .env copy step, mention AUTO_STIMULUS_DELAY_S, mention stimulus trigger endpoint
- Worker image in topology table: cudnn9 → cudnn8

**1.5 `.gitignore` audit**
- Currently excludes `scripts/debug_e2e.sh` and `scripts/verify_e2e.sh`. These are operational tools that should be tracked. Remove those exclusion lines.

**Why first:** The ADO agent's workspace-server loads .claude/skills/ files as context for refactor planning and code generation. If the skills describe a 5-container topology with cuDNN 9 and dd piping, the agent will generate code that conflicts with the actual system. Documentation must be correct before any agent touches the code.

---

## Phase 2 — Spec Amendment Registry

Create a formal tracking mechanism for spec deviations. The tech spec (v2.0) is a signed PDF that cannot be edited. Runtime deviations exist in code comments but are not consolidated. The ADO agent and review agent need a single authoritative source for "what the spec says" vs "what the implementation does."

**2.1 `docs/SPEC_AMENDMENTS.md`**
Create a registry of all known deviations from tech-spec-v2.0.pdf:

- SPEC-AMEND-001: §9.1 Worker image cuDNN 9 → cuDNN 8 for Pascal (SM 6.1) dp4a compatibility. Affects: Dockerfile, dependency matrix.
- SPEC-AMEND-002: §9.1 Capture Container Ubuntu 22.04 → Ubuntu 24.04 for GLIBC 2.38+ (scrcpy v3.1+ prebuilt binary). Affects: Dockerfile.
- SPEC-AMEND-003: §9.1 Container count 5 → 6. Orchestrator runs as a separate container (same image as worker, different CMD). Spec §4.C assumed Module C runs inside the worker process. Affects: docker-compose.yml.
- SPEC-AMEND-004: §4.A.1 scrcpy v2.x → v3.3.4. Dual-instance architecture (audio + video on separate port ranges). Direct --record to pipe replaces dd/fd3 for video path. Audio path retains exec 3<> pipe shield. Affects: entrypoint.sh.
- SPEC-AMEND-005: §4.C.2 Audio chunk size changed from 1s (32000 bytes) to 1/30s (~1067 bytes) to match video frame rate for temporal alignment. Affects: orchestrator.py run() loop.
- SPEC-AMEND-006: §4.D.1 TranscriptionEngine compute_type locked to "int8" (not configurable). SPEC-AMEND-001 downstream effect — INT8 uses dp4a which is available on Pascal, FP16 is not.

Each entry: amendment ID, spec section, original text, new behavior, rationale, affected files.

**2.2 `.env.example` audit**
- Add AUTO_STIMULUS_DELAY_S with documentation
- Add ADB_SERVER_SOCKET with documentation
- Verify all env vars used in docker-compose.yml are documented

**Why second:** The spec amendment registry is the document the review agent loads alongside the spec sections when checking PR compliance. Without it, the review agent will flag SPEC-AMEND-001 through 006 as violations.

---

## Phase 3 — Test Coverage for Gap Fix Code

The six gap fixes and two minor fixes from the E2E sprint added new code that has no tests. This is the highest-risk technical debt — the ADO agent may modify these files during feature work, and without tests, regressions will be silent.

**3.1 `tests/unit/worker/pipeline/test_serialization.py`**
- Test encode_bytes_fields: bytes → base64 string, None → None, already-string → unchanged (idempotent)
- Test decode_bytes_fields: base64 string → bytes, None → None, already-bytes → unchanged
- Test round-trip: encode then decode produces original bytes
- Test empty dict handling
- Test non-target fields are untouched

**3.2 `tests/unit/worker/pipeline/test_stimulus.py`**
- Test setup_auto_trigger: fires after delay, skips if already injected, returns None when delay=0
- Test start_redis_listener: mock Redis pub/sub, verify record_stimulus_injection called on message
- Test publish_stimulus_trigger: mock Redis client, verify publish called with correct channel

**3.3 `tests/unit/worker/pipeline/test_video_capture.py`**
- Test VideoCapture init: default pipe path, buffer maxlen
- Test get_latest_frame: returns None on empty buffer, returns latest frame, drains buffer
- Test is_running property: False before start, True during run, False after stop
- Test stop: sets _running=False, clears buffer
- Mock PyAV container for _decode_loop (do not test actual pipe opening)

**3.4 `tests/unit/worker/test_run_orchestrator.py`**
- Test main(): verify Orchestrator is instantiated with env vars
- Test signal handling: SIGTERM calls orchestrator.stop()
- Test stimulus trigger setup: auto_timer and redis_thread created
- Mock all infrastructure (asyncio loop, Orchestrator, stimulus module)

**3.5 `tests/unit/worker/tasks/test_inference_v3.py`** (supplement, not replace)
- Test _FORWARD_FIELDS forwarding: all six fields present in persist_metrics dispatch
- Test _FORWARD_FIELDS forwarding: missing fields in input are not added to output
- Test base64 decode: payload with base64-encoded _audio_data is decoded before ML pipeline
- Test base64 decode: None _audio_data passes through unchanged

**3.6 `tests/unit/api/routes/test_stimulus.py`**
- Test POST /api/v1/stimulus: mock Redis, verify publish called, check response schema
- Test stimulus with no subscribers: returns warning in response
- Test Redis unavailable: returns 503

**Why third:** Test coverage is the safety net for the ADO agent. Every file it might modify must have tests that break if the behavior changes.

---

## Phase 4 — Code Quality Cleanup

Remove E2E debugging artifacts, standardize error handling and logging, restore spec-section annotations where they eroded.

**4.1 `services/worker/tasks/inference.py` — Remove emoji logging**
The E2E debugging added emoji log lines (⏳, ✅, 📦) around Whisper model loading. Replace with standard logger.info() calls using the project's established pattern. The information content (model loading, download progress) is useful — only the formatting is wrong.

**4.2 `services/worker/pipeline/orchestrator.py` — Tighten error handling**
- Video revival block: bare `except Exception: pass` → `except Exception: logger.debug("Video revival failed", exc_info=True)`. Silent swallowing masks real errors.
- assemble_segment frame extraction: bare `except Exception as e: pass` → proper logging with exc_info=True
- _process_video_frame: the `hasattr(frame, "to_ndarray")` guard was added during E2E to handle both PyAV frames and raw numpy arrays. Add a comment explaining why this guard exists and when each branch fires.

**4.3 Spec-section annotation sweep**
Run a diff between the orchestrator.py we produced (with full §4.C.1, §12, §7.4 annotations) and the current repo version. Identify any spec-section comments that were lost during the E2E patching and restore them. Same for inference.py.

**4.4 `packages/ml_core/face_mesh.py` — Verify close() method exists**
Confirm the close() method fix was applied during E2E. If not, add it.

**4.5 Logging level audit**
- AU12 frame processing failures are currently logger.error() in the repo but should be logger.debug() (frame drops are expected and non-critical per §12)
- Verify all Module D pipeline failures use logger.warning() (recoverable) vs logger.error() (unrecoverable) consistently

**Why fourth:** Standardized code quality prevents the ADO agent from inheriting and replicating bad patterns. When it reads existing code to understand project conventions, it should see clean, consistent patterns.

---

## Phase 5 — Workspace-Server Configuration

The workspace-server (used by the ADO Foundry agent) loads CI configuration and spec governance from the repository. These files don't exist yet.

**5.1 `workspace.json`**
Create the CI configuration file at repository root:
```json
{
  "ci": {
    "steps": [
      {"name": "ruff-lint", "command": "ruff check packages/ services/ tests/"},
      {"name": "ruff-format", "command": "ruff format --check packages/ services/ tests/"},
      {"name": "mypy", "command": "mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases"},
      {"name": "pytest", "command": "pytest tests/ -x -q --tb=short"}
    ]
  }
}
```

Verify the workspace-server's `run_ci` function accepts this schema by checking its documentation. If it expects a different format, adapt accordingly.

**5.2 `docs/content.json`** (if spec governance is needed for the ADO pipeline)
This is the spec content index that the workspace-server's refactor planner validates against. Generation requires parsing the tech-spec-v2.0.pdf table of contents into a JSON structure with section IDs and titles. If the pipeline team has not yet defined the schema, defer this to the pipeline handoff and note it as a dependency.

**Why fifth:** Without workspace.json, the workspace-server falls back to hardcoded CI steps that may not match the project's actual CI configuration. The ADO agent's CI gate would use different commands than the GitHub Actions CI, producing inconsistent results.

---

## Phase 6 — Canonical Terminology Audit

Run the §0.3 retired synonym grep and fix any violations introduced during the E2E sprint.

**6.1 Run the canonical terminology check**
```bash
grep -rn "Celery node\|GPU worker\|inference worker\|task queue\|FIFO\|named pipe\|POSIX pipe\|audio pipe\|kernel pipe\|24-hour vault\|data vault\|transient storage\|secure buffer\|handoff schema\|payload schema\|inference payload\|FastAPI server\|web server\|ASGI server\|Celery worker\|scrcpy container\|capture service\|stream ingester\|relational database" services/ packages/ docker-compose.yml
```

Fix any matches found in code or comments (note: README.md uses some of these terms in the System Overview section for readability — those are acceptable in documentation context but not in code or docstrings).

**6.2 Run the full CI suite**
```bash
bash scripts/check.sh
```

All checks must pass. This is the gate for declaring the baseline clean.

**Why sixth:** The canonical terminology audit is item 15 on the §13 audit checklist. It's the final validation that the codebase speaks the spec's language consistently.

---

## Phase 7 — Final Validation

**7.1 Run the 15-item autonomous audit checklist**
Execute `.claude/commands/audit.md` against the hardened codebase. Every item should pass. Document any items that cannot pass (e.g., if the spec says cuDNN 9 but the implementation uses cuDNN 8, the audit item fails — the SPEC_AMENDMENTS.md explains why).

**7.2 Produce a HARDENING_SUMMARY.md**
Short document listing every change made during this hardening run: files modified, tests added, documentation updated, spec amendments registered. This becomes the "before" snapshot for the ADO agent — it knows exactly what state the baseline was in when it starts feature work.

**7.3 Commit and tag**
Single commit with message:
```
chore: harden baseline for ADO agent feature work

- Phase 1: Documentation alignment (skills, README, .gitignore)
- Phase 2: Spec amendment registry (6 amendments tracked)
- Phase 3: Test coverage for gap fix code (6 new test files)
- Phase 4: Code quality cleanup (logging, error handling, annotations)
- Phase 5: Workspace-server configuration (workspace.json)
- Phase 6: Canonical terminology audit (§0.3 compliance)
```

Tag: `v3.0-hardened-baseline`

This tag is the safe starting point for the ADO agent's feature branch.

---

## What This Plan Does NOT Cover

These are feature work items for the ADO agent, not hardening tasks:

- **Module B integration** — Wiring GroundTruthIngester into the orchestrator loop. Requires EulerStream API access. ADO work item with external dependency.
- **Value validation** — Confirming that actual AU12/reward values have real utility for the greeting line optimization use case. Requires multiple real sessions with live hosts. Analysis task, not code task.
- **Tech spec v2.1 updates** — Formally amending §9.1 (container count, cuDNN, scrcpy version), §4.A.1 (pipe architecture), §4.C (orchestrator separation). Requires spec author action.
- **Contextual Thompson Sampling** — Phase 2 of the mathematical recipe (contextual features, GLM). Deferred until per-arm observations reach n ≈ 50-100.
- **Macro-conversion tracking** — Module F scraping for `followed_back` boolean. Requires patchright integration with TikTok profile pages
