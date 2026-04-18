# LSIE-MLF Specification Amendment Registry

This document tracks all known deviations from the **Master Technical Specification** (signed PDF, `docs/tech-spec-v3.1.pdf`). The spec is a signed PDF that cannot be edited. This registry is the single authoritative source for "what the spec says" vs "what the implementation does."

Each amendment was made for a documented technical reason. The review agent should treat these as accepted deviations, not violations.

SPEC-AMEND-001 through SPEC-AMEND-006 were originally raised against the v3.0 text and carry forward unchanged against v3.1 (none of them were incorporated into the v3.1 base text). SPEC-AMEND-007 was issued as the v3.1 physiology extension and is now embedded in the base spec text itself — it is retained here for historical traceability of the v3.0 → v3.1 transition.

---

## SPEC-AMEND-001: Worker GPU Architecture Downgrade

| Field | Value |
|---|---|
| **Spec section** | §9.1 — Container Specifications |
| **Original text** | Worker image: `nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04` |
| **New behavior** | Worker image: `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` |
| **Rationale** | cuDNN 9 lacks SM 6.1 (Pascal) binaries. The target hardware (GTX 1080 Ti) requires cuDNN 8 for dp4a INT8 vectorization. cuDNN 9 causes a phantom docker tag CI failure and runtime incompatibility. |
| **Affected files** | `services/worker/Dockerfile`, `docker-compose.yml` (worker + orchestrator services), `packages/ml_core/transcription.py` (compute_type locked to `"int8"`), `README.md`, `.claude/skills/docker-topology/SKILL.md` |

---

## SPEC-AMEND-002: Capture Container Ubuntu Upgrade

| Field | Value |
|---|---|
| **Spec section** | §9.1 — Container Specifications |
| **Original text** | Capture Container image: `ubuntu:22.04` |
| **New behavior** | Capture Container image: `ubuntu:24.04` |
| **Rationale** | scrcpy v3.1+ prebuilt Linux x86_64 binaries require GLIBC 2.38+, which is not available in Ubuntu 22.04 (ships GLIBC 2.35). Ubuntu 24.04 ships GLIBC 2.39. |
| **Affected files** | `services/stream_ingest/Dockerfile`, `README.md`, `.claude/skills/docker-topology/SKILL.md` |

---

## SPEC-AMEND-003: Orchestrator as Separate Container

| Field | Value |
|---|---|
| **Spec section** | §9.1 — Container Specifications; §4.C — Orchestration & Synchronization |
| **Original text** | 5 containers (redis, postgres, stream_scrcpy, worker, api). Module C runs inside the worker process. |
| **New behavior** | 6 containers. Orchestrator runs as a separate container using the same image as the worker but with a different CMD (`python3.11 -m services.worker.run_orchestrator`). |
| **Rationale** | The orchestrator is the PRODUCER of Celery tasks (reads IPC pipes, processes video, assembles 30s segments) while the worker is the CONSUMER (runs ML inference). Separating them prevents the Celery consumer's concurrency model from conflicting with the orchestrator's asyncio event loop and persistent FFmpeg/PyAV subprocesses. |
| **Affected files** | `docker-compose.yml` (orchestrator service), `services/worker/run_orchestrator.py`, `README.md`, `.claude/skills/docker-topology/SKILL.md`, `.claude/skills/module-contracts/SKILL.md` |

---

## SPEC-AMEND-004: scrcpy Dual-Instance Architecture

| Field | Value |
|---|---|
| **Spec section** | §4.A.1 — Hardware & Transport |
| **Original text** | scrcpy v2.x. Single instance. Audio piped via `dd` writing to fd 3. |
| **New behavior** | scrcpy v3.3.4. Dual-instance architecture: audio instance on port range 27100:27199, video instance on port range 27200:27299. Direct `--record` to named pipe replaces `dd`/fd3 for the video path. Audio path retains `exec 3<>` pipe shield for resilience across scrcpy restarts. 4-second staggered startup to prevent ADB server push collisions. |
| **Rationale** | scrcpy v3.x `--record` writes directly to named pipes, eliminating the need for `dd` as an intermediary for video. Dual instances avoid muxing issues and allow independent restart on per-stream failure. The video pipe is intentionally unshielded so that PyAV crash triggers a scrcpy restart with a fresh MKV header. |
| **Affected files** | `services/stream_ingest/entrypoint.sh`, `.claude/skills/ipc-pipeline/SKILL.md`, `.claude/skills/module-contracts/SKILL.md`, `README.md` |

---

## SPEC-AMEND-005: Audio Chunk Size for Video Frame Alignment

| Field | Value |
|---|---|
| **Spec section** | §4.C.2 — Audio Resampling and Buffering |
| **Original text** | Audio read in 1-second chunks (32,000 bytes at 16 kHz s16le mono). |
| **New behavior** | Audio read in 1/30-second chunks (~1,067 bytes) to match the 30 FPS video frame rate for temporal alignment. Computed as `int(bytes_per_second / 30)` where `bytes_per_second = 16000 * 2 = 32000`. |
| **Rationale** | With video capture added to the orchestrator, audio and video processing must be temporally aligned. Reading audio in frame-sized chunks allows the orchestrator to interleave audio accumulation with video frame extraction at the same cadence. |
| **Affected files** | `services/worker/pipeline/orchestrator.py` (run loop, line ~569) |

---

## SPEC-AMEND-006: TranscriptionEngine compute_type Locked to INT8

| Field | Value |
|---|---|
| **Spec section** | §4.D.1 — Speech Transcription |
| **Original text** | `compute_type` configurable per deployment. |
| **New behavior** | `compute_type` hardcoded to `"int8"` as a class-level constant (`_COMPUTE_TYPE`). Cannot be overridden at instantiation. |
| **Rationale** | Downstream effect of SPEC-AMEND-001. INT8 quantization uses dp4a vectorization which is available on Pascal (SM 6.1). FP16 is NOT available on GTX 1080 Ti — allowing overrides would cause silent fallback to CPU or CUDA errors. Locking the value prevents misconfiguration. |
| **Affected files** | `packages/ml_core/transcription.py` (class constant `_COMPUTE_TYPE`, lines ~22-25) |

---

## SPEC-AMEND-007: Physiology Extension via API Ingress and Orchestrator Context

| Field | Value |
|---|---|
| **Spec section** | §4.B.2 — Physiological Ingestion Adapter; §4.C.4 — Physiological State Buffer; §4.E.2 — Physiology Persistence; §7C — Co-Modulation Analytics |
| **Original text** | `no physiological ingestion` |
| **New behavior** | The API Server accepts authenticated Oura webhook deliveries, normalizes them into Physiological Sample Event payloads, and enqueues them to Redis list `physio:events`. The orchestrator drains Redis into a per-`subject_role` Physiological State Buffer, computes `freshness_s` / `is_stale`, and injects Physiological Context into segment payloads. Module E persists per-segment physiology snapshots to `physiology_log` and rolling Co-Modulation Index results to `comodulation_log`. |
| **Rationale** | v3.1 extends the multimodal pipeline with wearable-derived physiology while preserving API/worker image separation, Redis-mediated inter-container transport, and the rule that raw webhook payloads remain transient rather than persistent analytical storage. The Redis queue keeps API ingress decoupled from orchestrator timing and avoids introducing direct container-to-container coupling for physiology delivery. |
| **Affected files** | `services/api/routes/physiology.py`, `packages/schemas/physiology.py`, `services/worker/pipeline/orchestrator.py`, `services/worker/pipeline/analytics.py`, `services/api/db/schema.py`, `data/sql/03-physiology.sql`, `CLAUDE.md`, `.claude/skills/module-contracts/SKILL.md`, `.claude/skills/docker-topology/SKILL.md`, `README.md`, `docs/SPEC_REFERENCE.md` |

---

## SPEC-AMEND-008: Operator Dashboard Framework — Streamlit Retired, Dual PySide6 Surfaces

| Field | Value |
|---|---|
| **Spec section** | §4.E.1 — Execution Details; §10.2 — Dependencies |
| **Original text** | "Operational metrics and experiment state are visualized through a `Streamlit` dashboard." Dependencies list includes `Streamlit`. |
| **New behavior** | The Streamlit dashboard is retired. Operator visualization is delivered through two PySide6 surfaces on the operator host, not in a container: (a) `scripts/debug_studio.py` — developer-facing debug/diagnostic GUI retained as a single-file tool for live pipeline introspection (video, landmarks, AU12 normalization, analytics poller); (b) `services/operator_console/` — production-grade operator dashboard split into clean modules (`api_client.py`, `workers.py`, `state.py`, `polling.py`, `viewmodels/`, `views/`, `widgets/`, `table_models/`) for session monitoring, experiment readback, physiology/co-modulation views, and operator actions (stimulus injection today; future device configuration and external-telemetry OAuth flows). The console surfaces are: **Overview** (active session + experiment + physiology + health rollup + latest encounter + alert attention queue), **Live Session** (per-segment encounter table + §7B reward-explanation detail pane exposing `p90_intensity`, `semantic_gate`, `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`; hosts the stimulus action rail), **Experiments** (Thompson Sampling posterior α/β + evaluation variance + selection count per arm; never implies `semantic_confidence` moves the reward), **Physiology** (operator/streamer RMSSD + heart rate + §4.C.4 four-state freshness: fresh/stale/absent/no-rmssd; §7C Co-Modulation Index with null-valid rendered as an INFO outcome carrying `null_reason`, not as an error), **Health** (§12 subsystem rollup keeping degraded/recovering/error visually distinct; `operator_action_hint` surfaced on the error summary card), and **Sessions** (history; double-click emits `session_selected(UUID)` through the shell's navigation handler). The stimulus action-rail lifecycle — `IDLE → SUBMITTING → ACCEPTED → MEASURING → COMPLETED` with `FAILED` as a terminal error — is driven by `StimulusUiContext` carried through the store; the authoritative `stimulus_time` is the orchestrator's readback (§4.C), never the click wall-clock, and the submit button disables during `SUBMITTING` to guard against double-fires beyond the `client_action_id` idempotency key. Reads are polled on per-surface cadences configured by `OperatorConsoleConfig`; the coordinator scopes jobs by active route so hidden pages do not poll. The `streamlit` pin is removed from `requirements/worker.txt`; `PySide6` and `pydantic` are pinned on the operator host in `requirements/cli.txt`. All data still flows through the FastAPI server's `/api/v1/operator/*` aggregate routes — the operator host does not add direct Postgres or Redis coupling. Packaging is driven by `build/operator_console.spec` (PyInstaller one-dir) with the ML-worker stack explicitly excluded. |
| **Rationale** | Streamlit's web-based rerun-on-interaction model does not fit operator workflows that need (1) live sub-second video and metric readback, (2) device-level operator actions (USB, external-wearable OAuth, stimulus injection), and (3) long-running stateful sessions without tab-refresh semantics. A native PySide6 application gives the operator a single cohesive surface while keeping developer debugging tooling separate. Splitting the production console into modules (API client, Qt workers, views, widgets) is explicitly to avoid the monolithic growth pattern seen in `scripts/debug_studio.py` — the debug tool is allowed to stay a single file because it is developer-facing, but the operator-facing console is maintained as a multi-module package so it can be extended safely over time. |
| **Affected files** | `services/operator_console/**`, `services/api/routes/operator.py`, `services/api/services/operator_read_service.py`, `services/api/services/operator_action_service.py`, `services/api/repos/operator_queries.py`, `packages/schemas/operator_console.py`, `scripts/debug_studio.py`, `requirements/cli.txt`, `requirements/worker.txt` (streamlit removed), `build/operator_console.spec`, `docs/SPEC_REFERENCE.md`, `.claude/skills/module-contracts/SKILL.md`, `README.md` |