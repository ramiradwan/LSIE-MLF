# LSIE-MLF Specification Amendment Registry

This document tracks all known deviations from the **Master Technical Specification v2.0** (`docs/tech-spec-v2.0.pdf`). The spec is a signed PDF that cannot be edited. This registry is the single authoritative source for "what the spec says" vs "what the implementation does."

Each amendment was made for a documented technical reason. The review agent should treat these as accepted deviations, not violations.

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
