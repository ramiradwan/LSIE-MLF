# LSIE-MLF

**Live Stream Inference Engine — Machine Learning Framework**

Real-time multimodal inference system that ingests raw audio-visual streams from tethered mobile hardware and executes concurrent ML evaluations including speech transcription, facial action unit analysis, acoustic feature extraction, and semantic evaluation.

---

## System Overview

LSIE-MLF is a containerized monorepo comprising five Docker services orchestrated via Docker Compose. The system captures uncompressed PCM audio and raw H.264 video from a USB-connected Android device, transports media through POSIX IPC pipes for zero-copy streaming, and routes GPU-bound inference tasks through a Celery distributed task queue.

The architecture enforces a schema-first data exchange paradigm using JSON Schema Draft 07 contracts validated by Pydantic, a modular boundary constraint system separating I/O-bound API logic from compute-bound ML inference, and absolute dependency determinism with pinned package versions.

All raw biometric data (facial landmarks, speech audio) is governed by the Ephemeral Vault policy: AES-256-GCM encrypted transient storage with mandatory 24-hour secure deletion. No raw personally identifiable media is persisted beyond the processing pipeline.

## Architecture

### Container Topology

| Container | Role | Image | Port |
|---|---|---|---|
| `api` | FastAPI REST endpoints | `python:3.11-slim` | 8000 |
| `worker` | Celery ML inference tasks | `nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04` | — |
| `redis` | Message Broker (Celery) | `redis:7-alpine` | 6379 (internal) |
| `postgres` | Persistent Store (metrics, experiments) | `postgres:16-alpine` | 5432 (internal) |
| `stream_scrcpy` | Capture Container (USB audio/video) | `ubuntu:22.04` + scrcpy v2.x | — |

### Core Modules

**Module A — Hardware & Transport.** Extracts uncompressed audio and video from USB-tethered Android devices via scrcpy v2.x. Raw PCM audio (s16le, 48 kHz mono) is streamed through a POSIX named pipe at `/tmp/ipc/audio_stream.raw` using a shared Docker volume.

**Module B — Ground Truth Ingestion.** Ingests real-time chat and interaction events from TikTok Webcast via the TikTokLive Python connector. Connection authentication uses EulerStream-signed parameters. Composite events are staged for temporal correlation with inference outputs.

**Module C — Orchestration & Synchronization.** Aligns timestamps across device hardware clocks, IPC media streams, and external WebSocket events. Performs temporal drift correction via ADB epoch polling every 30 seconds. Resamples audio from 48 kHz to 16 kHz using an FFmpeg subprocess pipeline.

**Module D — Multimodal ML Processing.** Executes speech transcription (faster-whisper, INT8 quantized on CUDA 12), facial landmark inference (MediaPipe Face Mesh, 478-vertex 3D mesh), AU12 intensity scoring (IOD-normalized geometric ratio), acoustic analysis (Praat via parselmouth), and semantic evaluation (Azure OpenAI, deterministic mode).

**Module E — Experimentation & Analytics.** Aggregates inference metrics into time-series datasets. Implements Thompson Sampling for adaptive experimentation. Persists results to PostgreSQL and exposes operational dashboards via Streamlit.

**Module F — Context Enrichment.** Asynchronous web scraping via Celery workers using patchright (patched Chromium) for external metadata enrichment.

### Data Flow Pipeline

```
Mobile Device (USB)
  → Capture Container (scrcpy raw PCM + H.264)
    → IPC Pipe (/tmp/ipc/audio_stream.raw)
      → ML Worker (FFmpeg resample 48→16 kHz)
        → Module D (Whisper + MediaPipe + Praat + Azure OpenAI)
          → Celery Task Queue (Redis)
            → Module E (pandas + SciPy + Thompson Sampling)
              → Persistent Store (PostgreSQL)
```

## Directory Structure

```
/
├── docker-compose.yml              # Root orchestrator: containers, networks, GPU, volumes
├── services/
│   ├── api/                        # API Server: FastAPI, Uvicorn, REST routing
│   ├── worker/                     # ML Worker: Celery consumers, inference pipeline
│   └── stream_ingest/              # Capture Container: ADB, scrcpy, IPC pipe lifecycle
├── packages/
│   ├── ml_core/                    # Shared ML utilities: AU12 math, landmark extraction, ASR wrappers
│   └── schemas/                    # Pydantic models, JSON Schema Draft 07 definitions
├── data/
│   ├── raw/                        # Transient debug media buffers (tmpfs in production)
│   ├── interim/                    # Intermediate processing states (downsampled audio, cropped frames)
│   └── processed/                  # Validated inference outputs, structured JSON payloads
└── requirements/                   # Pinned dependencies: base, api, ml
```

All Dockerfiles in `/services` set their build context to the monorepo root so that shared `/packages` directories are accessible during build.

## System Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.11.x | All containers. 3.12+ untested with CTranslate2. |
| NVIDIA CUDA | ≥ 12.0 | Required for faster-whisper / CTranslate2 |
| cuDNN | ≥ 9.0 | Bundled in worker image |
| NVIDIA Container Toolkit | latest | Docker GPU passthrough |
| GPU VRAM | ≥ 6 GB | faster-whisper large-v3 INT8 + MediaPipe |
| Docker Compose | v2.x | Required for `deploy.resources` GPU reservation |
| scrcpy | v2.x | Raw audio/video extraction |
| ADB | latest | Android device communication |

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| faster-whisper | 1.2.1 | GPU-accelerated speech transcription |
| mediapipe | 0.10.x | Face Mesh 478-landmark inference |
| parselmouth | 0.4.4 | Praat acoustic feature extraction |
| spacy | 3.7.x | NLP preprocessing |
| celery | 5.4.x | Distributed task queue |
| fastapi | 0.110.x | REST API framework |
| pydantic | 2.x | Schema validation |
| psycopg2-binary | 2.9.x | PostgreSQL driver |
| pandas | 2.2.x | Time-series processing |
| pycryptodome | ≥ 3.20.0 | AES-256-GCM encryption (Ephemeral Vault) |
| patchright | 1.58.2 | Stealth browser automation |
| TikTokLive | 5.0.x | TikTok WebSocket ingestion |

## Data Governance

The system enforces three data classification tiers:

**Transient Data.** Raw PCM audio and video frames exist only in volatile memory (`/tmp/ipc/` tmpfs and kernel pipe buffers) during active processing. No disk persistence.

**Debug Storage.** Encrypted diagnostic media buffers written to `/data/raw/` and `/data/interim/` only during algorithmic anomalies. AES-256-GCM encrypted with per-session keys held exclusively in process memory. Mandatory secure deletion (`shred -vfz -n 3`) every 24 hours.

**Permanent Analytical Storage.** Anonymized metrics only (AU12 scores, acoustic metrics, semantic evaluation results) persisted to PostgreSQL. Raw facial images, voiceprints, and personally identifiable biometric media are never stored in the Persistent Store.

## Quick Start

```bash
# Ensure NVIDIA Container Toolkit is installed and an Android device is connected via USB
docker compose up --build
```

The API Server will be available at `http://localhost:8000`.

## Specification Reference

This implementation is governed by the **LSIE-MLF Master Technical Specification v2.0** (Production Hardened, 15-Rule Deterministic Audit). All code decisions, module boundaries, interface contracts, error handling, and dependency versions are strictly pinpointed to the specification. See `docs/tech-spec-v2.0.pdf` for the full authoritative document.

## License

Confidential. All rights reserved.
