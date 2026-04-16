# LSIE-MLF

**Live Stream Inference Engine — Machine Learning Framework**

LSIE-MLF is a containerized monorepo for real-time multimodal inference during live-stream sessions. It combines tethered mobile audio/video capture with external telemetry, synchronizes those inputs into fixed-duration segments, and runs ML analysis for transcription, facial action units, acoustic features, semantic evaluation, and downstream analytics.

The repository is organized around a clear runtime split:

- **`api`** handles external ingress and lightweight application logic
- **`orchestrator`** assembles synchronized inference segments
- **`worker`** executes compute-heavy ML tasks and analytics
- **shared packages** provide schemas and reusable ML utilities

---

## Architecture at a Glance

### Runtime Topology

| Container | Responsibility | Image | Port |
|---|---|---|---|
| `api` | REST endpoints and webhook ingress | `python:3.11-slim` | `8000` |
| `worker` | Celery task consumers for ML inference and analytics | `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` | — |
| `orchestrator` | Segment assembly, synchronization, dispatch | same image as `worker` | — |
| `redis` | Message broker / queue | `redis:7-alpine` | internal |
| `postgres` | Persistent analytical store | `postgres:16-alpine` | internal |
| `stream_scrcpy` | USB media capture container | `ubuntu:24.04` + scrcpy | — |

### Functional Modules

| Module | Primary responsibility |
|---|---|
| **A — Hardware & Transport** | Capture raw audio/video from tethered mobile hardware |
| **B — Ground Truth Ingestion** | Accept external event/telemetry inputs |
| **C — Orchestration & Synchronization** | Align timestamps, assemble segments, attach context |
| **D — Multimodal ML Processing** | Run transcription, facial, acoustic, and semantic inference |
| **E — Experimentation & Analytics** | Persist metrics, run analytics, manage experimentation state |
| **F — Context Enrichment** | Run asynchronous metadata enrichment workflows |

### High-Level Data Flow

```text
Android device
  -> stream_scrcpy
    -> orchestrator
      -> worker
        -> postgres

External telemetry / webhook ingress
  -> api
    -> redis
      -> orchestrator
        -> worker
          -> postgres
```

---

## Repository Layout

```text
/
├── docker-compose.yml              # Root service topology and shared runtime wiring
├── services/
│   ├── api/                        # API Server: routes, ingress, app wiring
│   ├── worker/                     # Celery workers, ML execution, analytics
│   │   └── pipeline/               # Orchestration + analytics pipeline code
│   └── stream_ingest/              # Capture container entrypoints and device ingest
├── packages/
│   ├── ml_core/                    # Shared ML utilities and math
│   └── schemas/                    # Pydantic models and schema contracts
├── data/
│   ├── raw/                        # Transient/debug media buffers
│   ├── interim/                    # Intermediate processing artifacts
│   ├── processed/                  # Structured outputs prior to ingestion
│   └── sql/                        # PostgreSQL schema and seed data
└── requirements/
    ├── base.txt                    # Shared dependencies
    ├── api.txt                     # API-only dependencies
    └── worker.txt                  # Worker/orchestrator + heavy ML dependencies
```

### Build Notes

- All Dockerfiles under `services/` build from the **monorepo root**
- Shared code in `packages/` is available to all services
- Heavy ML dependencies belong in **worker-side requirements**, not the API image

---

## Quick Start

### 1) Configure environment

```bash
cp .env.example .env
# Edit .env with the required credentials and runtime settings
```

### 2) Prepare the host

Recommended local prerequisites:

- Docker Engine
- Docker Compose v2
- NVIDIA Container Toolkit (for GPU-backed inference)
- ADB / Android device connectivity if using live USB capture

### 3) Start the stack

```bash
docker compose up --build
```

Once the stack is running, the API server is available at:

```text
http://localhost:8000
```

### 4) Stop the stack

```bash
docker compose down
```

To remove volumes as well:

```bash
docker compose down -v
```

---

## Where to Make Changes

| If you need to change... | Start here |
|---|---|
| API routes, request handling, webhook ingress | `services/api/` |
| Worker task execution or ML runtime behavior | `services/worker/` |
| Segment assembly, synchronization, analytics pipeline | `services/worker/pipeline/` |
| Shared inference helpers or math | `packages/ml_core/` |
| Schemas and data contracts | `packages/schemas/` |
| Database tables / initialization SQL | `data/sql/` |
| Dependency placement | `requirements/base.txt`, `requirements/api.txt`, `requirements/worker.txt` |

---

## Development Notes

### Dependency Placement

Use the `requirements/` split intentionally:

- put **shared** packages in `requirements/base.txt`
- put **API-only** packages in `requirements/api.txt`
- put **worker/orchestrator / ML-heavy** packages in `requirements/worker.txt`

This keeps the API image lightweight and improves Docker layer caching.

### CI Expectations

Before opening a PR, run the same classes of checks used in CI:

- lint / format
- type checking
- unit tests

At a minimum, changes touching worker or analytics code should be validated against the full worker test path.

---

## Data Handling

Raw media and inbound telemetry should be treated as **processing inputs**, not long-term application records. Persistent storage is intended for structured analytical outputs, experiment state, and derived metrics.

Keep README-level guidance brief and put detailed governance, retention, and security rules in the technical specification and implementation docs.

---

## Specification

LSIE-MLF is implemented against the specification `docs/tech-spec-v3.1.pdf`.

This README is intentionally operational. It explains how the repository is organized, how to run it locally, and where to make changes. Detailed contracts, mathematical formulas, failure handling, and version history belong in the specification and amendment log.

If this README and the specification differ, the specification is authoritative.

---

## License

Confidential. All rights reserved.