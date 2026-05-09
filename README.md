# LSIE-MLF

**Live Stream Inference Engine — Machine Learning Framework**

LSIE-MLF is a desktop-first monorepo for real-time multimodal inference during live-stream sessions. It combines tethered mobile audio/video capture with external telemetry, synchronizes those inputs into fixed-duration segments, and runs ML analysis for transcription, facial action units, acoustic features, semantic evaluation, and downstream analytics.

The v4 runtime is organized around a host-side desktop process graph:

- **`services.desktop_app`** is the primary operator/runtime entrypoint
- **full GUI mode** opens the PySide Operator Console
- **operator API runtime** serves the same loopback API/control graph for CLI use without opening the GUI
- **`ui_api_shell`** serves the local desktop UI/API surface
- **`capture_supervisor`** manages physical capture supervision
- **`module_c_orchestrator`** assembles synchronized inference segments
- **`gpu_ml_worker`** executes compute-heavy ML tasks and analytics
- **`analytics_state_worker`** maintains local analytical state
- **`cloud_sync_worker`** drains the desktop cloud-sync outbox
- **shared packages** provide schemas and reusable ML utilities

---

## Architecture at a Glance

### Runtime Topology

| Desktop process | Responsibility |
|---|---|
| `ui_api_shell` | Local desktop UI/API shell |
| `capture_supervisor` | Physical capture supervision |
| `module_c_orchestrator` | Segment assembly, synchronization, dispatch |
| `gpu_ml_worker` | GPU-backed ML inference and analytics |
| `analytics_state_worker` | Local analytical state maintenance |
| `cloud_sync_worker` | Offline cloud-sync outbox draining |

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
  -> capture_supervisor
    -> module_c_orchestrator
      -> gpu_ml_worker
        -> analytics_state_worker

External telemetry / webhook ingress
  -> ui_api_shell / retained API surfaces
    -> analytics_state_worker
      -> module_c_orchestrator
        -> gpu_ml_worker
```

Oura webhook deliveries are treated as change notifications. Retained server/API surfaces may still use Redis-backed hydration queues, but the primary v4 operator path starts from the desktop app entrypoint rather than a Docker Compose stack.

---

## Repository Layout

```text
/
├── services/
│   ├── desktop_app/                # Primary v4 desktop runtime and process graph
│   ├── operator_console/           # Reusable/standalone PySide6 UI-only host
│   ├── api/                        # Retained API Server routes and ingress surfaces
│   ├── cloud_api/                  # Cloud API routes, services, repos, and SQL DDL
│   └── worker/                     # Retained ML execution, orchestration, and analytics code
│       └── pipeline/               # Orchestration + analytics pipeline code
├── packages/
│   ├── ml_core/                    # Shared ML utilities and math
│   └── schemas/                    # Pydantic models and schema contracts
├── services/cloud_api/db/sql/      # Cloud PostgreSQL schema and seed data
├── pyproject.toml                  # Canonical dependency declarations, extras, and tool config
└── uv.lock                        # Frozen dependency resolution for uv sync --frozen
```

### Runtime Notes

- `python -m services.desktop_app` is the primary v4 launch path
- Shared code in `packages/` is available to all services
- Heavy ML dependencies belong in the **`ml_backend` extra**, not the base API/runtime environment
- No Docker Compose or Dockerfile manifests are tracked for the active v4 desktop runtime; historical/spec references are not launch instructions

---

## Quick Start

### 1) Configure environment

```bash
cp .env.example .env
# Edit .env with the required credentials and runtime settings
```

### 2) Prepare the host

Recommended local prerequisites:

- Python 3.11
- uv
- CUDA-capable NVIDIA GPU and current NVIDIA driver for GPU-backed inference
- ADB / Android device connectivity if using live USB capture

### 3) Install dependencies

```bash
uv sync --frozen --extra ml_backend
```

### 4) Launch the app or operator API runtime

Full GUI app:

```bash
uv run python -m services.desktop_app
```

CLI/API-only workflow:

```bash
uv run python -m services.desktop_app --operator-api
```

Both modes run preflight and start the ProcessGraph with capture, orchestration, ML, analytics, and cloud-sync workers. The default command opens the PySide Operator Console; `--operator-api` starts the same loopback API/control surface for CLI use without opening the GUI.

### 5) Use the CLI

With either the full GUI app or operator API runtime running, the CLI defaults to `http://127.0.0.1:8000` or `LSIE_API_URL` if set:

```bash
uv run python -m scripts status
uv run python -m scripts health
uv run python -m scripts sessions start android://device --experiment greeting_line_v1
uv run python -m scripts sessions list
uv run python -m scripts stimulus submit <session-id> --note "test stimulus"
uv run python -m scripts live-session readback <session-id>
uv run python -m scripts sessions end <session-id>
```

If the loopback API selects another port, set `LSIE_API_URL` once for the shell or pass `--api-url <url>` on individual commands. The CLI talks only to the loopback API; it does not read SQLite directly.

### 6) Reusable Operator Console host

`services.operator_console` is a reusable/standalone PySide6 UI-only host. It polls an external API Server's `/api/v1/operator/*` aggregate routes and does **not** start capture, GPU inference, SQLite state, or cloud sync.

Use it only when developing or testing the UI against an already-running external API:

```bash
uv sync --frozen
uv run python -m services.operator_console
```

Environment variables (all optional; sensible defaults apply):

| Variable | Purpose |
|---|---|
| `LSIE_OPERATOR_API_BASE_URL` | External API Server base URL (default `http://localhost:8000`) |
| `LSIE_OPERATOR_API_TIMEOUT_SECONDS` | Per-request timeout, default `5` |
| `LSIE_OPERATOR_ENVIRONMENT_LABEL` | Free-text label shown in the statusline (e.g. `dev`, `staging`) |
| `LSIE_OPERATOR_*_POLL_MS` | Per-surface poll cadences (overview, sessions, health, …) — see `services/operator_console/config.py` for the full list |

The console ships six pages: Overview, Live Session, Experiments, Physiology, Health, and Sessions. Page behavior traces to the spec:

- Live Session's reward explanation uses `p90_intensity`, `semantic_gate`, `gated_reward`, `n_frames_in_window`, and `au12_baseline_pre` (§7B).
- Physiology surfaces `fresh` / `stale` / `absent` / `no-rmssd` as four distinct states (§4.C.4).
- Co-modulation `null` is rendered as a legitimate `null-valid` outcome with its `null_reason`, not as an error (§7C).
- Health distinguishes `degraded` / `recovering` / `error` with operator-action hints on the error summary card (§12).

### Historical Docker/server references

The current tracked tree has no Docker Compose or Dockerfile manifests for the active v4 desktop runtime. Docker, container, Message Broker, and Persistent Store references that remain in spec extracts or archived artifacts describe retained legacy/server architecture or historical context, not the default operator workflow.

---

## Where to Make Changes

| If you need to change... | Start here |
|---|---|
| API routes, request handling, webhook ingress | `services/api/` |
| Worker task execution or ML runtime behavior | `services/worker/` |
| Segment assembly, synchronization, analytics pipeline | `services/worker/pipeline/` |
| Shared inference helpers or math | `packages/ml_core/` |
| Schemas and data contracts | `packages/schemas/` |
| Cloud database tables / initialization SQL | `services/cloud_api/db/sql/` |
| Dependency placement | `pyproject.toml` and `uv.lock` |

---

## Development Notes

### Dependency Placement

Use `pyproject.toml` as the declaration surface and `uv.lock` as the frozen resolution surface:

- put **shared runtime** packages in `[project.dependencies]`
- put **ML-heavy worker/orchestrator** packages in `[project.optional-dependencies].ml_backend`
- refresh `uv.lock` whenever dependency declarations change

This keeps the base API/runtime environment lightweight while preserving a reproducible lockfile for `uv sync --frozen`.

### Local validation gates

For desktop-runtime changes, run targeted desktop validation first:

```bash
uv run pytest tests/v4_gate0/ tests/unit/desktop_app/ tests/integration/desktop_app/ tests/unit/worker/pipeline/test_orchestrator.py
uv run ruff check packages/ services/ tests/
uv run ruff format --check packages/ services/ tests/
uv run mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases
```

The full local check scripts are still available for repository-wide pre-push validation:

```bash
bash scripts/check.sh          # macOS / Linux / Git Bash on Windows
pwsh scripts/check.ps1         # PowerShell on Windows
```

There is no standing Docker Compose gate for the active v4 desktop runtime because no compose/Dockerfile manifests are tracked. Historical/spec Docker references should not be converted into launch or validation instructions.

At a minimum, changes touching worker or analytics code should be validated against the full worker test path.

---

## Data Handling

Raw media and inbound telemetry should be treated as **processing inputs**, not long-term application records. Persistent storage is intended for structured analytical outputs, experiment state, and derived metrics.

Keep README-level guidance brief and put detailed governance, retention, and security rules in the technical specification and implementation docs.

---

## Specification

LSIE-MLF is implemented against the single signed specification PDF committed as `docs/tech-spec-v*.pdf`.

This README is intentionally operational. It explains how the repository is organized, how to run it locally, and where to make changes. Detailed contracts, mathematical formulas, failure handling, and version history belong in the signed specification payload.

If this README and the specification differ, the specification is authoritative.

---

## License

Confidential. All rights reserved.