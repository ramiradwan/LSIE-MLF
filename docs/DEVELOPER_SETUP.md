# LSIE-MLF v4.0 Developer Setup

Setup guide for developers contributing to the Windows-native desktop runtime.
Tier 1 platform is Windows 11 x64 with NVIDIA Turing (SM 7.5+) for the
production GPU path. Pascal (GTX 10-series) hosts are supported only in
developer mode, where speech runs on the CPU.

## 1. Prerequisites

- **Windows 11 x64** (Tier 1; Linux WSL2 supported for parity smoke
  tests, macOS deferred to v4.1).
- **Python 3.11.x** — Python 3.12+ is hard-rejected by the preflight
  gate because `mediapipe == 0.10.9` ships no wheel for 3.12+.
- **`uv`** ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)) — package manager. The shipped
  `pyproject.toml` + `uv.lock` are authoritative; do not regenerate
  the lockfile without spec justification.
- **NVIDIA driver** with `nvidia-smi` on `PATH`. Verify with
  `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.
- **scrcpy 3.3+, ADB, FFmpeg** on `PATH` (or via the
  `LSIE_SCRCPY_PATH` / `LSIE_ADB_PATH` / `LSIE_FFMPEG_PATH` env
  overrides resolved by `services.desktop_app.os_adapter`).

Redis, Celery workers, Docker Compose services, and PostgreSQL are not
prerequisites for launching the v4 desktop runtime with
`python -m services.desktop_app`; local transport uses IPC queues/shared
memory and local state uses SQLite WAL. The active v4 desktop tree does not
track Docker Compose or Dockerfile manifests.

Use `python -m services.desktop_app` for the full GUI desktop app. Use
`python -m services.desktop_app --operator-api` for CLI automation and E2E
checks that need the loopback API/control graph without opening the PySide
Operator Console.

## 2. Environment hydration

```powershell
# from the repo root
uv sync --frozen --extra ml_backend --group dev
```

`uv sync --frozen --extra ml_backend` hydrates the virtual environment
to the exact versions checked into `uv.lock`, including the heavy GPU
backend wheels that the first-run launcher hydrates progressively for
end users. Add `--group dev` for pytest, mypy, and ruff.

## 3. GPU / compute-capability matrix

The current contract lives in the single signed `docs/tech-spec-v*.pdf` file. Use `scripts/spec_ref_check.py` to resolve runtime topology (§9), system requirements (§10.1), pinned packages (§10.2), and the embedded content payload.
The developer-facing rules are:

| Compute capability | Tier | Speech device | Cross-encoder | Required developer declaration |
|---|---|---|---|---|
| 7.5+ (Turing / Ampere / Ada / Hopper) | Production | CUDA | CUDA | none |
| 6.1 (Pascal — GTX 10-series) | Developer mode only | **CPU** | CUDA | **`.dev_machine` marker or `LSIE_DEV_FORCE_CPU_SPEECH=1`** |
| no GPU | Developer mode only | CPU | CPU | **`.dev_machine` marker or `LSIE_DEV_FORCE_CPU_SPEECH=1`** |

### 3.1 Developer-mode override

If you are on a Pascal host (for example GTX 1080 Ti), you can declare the
machine as a developer host in either of two ways before launching the desktop
app or running tests:

```powershell
$env:LSIE_DEV_FORCE_CPU_SPEECH = "1"
```

or create the marker once:

```powershell
ni -ItemType File "$env:LOCALAPPDATA\LSIE-MLF\.dev_machine" -Force
```

The runtime contract:

- Production uses the GPU speech path only on Turing-or-newer hardware.
- Pascal developer mode routes speech to the CPU while preserving other GPU
  paths such as PyTorch semantic scoring.
- No-GPU developer mode routes every ML path to the CPU.
- Without a developer declaration, the launcher fails closed on Pascal and
  no-GPU hosts.

### 3.2 Expected speech-path latency

faster-whisper INT8 on the developer's CPU transcribes a 30-second
segment in roughly two to five seconds depending on host CPU. The
production GPU path runs in well under one second on T4 / RTX 3060
and above. Latency parity between the two paths is **not** the
contract; output text equality on the Gate 0 corpus is — see
`tests/v4_gate0/`.

## 4. Running locally

Full GUI app:

```powershell
uv run python -m services.desktop_app
```

Operator API runtime for CLI use:

```powershell
uv run python -m services.desktop_app --operator-api
```

Basic CLI flow while either mode is running. The CLI defaults to `http://127.0.0.1:8000` or `LSIE_API_URL` if set:

```powershell
uv run python -m scripts status
uv run python -m scripts health
uv run python -m scripts sessions start android://device --experiment greeting_line_v1
uv run python -m scripts stimulus submit <session-id> --note "test stimulus"
uv run python -m scripts live-session readback <session-id>
uv run python -m scripts sessions end <session-id>
```

If the preferred port is occupied, set `LSIE_API_URL` once or use the logged loopback URL with `--api-url <url>`.

## 5. Running the test suite

```powershell
# Full v4 desktop surface (123+ tests, fast):
.venv\Scripts\python.exe -m pytest `
  tests/v4_gate0/ tests/unit/desktop_app/ tests/integration/desktop_app/ `
  tests/unit/worker/pipeline/test_orchestrator.py

# Whole repo:
.venv\Scripts\python.exe -m pytest tests/ -x -q

# Real-device smoke (Pixel on USB):
$env:LSIE_INTEGRATION_DEVICE = "1"
.venv\Scripts\python.exe -m pytest tests/integration/desktop_app/test_capture_supervisor_smoke.py

# Real-Credential-Manager round-trip:
$env:LSIE_INTEGRATION_KEYRING = "1"
.venv\Scripts\python.exe -m pytest tests/unit/desktop_app/privacy/test_secrets.py

# mypy strict + ruff:
.venv\Scripts\python.exe -m mypy services/ packages/
.venv\Scripts\python.exe -m ruff check services/ packages/ tests/
```

## 6. Local validation gates

Standard desktop validation does not require Docker Compose, Redis, or
PostgreSQL. For desktop-runtime changes, prefer the targeted desktop surface
above plus ruff and mypy.

There is no `docker compose config --quiet` gate for active v4 desktop
validation because no compose/Dockerfile manifests are tracked. Historical/spec
Docker references describe retired or external server/container context, not a
local validation prerequisite.

## 7. References

- **Signed spec:** `docs/tech-spec-v*.pdf`
- **Spec reference tooling:** `scripts/spec_ref_check.py`
- **Historical pivot handoff:** `docs/artifacts/v4_pivot_handoff_2026-05-01.md`
- **Historical implementation plan:** `docs/artifacts/LSIE-MLF_v4_0_Implementation_Plans.md`
- **Project rules:** `CLAUDE.md`
