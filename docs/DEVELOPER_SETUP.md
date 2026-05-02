# LSIE-MLF v4.0 Developer Setup

Setup guide for developers contributing to the v4.0 desktop pivot.
Tier 1 platform is Windows 11 x64 with NVIDIA Turing (SM 7.5+) for the
production GPU path. Pascal (GTX 10-series) hosts are supported as
**developer-only** machines via the `LSIE_DEV_FORCE_CPU_SPEECH` escape
hatch documented below.

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

The v4.0 production speech path requires Turing (SM 7.5+) on
CUDA 12.x with cuDNN 9. The exhaustive contract lives in
`docs/V4_SPEC_DRAFTS.md` §11.x; the developer-facing rules are:

| Compute capability | Tier             | Speech device | Cross-encoder | Required env |
|---|---|---|---|---|
| 7.5+ (Turing / Ampere / Ada / Hopper) | Production | CUDA | CUDA | none |
| 6.1 (Pascal — GTX 10-series)          | Developer  | **CPU** | CUDA | **`LSIE_DEV_FORCE_CPU_SPEECH=1`** |
| no GPU                                | CI / VM    | CPU  | CPU  | none |

### 3.1 Pascal developer override

If you are on a Pascal host (e.g. GTX 1080 Ti), set the override
before launching the desktop app or running tests:

```powershell
$env:LSIE_DEV_FORCE_CPU_SPEECH = "1"
```

The runtime contract:

- `packages.ml_core.transcription.resolve_speech_device()` returns
  `"cpu"` when the env is set, regardless of `nvidia-smi` output.
- The cross-encoder semantic scorer remains on the GPU because PyTorch
  alone — without CTranslate2 loaded in the same process — does not
  trigger the cuDNN 8 vs cuDNN 9 collision.
- WS2 P3 introduces a **production preflight gate** that hard-rejects
  SM<7.5 hosts unless a `.dev_machine` marker file is present at the
  platform-standard local-app-data path. To declare your machine a
  developer host, create the marker once:

  ```powershell
  ni -ItemType File "$env:LOCALAPPDATA\LSIE-MLF\.dev_machine" -Force
  ```

  The marker means "I accept the documented dev-mode constraints";
  the production launcher fails closed on Pascal without it.

### 3.2 Expected speech-path latency

faster-whisper INT8 on the developer's CPU transcribes a 30-second
segment in roughly two to five seconds depending on host CPU. The
production GPU path runs in well under one second on T4 / RTX 3060
and above. Latency parity between the two paths is **not** the
contract; output text equality on the Gate 0 corpus is — see
`tests/v4_gate0/`.

## 4. Running the test suite

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

# Real-Credential-Manager round-trip (WS4 P4 secrets):
$env:LSIE_INTEGRATION_KEYRING = "1"
.venv\Scripts\python.exe -m pytest tests/unit/desktop_app/privacy/test_secrets.py

# mypy strict + ruff:
.venv\Scripts\python.exe -m mypy services/ packages/
.venv\Scripts\python.exe -m ruff check services/ packages/ tests/
```

## 5. References

- **Pivot handoff:** `docs/artifacts/v4_pivot_handoff_2026-05-01.md`
- **Implementation plan:** `docs/artifacts/LSIE-MLF_v4_0_Implementation_Plans.md`
- **Spec accumulator:** `docs/V4_SPEC_DRAFTS.md`
- **Spec amendments registry:** `docs/SPEC_AMENDMENTS.md`
- **Project rules:** `CLAUDE.md`
