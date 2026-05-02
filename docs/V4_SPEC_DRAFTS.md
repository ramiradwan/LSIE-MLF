# LSIE-MLF v4.0 Spec Section Drafts

This file accumulates spec-section text written during the v4.0 desktop
pivot. The signed PDF (`docs/tech-spec-v4.0.pdf`) was committed before
all workstream-driven content updates landed; the text below is what
the next signed PDF revision must absorb. Each entry cites the
workstream / phase that authored it so the spec author can audit
provenance during the paste-merge.

The pivot handoff at `docs/artifacts/v4_pivot_handoff_2026-05-01.md`
§5.1 row "v4.0 spec content updates" is the umbrella tracking item.

---

## §11.x — Hardware Matrix and Developer Environment Override

**Author.** WS2 P1 (`feature/v4-desktop`).
**Supersedes.** SPEC-AMEND-001 / SPEC-AMEND-002 in `docs/SPEC_AMENDMENTS.md`
— both are recorded as "Historical — folded into v4.0 base text" once
this section lands.

### §11.x.1 Production hardware floor

The v4.0 production GPU floor is NVIDIA **Turing (SM 7.5+)** on **CUDA
12.x with cuDNN 9**, paired with **`ctranslate2 >= 4.5.0`** and **`torch
== 2.4.x`**. The v3.4 §9.1 floor of Pascal (SM 6.1) on cuDNN 8 is
retired. Vision retains `mediapipe == 0.10.9` with the legacy
`mp.solutions.face_mesh` API at 478 landmarks; the legacy API is the
only one that produces the landmark indexing the §6.1
`InferenceHandoffPayload._au12_series` field is anchored to.

`faster-whisper`'s `compute_type` is locked to `"int8"`. On Turing /
Ampere / Ada the INT8 path uses DP4A and the INT8 Tensor Cores; on the
Pascal developer escape hatch (§11.x.2) it is the right CPU default for
faster-whisper as well. The lock is not operator-configurable —
allowing overrides would silently fall back to FP16 on Pascal (which
lacks the kernel) and mask a misconfiguration of the production GPU
floor as a software issue.

### §11.x.2 Pascal developer override

The lead developer's workstation is a GTX 1080 Ti (Pascal, SM 6.1).
Pascal cannot host the v4.0 production speech path because the
`torch 2.4.x` cuDNN-9 link and the `ctranslate2 >= 4.5.0` cuDNN-9 link
both require Turing+ binaries. The escape hatch is the
`LSIE_DEV_FORCE_CPU_SPEECH=1` environment variable, which routes
`TranscriptionEngine` to CPU. The cross-encoder semantic scorer
remains on the Pascal GPU because PyTorch alone does not trigger the
cuDNN collision once CTranslate2 is moved off-GPU.

The escape hatch is a developer-only contract. The production
preflight gate (§11.x.3) hard-rejects Pascal in any host that does not
declare the dev-mode marker, so a misconfigured production install
cannot silently fall through to the override.

### §11.x.3 Preflight gate

Before any process spawn the launcher runs
`services/desktop_launcher/preflight.py`, which:

1. Parses `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
2. In **production mode** (no `.dev_machine` marker file at the
   platform-standard developer path), rejects with
   `HardwareUnsupportedError` if the detected compute capability is
   less than 7.5.
3. In **developer mode** (the marker is present, OR
   `LSIE_DEV_FORCE_CPU_SPEECH=1` is already set), emits the soft
   warning `PASCAL_DEV_MODE_REQUIRED` and proceeds, asserting the
   override into the runtime config.
4. Rejects Python `>= 3.12` because `mediapipe == 0.10.9` ships no
   wheel for that interpreter line.

The marker is keyed off the platform-standard local-app-data path
resolved by `services.desktop_app.os_adapter.resolve_state_dir()` so
the same code reads the marker on Windows (`%LOCALAPPDATA%\LSIE-MLF`),
Linux (`$XDG_DATA_HOME/lsie-mlf`), and the deferred macOS Tier 2.

### §11.x.4 Production GPU CI runner

Because no team member can validate the production CUDA speech path
locally during Sprint 1, the Gate 0 corpus (`tests/fixtures/v4_gate0/`)
must run against real Turing-or-newer hardware on every PR that touches
ML code. The `.github/workflows/gpu_replay_parity.yml` workflow runs
on a self-hosted GitHub Actions runner backed by an AWS
**g4dn.xlarge** instance (NVIDIA T4, SM 7.5). The workflow triggers on
PRs that touch `packages/ml_core/`, `services/desktop_app/processes/
gpu_ml_worker.py`, or `uv.lock`, drives the production CUDA path
through the corpus, and emits a parity diff against the golden
fixtures. A non-zero diff blocks merge.

### §11.x.5 Tier 2 design intent (deferred to v4.1)

macOS (Apple Silicon arm64 and Intel x64) is deferred to v4.1. The
following design intent is preserved so the Tier 2 work resumes
without re-derivation:

- **CPU speech path** on Apple Silicon: faster-whisper INT8 on CPU.
  CTranslate2 ≥ 4.5 on macOS does not yet ship a Metal backend; the
  CPU path is the only supported configuration.
- **MPS cross-encoder routing** with the contiguity workaround for
  PyTorch issue #SDPA-non-contiguous and explicit
  `torch.mps.empty_cache()` between segments to bound resident set.
- **App Nap suppression** via the Cocoa `NSProcessInfo
  beginActivityWithOptions:` bridge so the orchestration loop is not
  throttled by the Power Management subsystem.
- **Process-group supervision** for native subprocesses (scrcpy, ADB,
  FFmpeg) via `os.killpg` and `atexit` signal-routing — Windows Job
  Objects have no macOS equivalent; the POSIX path is the canonical
  Tier 2 supervision contract.
- **Keychain backend selection** for `keyring`. The `keyring` library
  picks `keyring.backends.macOS.Keyring` automatically; the v4.0
  desktop's `services/desktop_app/privacy/secrets.py` is the consumer
  surface and works against any well-behaved backend.
- **Hardened Runtime entitlements** plus `codesign` against the
  Developer ID Application certificate. Tracked in
  `build/MACOS_DEFERRED.md` as the v4.1 entry point.

---

<!-- Subsequent workstreams append their drafts below this line. -->
