# LSIE-MLF Specification Reference

This file extracts the key implementation-governing details from the immutable Master Technical Specification **v3.2** (`docs/tech-spec-v3.2.pdf`) together with accepted amendments in `docs/SPEC_AMENDMENTS.md`. The v3.1 release incorporated the physiology/co-modulation extension (SPEC-AMEND-007) into the base spec text. The v3.2 release additionally incorporates the dual-PySide6 operator surface (SPEC-AMEND-008) and the corrected Oura ingestion model (SPEC-AMEND-009 — webhooks treated as change notifications, OAuth2-backed hydration, normalized Physiological Chunk Event transport, Module C rolling-window RMSSD derivation, 1-minute resampling for co-modulation). SPEC-AMEND-001 through SPEC-AMEND-006 remain governing where the v3.2 text has not superseded them. The repository now ships the SPEC-AMEND-009 physiology path: webhook notifications enqueue hydration work to `physio:hydrate`, the API-hosted Oura hydration worker fetches provider resources and emits Physiological Chunk Event records to `physio:events`, the Orchestrator derives scalar `_physiological_context` snapshots from rolling chunk buffers, and Module E persists only scalar physiology analytics. Claude Code loads this on demand via the `@docs/SPEC_REFERENCE.md` import in CLAUDE.md.

For the signed base spec, see `docs/tech-spec-v3.2.pdf`; the extracted JSON index is `docs/content.json`, generated via `python scripts/spec_ref_check.py --from-pdf docs/tech-spec-v3.2.pdf --extract > docs/content.json`. The amendment registry defines the current governing behavior wherever an amendment overrides the base spec.

## Data flow pipeline (§2)

1. Mobile USB → Capture Container (scrcpy raw PCM s16le 48kHz + H.264)
2. Capture Container → Orchestrator Container via IPC Pipe `/tmp/ipc/audio_stream.raw` and `/tmp/ipc/video_stream.mkv`
3. API Server → Redis list `physio:hydrate`: authenticated Oura webhook notification normalized into hydration work
4. API-hosted Oura hydration worker → Redis list `physio:events`: OAuth2-backed provider fetch normalized into Physiological Chunk Event
5. Orchestrator internal: FFmpeg resample 48kHz → 16kHz via `pipe:1`, bounded Redis drain into Physiological State Buffer, rolling-window RMSSD derivation
6. Module B → Module C: Python dicts with uniqueId, event_type, timestamp_utc, payload
7. Module C → Module D: `InferenceHandoffPayload` validated by Pydantic, 30s segments, optional `_physiological_context`
8. Module D → Module E: Celery async task via Redis broker (session_id, segment_id, AU12, transcription, semantic, pitch, jitter, shimmer)
9. Module E → PostgreSQL: parameterized INSERT, DOUBLE PRECISION metrics, TIMESTAMPTZ timestamps, scalar-only physiology analytics tables

## Error handling matrix (§12)

Network disconnect: B=exponential backoff 10 retries, D=retry once then null, E=buffer 1000 records retry 5s, F=empty result.
Hardware loss: A=poll 2s for 60s then restart, C=freeze drift then zero after 5min, D=terminate worker if no GPU.
Worker crash: A=EAGAIN pipe overflow, B=reconnect WebSocket, C=restart FFmpeg, D=on-failure:5, E=log and continue, F=retry 2x.
Queue overload: A=silent discard, B/C=deque eviction, D=Redis buffer, E=CSV fallback, F=limit concurrency.

## Dependency pins (§10.2)

faster-whisper==1.2.1, mediapipe==0.10.x, parselmouth==0.4.4, spacy==3.7.x, psycopg2-binary==2.9.x, pandas==2.2.x, celery==5.4.x, redis==5.x, fastapi==0.110.x, uvicorn==0.29.x, pydantic==2.x, numpy==1.26.x, pycryptodome>=3.20.0, patchright==1.58.2, TikTokLive==5.0.x

## Container topology (§9)

redis:7-alpine → postgres:16-alpine → stream_scrcpy(ubuntu:24.04+scrcpy) → worker(nvidia/cuda:12.2.2-cudnn8) + orchestrator(nvidia/cuda:12.2.2-cudnn8) → api(python:3.11-slim, depends on redis + worker + postgres for physiology ingress, task dispatch, and persistence). Network: appnetwork bridge. Shared volume: ipc-share at /tmp/ipc/.

## §4.E.1 — Operator Console (SPEC-AMEND-008)

The operator surface is the PySide6 Operator Console running on the
operator's host, not a container. All data flows through the API
Server's `/api/v1/operator/*` aggregate routes — the console does not
open direct Postgres or Redis connections. Launch with
`python -m services.operator_console`; packaging is driven by
`build/operator_console.spec` (PyInstaller one-dir).

The console is split into six pages, each backed by a dedicated
viewmodel that subscribes to the shared `OperatorStore`:

- **Overview** — active session, experiment, physiology, health, latest
  encounter, and an attention queue fed by alerts. First surface the
  operator sees; clicking the active-session card jumps to Live Session.
- **Live Session** — per-segment encounter table plus a reward-explanation
  detail pane exposing §7B's inputs (`p90_intensity`, `semantic_gate`,
  `gated_reward`, `n_frames_in_window`, `baseline_b_neutral`). Hosts
  the stimulus action rail; countdown derives from the authoritative
  `_stimulus_time` readback, not from the click wall-clock.
- **Experiments** — §7B Thompson Sampling readback: active arm, per-arm
  posterior α/β, evaluation variance, and selection count. Must not
  imply `semantic_confidence` moves the reward.
- **Physiology** — operator and streamer RMSSD, heart rate, and
  freshness with §4.C.4's four distinct states (fresh / stale / absent
  / no-rmssd). §7C Co-Modulation Index `null` is rendered as a
  legitimate `null-valid` outcome with its `null_reason`, not an error.
- **Health** — §12 subsystem rollup keeping `degraded` (WARN),
  `recovering` (PROGRESS), and `error` (ERROR) visually distinct; the
  first error subsystem's `operator_action_hint` is surfaced on the
  error summary card so the operator does not have to scroll.
- **Sessions** — recent-sessions history; double-click emits
  `session_selected(UUID)` which the shell routes through the same
  navigation handler Overview uses.

Stimulus action-rail lifecycle (§4.C): `IDLE → SUBMITTING → ACCEPTED →
MEASURING → COMPLETED` (or `FAILED` on terminal error). The ActionBar
disables the submit button during SUBMITTING so a second physical click
cannot re-dispatch; idempotency across the wire uses `client_action_id`.

## Physiology extension references (v3.2)

### §4.B.2 — Physiological Ingestion Adapter
API Server route `POST /api/v1/ingest/oura/webhook` verifies `x-oura-signature` with `OURA_WEBHOOK_SECRET`, validates `subject_role` plus Oura notification metadata (`event_type`, `data_type`, `start_datetime`, `end_datetime`), deduplicates via Redis `SETNX` idempotency keys under `physio:seen:*`, and enqueues hydration metadata to Redis list `physio:hydrate`. Within the API Server runtime, `OuraHydrationService` drains `physio:hydrate`, fetches provider resources via OAuth2, normalizes them into Physiological Chunk Event records (`event_type="physiological_chunk"`, `provider="oura"`), and pushes validated payloads to Redis list `physio:events`. Accepted deliveries return `{status, event_id}`; duplicates return 200 with `status="duplicate"`; malformed payloads return 422; Redis failures return 503.

### §4.C.4 — Physiological State Buffer
The Orchestrator performs bounded non-blocking `LPOP` drains from `physio:events`, stores rolling per-`subject_role` chunk buffers in memory, and derives scalar `_physiological_context` snapshots over the trailing 5-minute window. It prefers overlapping `ibi` chunks, otherwise overlapping `session` chunks, computes `validity_ratio` / `is_valid`, carries `source_kind` and `derivation_method`, computes `freshness_s` at `assemble_segment()` wall-clock time rather than ADB drift-corrected device time, and leaves stale snapshots present with `is_stale=True` instead of blocking dispatch.

### §4.E.2 — Physiology Persistence
Module E persists per-segment derived physiological snapshots from `_physiological_context` into PostgreSQL table `physiology_log` with `session_id`, `segment_id`, `subject_role`, `rmssd_ms`, `heart_rate_bpm`, freshness metadata, provider, `source_kind`, `derivation_method`, `window_s`, `validity_ratio`, `is_valid`, and source timestamp. Persistence remains scalar-only for replay and analytics joins.

### §7C — Co-Modulation Analytics
Module E computes the Co-Modulation Index as a rolling Pearson correlation over 1-minute resampled, time-aligned valid RMSSD snapshots for `streamer` and `operator` across the trailing 10-minute window, persists the result to `comodulation_log`, and returns null when fewer than four aligned non-stale pairs exist or either aligned series has zero variance.

### SPEC-AMEND-001: Worker GPU Architecture Downgrade

Section 9.1 of tech-spec-v3.0 mandated cuDNN 9, which lacks SM 6.1 binaries and causes a phantom docker tag CI failure. The spec is formally amended to use `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`. To meet the 30ms latency target on the GTX 1080 Ti without FP16 capability, `faster-whisper` is explicitly locked to `compute_type="int8"` to utilize dp4a vectorization. OCI runtime overrides (NVIDIA_DISABLE_REQUIRE=1) and JIT compilation are strictly prohibited due to CVE-2024-0132 and thermal constraints.