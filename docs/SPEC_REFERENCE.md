# LSIE-MLF Specification Reference

This file extracts the key implementation-governing details from the immutable Master Technical Specification v3.0 together with accepted amendments in `docs/SPEC_AMENDMENTS.md`, including the v3.1 physiology extension/version-history context introduced by SPEC-AMEND-007. Claude Code loads this on demand via the `@docs/SPEC_REFERENCE.md` import in CLAUDE.md.

For the signed base spec, see `docs/tech-spec-v3.0.pdf`; the amendment registry defines the current governing behavior wherever the v3.1 physiology extension extends the original v3.0 text.

## Data flow pipeline (§2)

1. Mobile USB → Capture Container (scrcpy raw PCM s16le 48kHz + H.264)
2. Capture Container → Orchestrator Container via IPC Pipe `/tmp/ipc/audio_stream.raw` and `/tmp/ipc/video_stream.mkv`
3. API Server → Redis list `physio:events`: authenticated Oura webhook normalized into Physiological Sample Event
4. Orchestrator internal: FFmpeg resample 48kHz → 16kHz via `pipe:1`, bounded Redis drain into Physiological State Buffer
5. Module B → Module C: Python dicts with uniqueId, event_type, timestamp_utc, payload
6. Module C → Module D: `InferenceHandoffPayload` validated by Pydantic, 30s segments, optional `_physiological_context`
7. Module D → Module E: Celery async task via Redis broker (session_id, segment_id, AU12, transcription, semantic, pitch, jitter, shimmer)
8. Module E → PostgreSQL: parameterized INSERT, DOUBLE PRECISION metrics, TIMESTAMPTZ timestamps, physiology analytics tables

## Error handling matrix (§12)

Network disconnect: B=exponential backoff 10 retries, D=retry once then null, E=buffer 1000 records retry 5s, F=empty result.
Hardware loss: A=poll 2s for 60s then restart, C=freeze drift then zero after 5min, D=terminate worker if no GPU.
Worker crash: A=EAGAIN pipe overflow, B=reconnect WebSocket, C=restart FFmpeg, D=on-failure:5, E=log and continue, F=retry 2x.
Queue overload: A=silent discard, B/C=deque eviction, D=Redis buffer, E=CSV fallback, F=limit concurrency.

## Dependency pins (§10.2)

faster-whisper==1.2.1, mediapipe==0.10.x, parselmouth==0.4.4, spacy==3.7.x, psycopg2-binary==2.9.x, pandas==2.2.x, celery==5.4.x, redis==5.x, fastapi==0.110.x, uvicorn==0.29.x, pydantic==2.x, numpy==1.26.x, pycryptodome>=3.20.0, patchright==1.58.2, TikTokLive==5.0.x

## Container topology (§9)

redis:7-alpine → postgres:16-alpine → stream_scrcpy(ubuntu:24.04+scrcpy) → worker(nvidia/cuda:12.2.2-cudnn8) + orchestrator(nvidia/cuda:12.2.2-cudnn8) → api(python:3.11-slim, depends on redis + worker + postgres for physiology ingress, task dispatch, and persistence). Network: appnetwork bridge. Shared volume: ipc-share at /tmp/ipc/.

## Physiology extension references (v3.1)

### §4.B.2 — Physiological Ingestion Adapter
API Server route `POST /api/v1/ingest/oura/webhook` verifies `x-oura-signature` with `OURA_WEBHOOK_SECRET`, validates `subject_role` plus provider metrics, normalizes the body into a Physiological Sample Event, and enqueues it to Redis list `physio:events`. Accepted payloads return `{status, event_id}`; duplicates are handled via Redis `SETNX` idempotency keys under `physio:seen:*`; malformed payloads return 422; Redis failures return 503.

### §4.C.4 — Physiological State Buffer
The orchestrator performs bounded non-blocking `LPOP` drains from `physio:events`, stores the latest per-`subject_role` snapshot in memory, and injects `_physiological_context` into segment payloads. `freshness_s` is computed at `assemble_segment()` wall-clock time rather than ADB drift-corrected device time, and stale snapshots remain present with `is_stale=True` instead of blocking dispatch.

### §4.E.2 — Physiology Persistence
Module E persists per-segment physiological snapshots from `_physiological_context` into PostgreSQL table `physiology_log` with `session_id`, `segment_id`, `subject_role`, `rmssd_ms`, `heart_rate_bpm`, freshness metadata, provider, and source timestamp. The table is keyed for segment-level replay and analytics joins.

### §7C — Co-Modulation Analytics
Module E computes the Co-Modulation Index as a rolling Pearson correlation over aligned 5-minute RMSSD bins for `streamer` and `operator`, persists the result to `comodulation_log`, and returns null when insufficient aligned non-stale pairs exist for the window.

### SPEC-AMEND-001: Worker GPU Architecture Downgrade

Section 9.1 of tech-spec-v3.0 mandated cuDNN 9, which lacks SM 6.1 binaries and causes a phantom docker tag CI failure. The spec is formally amended to use `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`. To meet the 30ms latency target on the GTX 1080 Ti without FP16 capability, `faster-whisper` is explicitly locked to `compute_type="int8"` to utilize dp4a vectorization. OCI runtime overrides (NVIDIA_DISABLE_REQUIRE=1) and JIT compilation are strictly prohibited due to CVE-2024-0132 and thermal constraints.