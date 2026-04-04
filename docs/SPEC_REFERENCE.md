# LSIE-MLF Specification Reference

This file extracts the key implementation-governing details from the Master Technical Specification v3.0. Claude Code loads this on demand via the `@docs/SPEC_REFERENCE.md` import in CLAUDE.md.

For the full spec, see `docs/tech-spec-v3.0.pdf`.

## Data flow pipeline (§2)

1. Mobile USB → Capture Container (scrcpy raw PCM s16le 48kHz + H.264)
2. Capture Container → ML Worker via IPC Pipe `/tmp/ipc/audio_stream.raw`
3. ML Worker internal: FFmpeg resample 48kHz → 16kHz via `pipe:1`
4. Module B → Module C: Python dicts with uniqueId, event_type, timestamp_utc, payload
5. Module C → Module D: `InferenceHandoffPayload` validated by Pydantic, 30s segments
6. Module D → Module E: Celery async task via Redis broker (session_id, segment_id, AU12, transcription, semantic, pitch, jitter, shimmer)
7. Module E → PostgreSQL: parameterized INSERT, DOUBLE PRECISION metrics, TIMESTAMPTZ timestamps

## Error handling matrix (§12)

Network disconnect: B=exponential backoff 10 retries, D=retry once then null, E=buffer 1000 records retry 5s, F=empty result.
Hardware loss: A=poll 2s for 60s then restart, C=freeze drift then zero after 5min, D=terminate worker if no GPU.
Worker crash: A=EAGAIN pipe overflow, B=reconnect WebSocket, C=restart FFmpeg, D=on-failure:5, E=log and continue, F=retry 2x.
Queue overload: A=silent discard, B/C=deque eviction, D=Redis buffer, E=CSV fallback, F=limit concurrency.

## Dependency pins (§10.2)

faster-whisper==1.2.1, mediapipe==0.10.x, parselmouth==0.4.4, spacy==3.7.x, psycopg2-binary==2.9.x, pandas==2.2.x, celery==5.4.x, redis==5.x, fastapi==0.110.x, uvicorn==0.29.x, pydantic==2.x, numpy==1.26.x, pycryptodome>=3.20.0, patchright==1.58.2, TikTokLive==5.0.x

## Container topology (§9)

redis:7-alpine → postgres:16-alpine → stream_scrcpy(ubuntu:22.04+scrcpy) → worker(nvidia/cuda:12.2.2-cudnn8) → api(python:3.11-slim). Network: appnetwork bridge. Shared volume: ipc-share at /tmp/ipc/.

### SPEC-AMEND-001: Worker GPU Architecture Downgrade

Section 9.1 of tech-spec-v3.0 mandated cuDNN 9, which lacks SM 6.1 binaries and causes a phantom docker tag CI failure. The spec is formally amended to use `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`. To meet the 30ms latency target on the GTX 1080 Ti without FP16 capability, `faster-whisper` is explicitly locked to `compute_type="int8"` to utilize dp4a vectorization. OCI runtime overrides (NVIDIA_DISABLE_REQUIRE=1) and JIT compilation are strictly prohibited due to CVE-2024-0132 and thermal constraints.
