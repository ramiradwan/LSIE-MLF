---
name: module-contracts
description: Formal input/output/dependency/failure contracts for Modules A through F. Use when implementing inter-module communication, adding new pipeline stages, debugging data flow issues, or working on tasks/inference.py or pipeline/orchestrator.py.
---

# Module Contracts (§4.A–F)

## Module A — Hardware & Transport
Inputs: USB Android device, raw PCM s16le 48kHz + H.264 via scrcpy v3.3.4 (SPEC-AMEND-004).
Outputs: continuous PCM stream to `/tmp/ipc/audio_stream.raw`, MKV video stream to `/tmp/ipc/video_stream.mkv`.
Architecture: dual scrcpy instances — audio on port range 27100:27199, video on port range 27200:27299. 4-second staggered startup to prevent ADB server push collision.
scrcpy flags (audio): `--no-video --no-playback --audio-codec=raw --audio-buffer=30 --audio-dup --record=AUDIO_PIPE --record-format=wav --port=27100:27199 --tunnel-host=HOST_IP`.
scrcpy flags (video): `--no-audio --no-playback --video-codec=h264 --max-fps=30 --record=VIDEO_PIPE --record-format=mkv --port=27200:27299 --tunnel-host=HOST_IP`.
Side effects: creates named pipes (audio + video), opens fd 3 for audio pipe shield.
Failures: USB disconnect → 2s poll for 60s. Pipe overflow → silent discard. ADB tunnel cleanup on restart cycle.

## Module B — Ground Truth Ingestion
Inputs: authenticated TikTok Webcast WebSocket (EulerStream-signed).
Outputs: parsed dicts with uniqueId, event_type, timestamp_utc, gift_value, combo_events.
Side effects: persistent WebSocket, signature token refresh.
Failures: disconnect → exponential backoff (1s initial, 30s max, 10 retries). EulerStream outage → degraded mode. Bad protobuf → log and discard.
**Note:** Module B is fully implemented (SignatureProvider protocol, EulerStreamSigner, WebSocket reconnection, Action_Combo constraint) but not wired into the orchestrator loop. Integration requires EulerStream third-party API access — an external dependency not available in the current test environment. Wiring Module B is an ADO work item, not a hardening task.

## Module C — Orchestration & Synchronization
Container: runs as a **separate orchestrator container** (same image as worker, different CMD: `python3.11 -m services.worker.run_orchestrator`) (SPEC-AMEND-003).
Inputs: raw PCM from IPC Pipe (`/tmp/ipc/audio_stream.raw`), video frames from `/tmp/ipc/video_stream.mkv` (via PyAV), ground truth events from B, ADB epoch timestamps.
Outputs: 16kHz resampled audio, drift-corrected metadata in deque, InferenceHandoffPayload segments (30s windows), AU12 scores from video frames.
Stimulus injection: triggered via Redis pub/sub (`stimulus:trigger` channel) or auto-trigger timer (controlled by `AUTO_STIMULUS_DELAY_S` env var). Injects Thompson Sampling arm assignment into the experiment.
Video capture: self-healing thread reads from video pipe via PyAV, buffers latest frame in deque. Revival on pipe break with fresh MKV header from scrcpy restart.
Side effects: persistent FFmpeg subprocess, in-memory deque buffer, Celery task dispatch via `process_segment.delay()`.
Failures: missing pipe → poll 30s. FFmpeg crash → auto-restart. ADB failure → freeze drift, reset after 5min. Video pipe break → revival attempt, graceful degradation to audio-only.

## Module D — Multimodal ML Processing
Inputs: 16kHz audio, raw video frames, InferenceHandoffPayload.
Outputs: transcripts, 478-vertex landmarks, AU12 scores, acoustic metrics, semantic eval JSON.
Side effects: GPU model loading, Azure OpenAI HTTPS calls.
Failures: GPU OOM → worker restart. Whisper load fail → abort startup. No face → null metrics. LLM timeout → retry once then null.

## Module E — Experimentation & Analytics
Inputs: inference payloads via Celery from Module D.
Outputs: time-series in PostgreSQL, Thompson Sampling assignments, Streamlit dashboards.
Failures: DB outage → buffer 1000 records, retry 5s, overflow to CSV. Celery deser error → log and discard.

## Module F — Context Enrichment
Inputs: Celery tasks with target_url, scrape_type, timeout_seconds.
Outputs: JSON with source_url, scraped_at_utc, extracted data dict.
Side effects: ephemeral headless browser instances, outbound HTTP.
Failures: CAPTCHA → empty result. Browser crash → retry. Timeout → terminate. Broker outage → Celery reconnect.
