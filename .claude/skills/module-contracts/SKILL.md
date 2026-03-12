---
name: module-contracts
description: Formal input/output/dependency/failure contracts for Modules A through F. Use when implementing inter-module communication, adding new pipeline stages, debugging data flow issues, or working on tasks/inference.py or pipeline/orchestrator.py.
---

# Module Contracts (§4.A–F)

## Module A — Hardware & Transport
Inputs: USB Android device, raw PCM s16le 48kHz + H.264 via scrcpy.
Outputs: continuous PCM stream to `/tmp/ipc/audio_stream.raw`, raw video frames.
Side effects: creates named pipe, opens fd 3.
Failures: USB disconnect → 2s poll for 60s. Pipe overflow → silent discard.

## Module B — Ground Truth Ingestion
Inputs: authenticated TikTok Webcast WebSocket (EulerStream-signed).
Outputs: parsed dicts with uniqueId, event_type, timestamp_utc, gift_value, combo_events.
Side effects: persistent WebSocket, signature token refresh.
Failures: disconnect → exponential backoff (1s initial, 30s max, 10 retries). EulerStream outage → degraded mode. Bad protobuf → log and discard.

## Module C — Orchestration & Synchronization
Inputs: raw PCM from IPC Pipe, ground truth events from B, ADB epoch timestamps.
Outputs: 16kHz resampled audio, drift-corrected metadata in deque, InferenceHandoffPayload segments (30s windows).
Side effects: persistent FFmpeg subprocess, in-memory deque buffer.
Failures: missing pipe → poll 30s. FFmpeg crash → auto-restart. ADB failure → freeze drift, reset after 5min.

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
