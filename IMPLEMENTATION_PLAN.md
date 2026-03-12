# LSIE-MLF Implementation Plan

Phased build order for full implementation of every scaffolded file. Each phase completes a self-contained, testable layer before downstream phases depend on it. Spec section references are pinned to every task.

---

## Phase 0 — Shared Contracts & Core Math (no external dependencies)

These files have zero infrastructure dependencies and form the foundation everything else imports.

**0.1 `packages/schemas/inference_handoff.py`** — Already complete. Pydantic model for InferenceHandoffPayload (§6.1). Verify JSON Schema Draft 07 export and round-trip serialization.

**0.2 `packages/schemas/events.py`** — Already complete. LiveEvent, GiftEvent, ComboEvent, GroundTruthRecord models (§4.B contract). Add unit tests for Action_Combo validation.

**0.3 `packages/schemas/evaluation.py`** — Already complete. SemanticEvaluationResult with `extra="forbid"` (§8.2). Add unit test confirming additionalProperties rejection.

**0.4 `packages/ml_core/au12.py`** — Implement `compute_intensity()` per the corrected §7.5 Python reference. This is verbatim math from the spec: landmark extraction (§7.2), IOD derivation (§7.3), baseline calibration and scoring (§7.4). Include epsilon guard, 5.0 hard-clamp, and type annotations. Unit test with synthetic landmark arrays.

**0.5 `packages/ml_core/encryption.py`** — Implement `encrypt()` and `decrypt()` using PyCryptodome AES-256-GCM (§5.1). Implement `secure_delete()` calling `shred -vfz -n 3`. Unit test encrypt→decrypt round-trip and nonce uniqueness.

**Why first:** Every downstream module imports these schemas and utilities. Completing them first means all later phases have validated contracts to code against.

---

## Phase 1 — ML Core Library (packages/ml_core)

Each wrapper is independently testable with mock inputs.

**1.1 `packages/ml_core/face_mesh.py`** — Implement MediaPipe Face Mesh initialization and `extract_landmarks()` returning (478, 3) ndarray or None (§4.D.2). Test with a sample image.

**1.2 `packages/ml_core/transcription.py`** — Implement faster-whisper `load_model()` and `transcribe()` with INT8/CUDA configuration (§4.D.1). Test with a short audio clip.

**1.3 `packages/ml_core/acoustic.py`** — Implement parselmouth `analyze()` returning pitch F0, jitter, shimmer (§4.D.3). Test with synthetic audio.

**1.4 `packages/ml_core/preprocessing.py`** — Implement spaCy `load_model()` and `preprocess()` for text normalization (§4.D.4). Test with noisy transcription strings.

**1.5 `packages/ml_core/semantic.py`** — Implement Azure OpenAI client initialization and `evaluate()` with deterministic params: temperature=0, top_p=1.0, seed=42, structured JSON output (§8.1–8.3). The canonical system prompt is already defined as a constant. Test with mock HTTP responses.

**Why second:** These are pure library functions with well-defined inputs and outputs. They must work correctly before the worker pipeline calls them.

---

## Phase 2 — Database Layer

**2.1 `services/api/db/schema.py`** — Already complete (DDL SQL). Execute against a local PostgreSQL instance to verify table creation, foreign keys, and indexes.

**2.2 `services/api/db/connection.py`** — Implement psycopg2 connection pool `init_pool()`, `close_pool()`, and `get_connection()` using environment variables (§2 step 7). Test pool lifecycle.

**2.3 `services/worker/pipeline/analytics.py` → `MetricsStore`** — Implement `connect()`, `insert_metrics()`, `_flush_buffer()`, and `_overflow_to_csv()`. Wire up the isolation levels: SERIALIZABLE for experiment updates, READ COMMITTED for metric inserts (§2 step 7). Implement the 1000-record buffer with 5-second retry and CSV fallback (§12.1 Module E). Test with actual PostgreSQL writes.

**2.4 `services/worker/pipeline/analytics.py` → `ThompsonSamplingEngine`** — Implement `select_arm()` and `update()` using SciPy Beta distributions (§4.E.1). Test with known alpha/beta values.

**Why third:** The Persistent Store is the terminal sink of the data pipeline. Having it fully functional means Phase 3 and 4 work can write real data end-to-end.

---

## Phase 3 — Worker Pipeline (Modules B, C)

**3.1 `services/worker/pipeline/orchestrator.py` → `DriftCorrector`** — Implement `poll()` executing ADB epoch command, computing `drift_offset = host_utc - android_epoch` (§4.C.1). Implement fallback: freeze after 3 failures, reset to zero after 5 minutes. Test with mocked ADB output.

**3.2 `services/worker/pipeline/orchestrator.py` → `AudioResampler`** — Implement `start()`, `read_chunk()`, `stop()` wrapping the FFmpeg subprocess (§4.C.2). Use the exact command from the spec: `ffmpeg -f s16le -ar 48000 -ac 1 -i /tmp/ipc/audio_stream.raw -ar 16000 -f s16le -ac 1 pipe:1`. Implement 1-second restart on crash (§2 step 3). Test with a synthetic PCM file.

**3.3 `services/worker/pipeline/orchestrator.py` → `Orchestrator`** — Implement `assemble_segment()` building InferenceHandoffPayload with 30-second windows, Pydantic validation, and drift-corrected timestamps (§2 step 5). Implement main `run()` loop coordinating drift polling, audio reading, event buffering, and segment dispatch to `process_segment()`.

**3.4 `services/worker/pipeline/ground_truth.py`** — Implement TikTokLive WebSocket connection with EulerStream signatures (§4.B.1). Implement `_handle_event()` parsing protobuf into LiveEvent dicts and applying Action_Combo constraint. Implement `run()` with exponential backoff reconnection: 1s initial, 30s max, 10 retries (§12.1 Module B).

**Why fourth:** The orchestrator assembles the data that feeds Module D. Ground truth ingestion runs in parallel. Both must be stable before end-to-end inference.

---

## Phase 4 — Celery Tasks (Modules D, E, F)

**4.1 `services/worker/tasks/inference.py` → `process_segment()`** — Wire up the full Module D pipeline: call transcription, face_mesh, AU12, acoustic, preprocessing, and semantic evaluation from ml_core. Assemble the output dict per §2 step 6 payload spec. Dispatch `persist_metrics` as a downstream Celery task.

**4.2 `services/worker/tasks/inference.py` → `persist_metrics()`** — Call `MetricsStore.insert_metrics()` with full error handling per §12.1 Module E.

**4.3 `services/worker/tasks/enrichment.py` → `scrape_context()`** — Implement patchright browser launch, scraping, and result extraction (§4.F.1). Implement all failure modes: CAPTCHA/HTTP → empty result, browser crash → retry, timeout → terminate (§4.F contract).

**4.4 `services/worker/vault_cron.py`** — Already complete. Verify it calls `EphemeralVault.secure_delete()` correctly and handles exceptions.

**Why fifth:** Tasks depend on the full ml_core library (Phase 1), database layer (Phase 2), and orchestrator schemas (Phase 3).

---

## Phase 5 — Capture Container (Module A)

**5.1 `services/stream_ingest/entrypoint.sh`** — Complete the `start_audio_capture()` function: launch scrcpy with `--audio-codec=raw --audio-buffer=30 --audio-dup --no-audio-playback`, pipe stdout to fd 3 via dd (§4.A.1 steps 3–4). Test full IPC pipe lifecycle: create → open → write → shutdown → crash recovery.

**5.2 `services/stream_ingest/Dockerfile`** — Add scrcpy v2.x installation from official release. Verify the image builds and entrypoint runs.

**Why sixth:** Module A is a standalone container that only writes to the IPC pipe. It can be developed in parallel with Phases 1–4 but is listed here because end-to-end testing requires Phases 3–4 to be reading from the pipe.

---

## Phase 6 — API Server

**6.1 `services/api/main.py`** — Already complete (FastAPI app with lifespan, routers). Verify startup with Uvicorn.

**6.2 `services/api/routes/metrics.py`** — Implement `get_metrics()`, `get_au12_timeseries()`, `get_acoustic_timeseries()` querying the Persistent Store via the connection pool. Parameterized queries only (§2 step 7).

**6.3 `services/api/routes/sessions.py`** — Implement `list_sessions()` and `get_session()` with summary metric aggregation.

**6.4 `services/api/Dockerfile`** — Verify build excludes ML packages (§3.2). Test image runs and serves on port 8000.

**Why seventh:** The API is a read-only consumer of the Persistent Store. It has no dependencies on the worker pipeline at runtime, but testing it meaningfully requires seeded data from Phases 2–4.

---

## Phase 7 — Docker Compose Integration & Dashboard

**7.1 `docker-compose.yml`** — Already complete. Validate the full five-container topology with `docker compose config`. Verify service dependency order (§9.6), GPU reservation (§9.3), device passthrough (§9.4), and volume mounts (§9.2).

**7.2 `services/worker/Dockerfile`** — Finalize: add spaCy model download, verify CUDA/cuDNN compatibility. Test GPU access inside container.

**7.3 `services/worker/dashboard.py`** — Implement Streamlit dashboard (§4.E.1): AU12 time-series plots, acoustic metric charts, Thompson Sampling experiment state, session overview.

**7.4 End-to-end smoke test** — Connect Android device, start all containers, verify data flows through every pipeline step (§2 steps 1–7), confirm metrics appear in PostgreSQL and API responses.

---

## Phase 8 — Hardening & Audit

**8.1 Run the 15-item autonomous implementation audit checklist (§13)** — Verify every criterion passes.

**8.2 Error handling sweep** — Confirm all four failure categories (§12.1–12.4) are implemented across every module.

**8.3 Canonical terminology audit** — Grep the entire codebase for retired synonyms from §0.3 and replace with canonical names.

**8.4 Dependency version lock** — Pin all packages to exact versions in requirements files. Verify against §10.2 matrix.

---

## File → Phase Mapping (quick reference)

| File | Phase |
|---|---|
| `packages/schemas/inference_handoff.py` | 0.1 |
| `packages/schemas/events.py` | 0.2 |
| `packages/schemas/evaluation.py` | 0.3 |
| `packages/ml_core/au12.py` | 0.4 |
| `packages/ml_core/encryption.py` | 0.5 |
| `packages/ml_core/face_mesh.py` | 1.1 |
| `packages/ml_core/transcription.py` | 1.2 |
| `packages/ml_core/acoustic.py` | 1.3 |
| `packages/ml_core/preprocessing.py` | 1.4 |
| `packages/ml_core/semantic.py` | 1.5 |
| `services/api/db/schema.py` | 2.1 |
| `services/api/db/connection.py` | 2.2 |
| `services/worker/pipeline/analytics.py` | 2.3–2.4 |
| `services/worker/pipeline/orchestrator.py` | 3.1–3.3 |
| `services/worker/pipeline/ground_truth.py` | 3.4 |
| `services/worker/tasks/inference.py` | 4.1–4.2 |
| `services/worker/tasks/enrichment.py` | 4.3 |
| `services/worker/vault_cron.py` | 4.4 |
| `services/stream_ingest/entrypoint.sh` | 5.1 |
| `services/stream_ingest/Dockerfile` | 5.2 |
| `services/api/main.py` | 6.1 |
| `services/api/routes/metrics.py` | 6.2 |
| `services/api/routes/sessions.py` | 6.3 |
| `services/api/Dockerfile` | 6.4 |
| `docker-compose.yml` | 7.1 |
| `services/worker/Dockerfile` | 7.2 |
| `services/worker/dashboard.py` | 7.3 |
