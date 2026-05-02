# LSIE-MLF Specification Reference

This file extracts the key implementation-governing details from the currently committed, signed Master Technical Specification PDF (`docs/tech-spec-v*.pdf`) and its extracted machine-readable index. The current base text includes the physiology/co-modulation, operator-console, corrected Oura-ingestion, acoustic-analytics, attribution, and deterministic semantic-scorer rollouts. `docs/SPEC_AMENDMENTS.md` remains the traceability registry for historical amendments and any future accepted deviations not yet folded into the signed PDF. The repository ships the physiology path where webhook notifications enqueue hydration work to `physio:hydrate`, the API-hosted Oura hydration worker fetches provider resources and emits PhysiologicalChunkEvent records to `physio:events`, the Orchestrator derives scalar `_physiological_context` snapshots from rolling chunk buffers, and Module E persists derived records for reward results and Thompson Sampling state, physiology/co-modulation, observational acoustics, bounded semantic outputs, and attribution ledger entries. Operator-facing analytics and session-state reads are served through Module E / API surfaces for PySide6 clients and are not part of the Persistent Store persistence boundary. Claude Code loads this on demand via the `@docs/SPEC_REFERENCE.md` import in CLAUDE.md.

For the signed base spec, see the single `docs/tech-spec-v*.pdf` file committed in this repository; the extracted JSON index is `docs/content.json`, generated via `python scripts/spec_ref_check.py --extract > docs/content.json`. The amendment registry defines the current governing behavior wherever an amendment overrides the base spec.

Authoritative artifact: `docs/registries/error_handling_matrix.yaml` is the review-controlled §12 failure-category registry used by the §13.11 structural audit.
Authoritative artifact: `docs/registries/variable_traceability.yaml` is the review-controlled §11 variable-producer registry used by the §13.13 structural audit.

## Version-agnostic current vocabulary additions (§0, §0.3, §6.1, §6.4, §7E, §8, §13.15)

Use the current canonical identifiers consistently in code, schemas, comments, and operator-facing implementation notes. §0 and §0.3 define canonical terminology and retired synonyms; §13.15 is the canonical-terminology audit guard.

Governing-section map for current additions:

- `segment_id` and `BanditDecisionSnapshot` — governed by §6.1 `InferenceHandoffPayload` schema requirements; `segment_id` is also defined in the §7E.8 attribution variable dictionary.
- `AttributionEvent`, `OutcomeEvent`, `EventOutcomeLink`, and `AttributionScore` — governed by §6.4 attribution schemas; §13.24 audits their schema and persistence conformance.
- `semantic_method` and `semantic_method_version` — carried as deterministic semantic-method metadata on §6.4 attribution records and governed by §8 deterministic semantic scorer method selection/versioning.
- `bounded_reason_code` / `semantic_reason_code` — governed by §8's bounded semantic reason-code requirement and the §11.4.22 `semantic_reason_code` variable definition; never persist unbounded semantic rationales.
- `soft_reward_candidate` — governed by §7E.3 semantic attribution scoring, §7E.8 attribution variables, and §11.5.9 storage in the Persistent Store `attribution_score` table; it remains observational and does not modulate §7B reward updates.

## Data flow pipeline (§2)

1. Mobile USB → Capture Container (scrcpy raw PCM s16le 48kHz + H.264)
2. Capture Container → Orchestrator Container via IPC Pipe `/tmp/ipc/audio_stream.raw` and `/tmp/ipc/video_stream.mkv`
3. API Server → Redis list `physio:hydrate`: authenticated Oura webhook notification normalized into hydration work
4. API-hosted Oura hydration worker → Redis list `physio:events`: OAuth2-backed provider fetch normalized into PhysiologicalChunkEvent
5. Orchestrator internal: FFmpeg resample 48kHz → 16kHz via `pipe:1`, bounded Redis drain into Physiological State Buffer, rolling-window RMSSD derivation
6. Module B → Module C: Python dicts with uniqueId, event_type, timestamp_utc, payload
7. Module C → Module D: `InferenceHandoffPayload` validated by Pydantic, 30s segments, optional `_physiological_context`
8. Module D → Module E: Celery asynchronous task dispatch through the Message Broker. The JSON payload carries `session_id`, deterministic `segment_id`, stable `segment_window_start_utc` / `segment_window_end_utc` bounds, carried-forward `_bandit_decision_snapshot`, optional carried-forward `_physiological_context`, Module C-derived `_au12_series`, transcription, semantic outputs (`reasoning` as a bounded reason code, `is_match`, `confidence_score`, `semantic_method`, `semantic_method_version`), and canonical §7D observational acoustic analytics fields (`f0_valid_measure`, `f0_valid_baseline`, `perturbation_valid_measure`, `perturbation_valid_baseline`, `voiced_coverage_measure_s`, `voiced_coverage_baseline_s`, `f0_mean_measure_hz`, `f0_mean_baseline_hz`, `f0_delta_semitones`, `jitter_mean_measure`, `jitter_mean_baseline`, `jitter_delta`, `shimmer_mean_measure`, `shimmer_mean_baseline`, `shimmer_delta`). Redis append-only persistence keeps queued tasks durable across broker restarts. If the broker is unreachable, Celery retries with exponential backoff up to five attempts, then writes failed payloads to `/data/processed/failed_tasks/`. Legacy scalar acoustic outputs are retired; implementations expose the canonical §7D windowed fields only.
9. Module E → Persistent Store: psycopg2-binary PostgreSQL connection-pool writes use parameterized SQL INSERTs storing numeric metrics as `DOUBLE PRECISION`, validity flags as `BOOLEAN`, and timestamps as `TIMESTAMPTZ`; experiment updates use SERIALIZABLE isolation and metric inserts use READ COMMITTED. If the database is unreachable, Module E buffers up to 1000 records in memory, retries every 5 seconds, then overflows to CSV. The current persistence scope covers derived analytics rather than scalar-only physiology tables: reward results and Thompson Sampling state, scalar physiology snapshots plus co-modulation logs, canonical observational acoustic analytics, bounded semantic evaluation outputs, and attribution ledger records (`AttributionEvent`, `OutcomeEvent`, `EventOutcomeLink`, `AttributionScore`). Operator-facing analytics and session-state read APIs are exposed by Module E through API surfaces for PySide6 clients; they consume persisted analytics but are not part of the Persistent Store persistence boundary.

## Error handling matrix (§12)

LSIE-MLF defines four standardized failure categories; each module implements deterministic recovery behavior for each category. Physiological webhook rejections (invalid HMAC signatures, malformed payloads, duplicate suppression, and Redis enqueue retries) are application-level ingress guardrails, not separate failure categories. Stale physiological snapshots are annotated with `is_stale = true` and do not block segment dispatch; Module E excludes stale snapshots from co-modulation computation. Acoustic insufficiency is also a data-quality condition: insufficient voiced coverage, invalid perturbation spans, undefined Praat outputs, or local Praat extraction failure cause Module D to emit deterministic false validity flags and null dependent acoustic outputs without retrying the task, crashing the worker, or blocking downstream persistence.

- Network disconnection: Module A=N/A; Module B=exponential WebSocket reconnection up to 10 retries, while physiological webhook ingress is handled by the API Server and upstream transport failures rely on provider-managed retries after timeout or HTTP 503; Module C=N/A; Module D=network loss affects only `llm_gray_band` fallback, retry the Azure OpenAI gray-band request once, then emit a deterministic semantic error reason code with bounded false/null-compatible outputs while local cross-encoder primary scoring remains available; Module E=buffer database writes and retry every 5 seconds; Module F=return empty scrape result.
- Hardware device loss: Module A=USB reconnection polling every 2 seconds for 60 seconds, then container restart; Module B=N/A; Module C=freeze drift offset after three consecutive ADB poll failures, return the cached offset while frozen, then reset drift to zero after 300 seconds / 5 minutes; Module D=terminate worker container if GPU is unavailable; Module E=N/A; Module F=N/A.
- Worker process crash: Module A=pipe overflow handled through non-blocking EAGAIN writes; Module B=reconnect WebSocket on restart; Module C=restart FFmpeg within 1 second when the resampler detects a dead subprocess or EOF, and recreate/restart video capture after IPC / PyAV pipe break; Module D=Docker restart policy `on-failure:5`; Module E=log Celery task failure and continue; Module F=retry scraping task up to two times.
- Queue overload: Module A=kernel pipe overflow discards data silently; Module B=deque ring buffer evicts oldest entries, physiological event enqueue uses Redis list `physio:events`, and enqueue backpressure returns HTTP 503 so the provider retries instead of silently dropping events; Module C=deque maxlen eviction plus bounded physiological drain (max 100 events per cycle), with malformed physiological events logged and discarded; Module D=Celery queue buffers tasks in Redis; Module E=overflow records written to CSV fallback storage and null co-modulation results recorded when physiological coverage is incomplete or non-stale paired samples are insufficient; Module F=limit concurrent scraping tasks.

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
  `gated_reward`, `n_frames_in_window`, `au12_baseline_pre`). Hosts
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

## Physiology extension references

### §4.B.2 — Physiological Ingestion Adapter
API Server route `POST /api/v1/ingest/oura/webhook` verifies `x-oura-signature` with `OURA_WEBHOOK_SECRET`, validates `subject_role` plus Oura notification metadata (`event_type`, `data_type`, `start_datetime`, `end_datetime`), deduplicates via Redis `SETNX` idempotency keys under `physio:seen:*`, and enqueues hydration metadata to Redis list `physio:hydrate`. Within the API Server runtime, `OuraHydrationService` drains `physio:hydrate`, fetches provider resources via OAuth2, normalizes them into PhysiologicalChunkEvent records (`event_type="physiological_chunk"`, `provider="oura"`), and pushes validated payloads to Redis list `physio:events`. Accepted deliveries return `{status, event_id}`; duplicates return 200 with `status="duplicate"`; malformed payloads return 422; Redis failures return 503.

### §4.C.4 — Physiological State Buffer
The Orchestrator performs bounded non-blocking `LPOP` drains from `physio:events`, stores rolling per-`subject_role` chunk buffers in memory, and derives scalar `_physiological_context` snapshots over the trailing 5-minute window. It prefers overlapping `ibi` chunks, otherwise overlapping `session` chunks, computes `validity_ratio` / `is_valid`, carries `source_kind` and `derivation_method`, computes `freshness_s` at `assemble_segment()` wall-clock time rather than ADB drift-corrected device time, and leaves stale snapshots present with `is_stale=True` instead of blocking dispatch.

### §4.E.2 — Physiology Persistence
Module E persists per-segment derived physiological snapshots from `_physiological_context` into PostgreSQL table `physiology_log` with `session_id`, `segment_id`, `subject_role`, `rmssd_ms`, `heart_rate_bpm`, freshness metadata, provider, `source_kind`, `derivation_method`, `window_s`, `validity_ratio`, `is_valid`, and source timestamp. Persistence remains scalar-only for replay and analytics joins.

### §7C — Co-Modulation Analytics
Module E computes the Co-Modulation Index as a rolling Pearson correlation over 1-minute resampled, time-aligned valid RMSSD snapshots for `streamer` and `operator` across the trailing 10-minute window, persists the result to `comodulation_log`, and returns null when fewer than four aligned non-stale pairs exist or either aligned series has zero variance.

### SPEC-AMEND-001: Worker GPU Architecture Downgrade — *superseded by v4.0 §11.x*

The v3.4 cuDNN 8 / Pascal (SM 6.1) posture documented in SPEC-AMEND-001 is **superseded** by the v4.0 hardware matrix drafted at `docs/V4_SPEC_DRAFTS.md` §11.x. The v4.0 production GPU floor is NVIDIA Turing (SM 7.5+) on CUDA 12.x with cuDNN 9, paired with `ctranslate2 >= 4.5.0` and `torch == 2.4.x`. Pascal developer hosts route via the `LSIE_DEV_FORCE_CPU_SPEECH=1` escape hatch (CPU speech, GPU cross-encoder); the production preflight gate (`services/desktop_launcher/preflight.py`) hard-rejects SM<7.5 unless the `.dev_machine` marker is present. `faster-whisper.compute_type` remains locked to `"int8"` — on Turing+ it uses DP4A and the INT8 Tensor Cores, on the Pascal CPU escape hatch it is the right faster-whisper default. The historical SPEC-AMEND-001 entry in `docs/SPEC_AMENDMENTS.md` is retained for traceability.
