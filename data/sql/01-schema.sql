-- =============================================================================
-- LSIE-MLF Database Schema — §5.2 Data Classification / §11 Variable Extraction Matrix
--
-- Executed automatically by PostgreSQL on first container startup via
-- docker-entrypoint-initdb.d/ mount. All tables use IF NOT EXISTS for
-- idempotent re-runs.
--
-- §2.7 — Scalar analytical fields use their §11-defined SQL types; floating-point metrics use DOUBLE PRECISION, integral gates/counts retain INTEGER or BOOLEAN, and timestamps use TIMESTAMPTZ.
-- §5.2 — Only anonymized analytical metrics are persisted.
-- =============================================================================

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY,
    stream_url      TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);

-- §11 — AU12 intensity and canonical §7D observational acoustics.
-- Fresh installs create the canonical §7D field set directly here.
-- Existing deployments are upgraded by 04-metrics-observational-acoustics.sql.
CREATE TABLE IF NOT EXISTS metrics (
    id                              BIGSERIAL PRIMARY KEY,
    session_id                      UUID NOT NULL REFERENCES sessions(session_id),
    segment_id                      TEXT NOT NULL,
    timestamp_utc                   TIMESTAMPTZ NOT NULL,
    au12_intensity                  DOUBLE PRECISION,
    f0_valid_measure                BOOLEAN,
    f0_valid_baseline               BOOLEAN,
    perturbation_valid_measure      BOOLEAN,
    perturbation_valid_baseline     BOOLEAN,
    voiced_coverage_measure_s       DOUBLE PRECISION,
    voiced_coverage_baseline_s      DOUBLE PRECISION,
    f0_mean_measure_hz              DOUBLE PRECISION,
    f0_mean_baseline_hz             DOUBLE PRECISION,
    f0_delta_semitones              DOUBLE PRECISION,
    jitter_mean_measure             DOUBLE PRECISION,
    jitter_mean_baseline            DOUBLE PRECISION,
    jitter_delta                    DOUBLE PRECISION,
    shimmer_mean_measure            DOUBLE PRECISION,
    shimmer_mean_baseline           DOUBLE PRECISION,
    shimmer_delta                   DOUBLE PRECISION,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- §11 — ASR Transcription
CREATE TABLE IF NOT EXISTS transcripts (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id),
    segment_id      TEXT NOT NULL,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    text            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- §11 — Semantic Match (bounded deterministic evaluation)
CREATE TABLE IF NOT EXISTS evaluations (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id),
    segment_id      TEXT NOT NULL,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    reasoning       TEXT,
    is_match        BOOLEAN,
    confidence      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- §11 — Action Combo Trigger (ground truth events)
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id),
    unique_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    gift_value      INTEGER,
    is_combo        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- §11 — Evaluation Variance (Thompson Sampling)
CREATE TABLE IF NOT EXISTS experiments (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    label           TEXT,
    arm             TEXT NOT NULL,
    greeting_text   TEXT,
    alpha_param     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta_param      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    end_dated_at    TIMESTAMPTZ DEFAULT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rollout-safe additive experiment-admin columns for existing deployments.
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS label TEXT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS greeting_text TEXT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS enabled BOOLEAN;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS end_dated_at TIMESTAMPTZ;
UPDATE experiments SET label = experiment_id WHERE label IS NULL;
UPDATE experiments SET greeting_text = arm WHERE greeting_text IS NULL;
UPDATE experiments SET enabled = TRUE WHERE enabled IS NULL;
ALTER TABLE experiments ALTER COLUMN enabled SET DEFAULT TRUE;
ALTER TABLE experiments ALTER COLUMN enabled SET NOT NULL;
ALTER TABLE experiments ALTER COLUMN end_dated_at SET DEFAULT NULL;

-- §4.E.1 / §11 — Encounter Audit Log (reward computation trace)
CREATE TABLE IF NOT EXISTS encounter_log (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          UUID NOT NULL REFERENCES sessions(session_id),
    segment_id          TEXT NOT NULL,
    experiment_id       TEXT NOT NULL,
    arm                 TEXT NOT NULL,
    timestamp_utc       TIMESTAMPTZ NOT NULL,
    gated_reward        DOUBLE PRECISION NOT NULL,
    p90_intensity       DOUBLE PRECISION NOT NULL,
    semantic_gate       INTEGER NOT NULL,
    n_frames_in_window  INTEGER NOT NULL,
    au12_baseline_pre   DOUBLE PRECISION,
    stimulus_time       DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- §11 — External Context Metadata
CREATE TABLE IF NOT EXISTS context (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID REFERENCES sessions(session_id),
    source_url      TEXT NOT NULL,
    scraped_at_utc  TIMESTAMPTZ NOT NULL,
    data            JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_evaluations_session ON evaluations(session_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_experiments_lookup ON experiments(experiment_id, arm);
CREATE INDEX IF NOT EXISTS idx_encounter_log_experiment ON encounter_log(experiment_id, arm);
CREATE INDEX IF NOT EXISTS idx_encounter_log_session ON encounter_log(session_id, timestamp_utc);
