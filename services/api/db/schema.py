"""
Database Schema — §5.2 Data Classification / §11 Variable Extraction Matrix

SQL table definitions for the Persistent Store (PostgreSQL).
Scalar analytical fields use their §11-defined SQL types;
floating-point metrics use DOUBLE PRECISION, integral gates/counts retain
INTEGER or BOOLEAN, and timestamps use TIMESTAMPTZ (§2.7).
"""

from __future__ import annotations

# §5.2 — Permanent Analytical Storage tier
# Only anonymized analytical metrics are persisted.

SCHEMA_SQL: str = """
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY,
    stream_url      TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);

-- §11 — AU12 Intensity Score, Vocal Pitch, Jitter, Shimmer
CREATE TABLE IF NOT EXISTS metrics (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id),
    segment_id      TEXT NOT NULL,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    au12_intensity  DOUBLE PRECISION,
    pitch_f0        DOUBLE PRECISION,
    jitter          DOUBLE PRECISION,
    shimmer         DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

-- §11 — Semantic Match (Azure OpenAI evaluation)
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
    arm             TEXT NOT NULL,
    alpha_param     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta_param      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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
    is_valid            BOOLEAN NOT NULL,
    n_frames            INTEGER NOT NULL,
    baseline_neutral    DOUBLE PRECISION,
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
"""
