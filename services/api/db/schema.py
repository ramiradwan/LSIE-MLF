"""
Database Schema — §5.2 Data Classification / §11 Variable Extraction Matrix

SQL table definitions for the Persistent Store (PostgreSQL).
All metrics stored as DOUBLE PRECISION, timestamps as TIMESTAMPTZ (§2 step 7).
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
"""
