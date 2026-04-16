-- =============================================================================
-- LSIE-MLF Physiology Schema — §5, §7C, §4.E.2 (v3.1 amendment)
--
-- New tables for physiological telemetry persistence and co-modulation
-- analytics. Follows the existing pattern from 01-schema.sql:
--   - session_id FK to sessions table
--   - TIMESTAMPTZ for temporal fields
--   - DOUBLE PRECISION for floating-point metrics
--   - Indexes on session + temporal columns
--
-- Data governance (§5):
--   Only normalized analytical derivatives enter these tables.
--   Raw Oura webhook JSON payloads are Transient Sensitive Data and
--   MUST NOT be persisted here. The physiology_log stores per-segment
--   snapshots (scalar RMSSD/HR + quality metadata), not raw provider
--   payloads.
--
-- Placement: data/sql/03-physiology.sql
-- Execution: mounted to /docker-entrypoint-initdb.d/ via docker-compose.yml
-- Idempotent: CREATE TABLE IF NOT EXISTS + ON CONFLICT DO NOTHING
-- =============================================================================

-- §4.E.2 / §7C — Per-segment physiological snapshot log
-- Records the physiological context attached to each 30-second segment.
-- One row per subject_role per segment (max 2 rows per segment).
CREATE TABLE IF NOT EXISTS physiology_log (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              UUID NOT NULL REFERENCES sessions(session_id),
    segment_id              TEXT NOT NULL,
    subject_role            TEXT NOT NULL CHECK (subject_role IN ('streamer', 'operator')),
    rmssd_ms                DOUBLE PRECISION,
    heart_rate_bpm          INTEGER,
    freshness_s             DOUBLE PRECISION NOT NULL,
    is_stale                BOOLEAN NOT NULL,
    provider                TEXT NOT NULL,
    source_timestamp_utc    TIMESTAMPTZ NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_physiology_session
    ON physiology_log(session_id, subject_role, created_at);

CREATE INDEX IF NOT EXISTS idx_physiology_segment
    ON physiology_log(session_id, segment_id);


-- §7C — Co-modulation analytics log
-- Records rolling co-modulation correlation computations.
-- One row per analysis window per session.
CREATE TABLE IF NOT EXISTS comodulation_log (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              UUID NOT NULL REFERENCES sessions(session_id),
    window_end_utc          TIMESTAMPTZ NOT NULL,
    window_minutes          INTEGER NOT NULL,
    co_modulation_index     DOUBLE PRECISION,
    n_paired_observations   INTEGER NOT NULL,
    coverage_ratio          DOUBLE PRECISION NOT NULL,
    streamer_rmssd_mean     DOUBLE PRECISION,
    operator_rmssd_mean     DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comod_session
    ON comodulation_log(session_id, window_end_utc);
