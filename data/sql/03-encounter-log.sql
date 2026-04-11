-- =============================================================================
-- LSIE-MLF Encounter Audit Log Migration
--
-- §4.E.1 — Persistent reward computation trace for Thompson Sampling encounters.
-- §11 — Variable Extraction Matrix: captures all reward pipeline outputs.
-- §5.2 — Permanent Analytical Storage: anonymized scalar metrics only.
-- §2.7 — Reward trace columns retain their §11 SQL types; floating-point values use DOUBLE PRECISION and timestamps use TIMESTAMPTZ.
--
-- Safe to run against databases that already have the base schema.
-- IF NOT EXISTS guards ensure idempotent re-runs.
-- =============================================================================

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

CREATE INDEX IF NOT EXISTS idx_encounter_log_experiment ON encounter_log(experiment_id, arm);
CREATE INDEX IF NOT EXISTS idx_encounter_log_session ON encounter_log(session_id, timestamp_utc);
