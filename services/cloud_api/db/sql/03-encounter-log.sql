-- =============================================================================
-- LSIE-MLF Encounter Audit Log Migration
--
-- §4.E.1 — Persistent reward computation trace for Thompson Sampling encounters.
-- §11 — Variable Extraction Matrix: captures all canonical reward pipeline outputs.
-- §5.2 — Permanent Analytical Storage: anonymized scalar metrics only.
-- §2.7 — Reward trace columns retain their §11 SQL types; floating-point values use DOUBLE PRECISION and timestamps use TIMESTAMPTZ.
--
-- Safe to run against databases that already have the base schema.
-- IF NOT EXISTS guards ensure idempotent re-runs.
-- =============================================================================

CREATE TABLE IF NOT EXISTS encounter_log (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              UUID NOT NULL REFERENCES sessions(session_id),
    segment_id              TEXT NOT NULL,
    experiment_id           TEXT NOT NULL,
    arm                     TEXT NOT NULL,
    timestamp_utc           TIMESTAMPTZ NOT NULL,
    gated_reward            DOUBLE PRECISION NOT NULL,
    p90_intensity           DOUBLE PRECISION NOT NULL,
    semantic_gate           INTEGER NOT NULL,
    n_frames_in_window      INTEGER NOT NULL,
    au12_baseline_pre       DOUBLE PRECISION,
    stimulus_time           DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'n_frames'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'n_frames_in_window'
    ) THEN
        ALTER TABLE encounter_log RENAME COLUMN n_frames TO n_frames_in_window;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'baseline_neutral'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'au12_baseline_pre'
    ) THEN
        ALTER TABLE encounter_log RENAME COLUMN baseline_neutral TO au12_baseline_pre;
    END IF;

    ALTER TABLE encounter_log ADD COLUMN IF NOT EXISTS n_frames_in_window INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE encounter_log ADD COLUMN IF NOT EXISTS au12_baseline_pre DOUBLE PRECISION;
    ALTER TABLE encounter_log ALTER COLUMN n_frames_in_window DROP DEFAULT;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'is_valid'
    ) THEN
        ALTER TABLE encounter_log DROP COLUMN is_valid;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'n_frames'
    ) THEN
        ALTER TABLE encounter_log DROP COLUMN n_frames;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'encounter_log' AND column_name = 'baseline_neutral'
    ) THEN
        ALTER TABLE encounter_log DROP COLUMN baseline_neutral;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_encounter_log_experiment ON encounter_log(experiment_id, arm);
CREATE INDEX IF NOT EXISTS idx_encounter_log_session ON encounter_log(session_id, timestamp_utc);
