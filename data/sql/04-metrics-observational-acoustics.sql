-- =============================================================================
-- LSIE-MLF Observational Acoustic Metrics Canonicalization Migration
--
-- §2.7 / §11.4 / §13.23 — Ensure already-initialized metrics tables expose
-- only the canonical §7D observational acoustic columns.
--
-- Safe rollout rules:
--   - drop retired legacy scalar acoustic columns in place; historical values
--     are not archived by policy
--   - mean/delta analytics remain nullable; no NOT NULL and no defaults
--   - persist only anonymized analytical outputs, never raw audio,
--     waveform arrays, embeddings, or reconstructive voiceprint data
--
-- Idempotent by design via DROP/ADD COLUMN IF EXISTS/IF NOT EXISTS.
-- =============================================================================

DO $$
BEGIN
    EXECUTE format(
        'ALTER TABLE metrics DROP COLUMN IF EXISTS %I',
        'pitch_' || 'f0'
    );
END;
$$;

ALTER TABLE metrics
    DROP COLUMN IF EXISTS jitter,
    DROP COLUMN IF EXISTS shimmer;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS f0_valid_measure BOOLEAN;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS f0_valid_baseline BOOLEAN;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS perturbation_valid_measure BOOLEAN;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS perturbation_valid_baseline BOOLEAN;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS voiced_coverage_measure_s DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS voiced_coverage_baseline_s DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS f0_mean_measure_hz DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS f0_mean_baseline_hz DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS f0_delta_semitones DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS jitter_mean_measure DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS jitter_mean_baseline DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS jitter_delta DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS shimmer_mean_measure DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS shimmer_mean_baseline DOUBLE PRECISION;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS shimmer_delta DOUBLE PRECISION;
