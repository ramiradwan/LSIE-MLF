-- =============================================================================
-- LSIE-MLF Observational Acoustic Metrics Rollout Migration
--
-- §2.7 / §11.4 / §13.23 — Add the canonical §7D observational acoustic
-- columns to the metrics table for already-initialized environments.
--
-- Safe rollout rules:
--   - additive only; no destructive schema changes
--   - legacy pitch_f0 / jitter / shimmer columns remain in place
--   - mean/delta analytics remain nullable; no NOT NULL and no defaults
--   - persist only anonymized analytical outputs, never raw audio,
--     waveform arrays, embeddings, or reconstructive voiceprint data
--
-- Idempotent by design via ADD COLUMN IF NOT EXISTS.
-- =============================================================================

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
