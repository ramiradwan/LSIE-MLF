-- =============================================================================
-- LSIE-MLF Experiment Seed Data — §4.E.1 Thompson Sampling
--
-- Seeds the experiments table with the four greeting line variants used by
-- the orchestrator's _select_experiment_arm() method. Arm names must match
-- the GREETING_LINES dict in services/worker/pipeline/orchestrator.py.
--
-- Prior: Beta(1, 1) — uninformative uniform prior per the locked-in
-- mathematical recipe (Formalizing TS for LSIE-MLF v2.0, §Prior Selection).
--
-- ON CONFLICT DO NOTHING ensures idempotent re-runs.
-- =============================================================================

INSERT INTO experiments (experiment_id, arm, alpha_param, beta_param)
VALUES
    ('greeting_line_v1', 'warm_welcome',      1.0, 1.0),
    ('greeting_line_v1', 'direct_question',   1.0, 1.0),
    ('greeting_line_v1', 'compliment_content', 1.0, 1.0),
    ('greeting_line_v1', 'simple_hello',      1.0, 1.0)
ON CONFLICT DO NOTHING;
