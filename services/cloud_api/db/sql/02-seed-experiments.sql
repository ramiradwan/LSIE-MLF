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

INSERT INTO experiments (
    experiment_id,
    label,
    arm,
    greeting_text,
    alpha_param,
    beta_param,
    enabled
)
VALUES
    (
        'greeting_line_v1',
        'Greeting Line V1',
        'warm_welcome',
        'Hey! Thanks for streaming, you''re awesome!',
        1.0,
        1.0,
        TRUE
    ),
    (
        'greeting_line_v1',
        'Greeting Line V1',
        'direct_question',
        'Hi! What''s the best advice you''ve gotten today?',
        1.0,
        1.0,
        TRUE
    ),
    (
        'greeting_line_v1',
        'Greeting Line V1',
        'compliment_content',
        'Love the energy on this stream! How long have you been live?',
        1.0,
        1.0,
        TRUE
    ),
    (
        'greeting_line_v1',
        'Greeting Line V1',
        'simple_hello',
        'Hello! Just joined, happy to be here!',
        1.0,
        1.0,
        TRUE
    )
ON CONFLICT DO NOTHING;
