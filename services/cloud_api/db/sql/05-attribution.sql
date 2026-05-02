-- =============================================================================
-- LSIE-MLF Attribution Ledger Schema — §4.E.3 / §6.4 / §7E (v3.4)
--
-- Stable relational substrate for observational attribution analytics:
--   - AttributionEvent: deterministic interaction event identity
--   - OutcomeEvent: delayed downstream outcome capture
--   - EventOutcomeLink: deterministic event→outcome eligibility links
--   - AttributionScore: method-specific derived attribution scores
--
-- Replay/backfill idempotency is enforced in PostgreSQL by UUID primary keys
-- plus natural UNIQUE constraints mirroring each deterministic UUIDv5 input
-- tuple. This file is safe to rerun via IF NOT EXISTS and idempotent
-- constraint canonicalization guards.
--
-- Data governance (§5): persist only derived/versioned attribution artifacts.
-- Raw media, raw physiological chunks, and unbounded semantic prose MUST
-- NOT be stored in these tables.
-- =============================================================================

CREATE TABLE IF NOT EXISTS attribution_event (
    event_id                    UUID PRIMARY KEY,
    session_id                  UUID NOT NULL REFERENCES sessions(session_id),
    segment_id                  TEXT NOT NULL CHECK (segment_id ~ '^[0-9a-f]{64}$'),
    event_type                  TEXT NOT NULL CHECK (event_type IN ('greeting_interaction')),
    event_time_utc              TIMESTAMPTZ NOT NULL,
    stimulus_time_utc           TIMESTAMPTZ,
    selected_arm_id             TEXT NOT NULL,
    expected_rule_text_hash     TEXT NOT NULL CHECK (expected_rule_text_hash ~ '^[0-9a-f]{64}$'),
    semantic_method             TEXT NOT NULL,
    semantic_method_version     TEXT NOT NULL,
    semantic_p_match            DOUBLE PRECISION CHECK (
        semantic_p_match IS NULL OR semantic_p_match BETWEEN 0.0 AND 1.0
    ),
    semantic_reason_code        TEXT,
    reward_path_version         TEXT NOT NULL,
    bandit_decision_snapshot    JSONB NOT NULL,
    evidence_flags              TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    finality                    TEXT NOT NULL CHECK (
        finality IN ('online_provisional', 'offline_final')
    ),
    schema_version              TEXT NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_attribution_event_deterministic_identity
        UNIQUE (session_id, segment_id, event_type, reward_path_version)
);

ALTER TABLE attribution_event
    DROP CONSTRAINT IF EXISTS attribution_event_semantic_method_check;

ALTER TABLE attribution_event
    DROP CONSTRAINT IF EXISTS ck_attribution_event_semantic_method;

ALTER TABLE attribution_event
    ADD CONSTRAINT ck_attribution_event_semantic_method
    CHECK (semantic_method IN ('cross_encoder', 'llm_gray_band', 'azure_llm_legacy')) NOT VALID;

CREATE TABLE IF NOT EXISTS outcome_event (
    outcome_id          UUID PRIMARY KEY,
    session_id          UUID NOT NULL REFERENCES sessions(session_id),
    outcome_type        TEXT NOT NULL CHECK (outcome_type IN ('creator_follow')),
    outcome_value       DOUBLE PRECISION NOT NULL,
    outcome_time_utc    TIMESTAMPTZ NOT NULL,
    source_system       TEXT NOT NULL,
    source_event_ref    TEXT,
    confidence          DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    finality            TEXT NOT NULL CHECK (finality IN ('online_provisional', 'offline_final')),
    schema_version      TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_outcome_event_deterministic_identity
        UNIQUE NULLS NOT DISTINCT (
            session_id,
            outcome_type,
            outcome_time_utc,
            source_system,
            source_event_ref
        )
);

CREATE TABLE IF NOT EXISTS event_outcome_link (
    link_id             UUID PRIMARY KEY,
    event_id            UUID NOT NULL REFERENCES attribution_event(event_id),
    outcome_id          UUID NOT NULL REFERENCES outcome_event(outcome_id),
    lag_s               DOUBLE PRECISION NOT NULL CHECK (lag_s >= 0.0),
    horizon_s           DOUBLE PRECISION NOT NULL CHECK (horizon_s > 0.0),
    link_rule_version   TEXT NOT NULL,
    eligibility_flags   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    finality            TEXT NOT NULL CHECK (finality IN ('online_provisional', 'offline_final')),
    schema_version      TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_event_outcome_link_lag_within_horizon CHECK (lag_s <= horizon_s),
    CONSTRAINT uq_event_outcome_link_deterministic_identity
        UNIQUE (event_id, outcome_id, link_rule_version)
);

CREATE TABLE IF NOT EXISTS attribution_score (
    score_id                UUID PRIMARY KEY,
    event_id                UUID NOT NULL REFERENCES attribution_event(event_id),
    outcome_id              UUID REFERENCES outcome_event(outcome_id),
    attribution_method      TEXT NOT NULL,
    method_version          TEXT NOT NULL,
    score_raw               DOUBLE PRECISION,
    score_normalized        DOUBLE PRECISION,
    confidence              DOUBLE PRECISION CHECK (
        confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0
    ),
    evidence_flags          TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    finality                TEXT NOT NULL CHECK (
        finality IN ('online_provisional', 'offline_final')
    ),
    schema_version          TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_attribution_score_deterministic_identity
        UNIQUE NULLS NOT DISTINCT (
            event_id,
            outcome_id,
            attribution_method,
            method_version
        )
);

-- AttributionEvent read paths.
CREATE INDEX IF NOT EXISTS idx_attribution_event_session
    ON attribution_event(session_id);
CREATE INDEX IF NOT EXISTS idx_attribution_event_segment
    ON attribution_event(segment_id);
CREATE INDEX IF NOT EXISTS idx_attribution_event_time
    ON attribution_event(event_time_utc);
CREATE INDEX IF NOT EXISTS idx_attribution_event_finality
    ON attribution_event(finality);

-- OutcomeEvent read paths.
CREATE INDEX IF NOT EXISTS idx_outcome_event_session
    ON outcome_event(session_id);
CREATE INDEX IF NOT EXISTS idx_outcome_event_time
    ON outcome_event(outcome_time_utc);
CREATE INDEX IF NOT EXISTS idx_outcome_event_finality
    ON outcome_event(finality);

-- Event/outcome join paths.
CREATE INDEX IF NOT EXISTS idx_event_outcome_link_event
    ON event_outcome_link(event_id);
CREATE INDEX IF NOT EXISTS idx_event_outcome_link_outcome
    ON event_outcome_link(outcome_id);
CREATE INDEX IF NOT EXISTS idx_event_outcome_link_finality
    ON event_outcome_link(finality);
CREATE INDEX IF NOT EXISTS idx_attribution_score_event
    ON attribution_score(event_id);
CREATE INDEX IF NOT EXISTS idx_attribution_score_outcome
    ON attribution_score(outcome_id);
CREATE INDEX IF NOT EXISTS idx_attribution_score_finality
    ON attribution_score(finality);
