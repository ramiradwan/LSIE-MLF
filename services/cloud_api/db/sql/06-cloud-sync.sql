CREATE TABLE IF NOT EXISTS segment_telemetry (
    segment_id TEXT PRIMARY KEY CHECK (segment_id ~ '^[a-f0-9]{64}$'),
    session_id UUID NOT NULL,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_segment_telemetry_session_received
    ON segment_telemetry (session_id, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_segment_telemetry_received
    ON segment_telemetry (received_at DESC);

CREATE TABLE IF NOT EXISTS posterior_delta_log (
    event_id UUID PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    arm_id TEXT NOT NULL,
    delta_alpha DOUBLE PRECISION NOT NULL CHECK (delta_alpha >= 0.0 AND delta_alpha <= 1.0),
    delta_beta DOUBLE PRECISION NOT NULL CHECK (delta_beta >= 0.0 AND delta_beta <= 1.0),
    segment_id TEXT NOT NULL CHECK (segment_id ~ '^[a-f0-9]{64}$'),
    client_id TEXT NOT NULL,
    applied_at_utc TIMESTAMPTZ NOT NULL,
    decision_context_hash TEXT CHECK (decision_context_hash IS NULL OR decision_context_hash ~ '^[a-f0-9]{64}$'),
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (segment_id, client_id, arm_id)
);

CREATE INDEX IF NOT EXISTS idx_posterior_delta_log_segment_received
    ON posterior_delta_log (segment_id, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_posterior_delta_log_received
    ON posterior_delta_log (received_at DESC);
