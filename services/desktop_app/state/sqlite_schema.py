"""SQLite schema port (WS4 P1).

Local persistence layer for the v4.0 desktop process graph. Replaces
the v3.4 PostgreSQL DDL spread across ``data/sql/0{1,3,4,5}*.sql`` so
``analytics_state_worker`` (sole writer) and ``ui_api_shell`` (read
side, ``PRAGMA query_only=1``) can talk to a local SQLite file under
the platform's standard application-data directory.

Translation rules from PostgreSQL to SQLite:

  ``BIGSERIAL`` → ``INTEGER PRIMARY KEY AUTOINCREMENT``
  ``UUID``      → ``TEXT`` (Pydantic shape-validates at write time).
  ``JSONB``     → ``TEXT`` carrying a JSON string; the writer Pydantic-
                  validates the model before serialising.
  ``TIMESTAMPTZ`` → ``TEXT`` containing an ISO-8601 UTC string. This is
                  the canonical wire format produced by
                  ``Orchestrator._canonical_utc_timestamp``.
  ``DOUBLE PRECISION`` → ``REAL``.
  ``BOOLEAN``   → ``INTEGER`` (0 / 1; SQLite stores BOOLEAN as INTEGER).
  ``TEXT[]``    → ``TEXT`` (JSON array string; Pydantic validates).
  ``DEFAULT NOW()`` → ``DEFAULT CURRENT_TIMESTAMP`` (SQLite's built-in
                  ``CURRENT_TIMESTAMP`` produces a space-separated UTC
                  string; readers normalise on the way out).
  ``REFERENCES`` foreign keys are kept; PRAGMA foreign_keys=ON enables
                  enforcement on every connection.

Divergences from PostgreSQL the writer / reader compensates for:

  Regex ``CHECK`` constraints (``segment_id ~ '^[0-9a-f]{64}$'``) are
  dropped — Pydantic enforces shape at write time, SQLite has no
  native regex without an application-defined callback. Enum-style
  ``CHECK ... IN (...)`` constraints survive verbatim.

  ``UNIQUE NULLS NOT DISTINCT`` (PostgreSQL 15+) has no SQLite
  equivalent; the affected uniqueness contracts (e.g.
  ``outcome_event`` deterministic identity) are app-level enforced
  via deterministic UUIDv5 keys upstream of the write.

Per the WS4 P1 plan, the ``experiments`` table is locally seeded with
the four greeting variants from ``data/sql/02-seed-experiments.sql``
so the operator console renders real Experiments-page data
immediately. WS5 P2 will replace the seed with a cloud-synced
``ExperimentBundle`` cache.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from typing import Final

logger = logging.getLogger(__name__)


# WAL + busy-timeout configuration per WS4 P1. The writer applies the
# full set; the reader applies only the read-side subset and adds
# ``query_only=1`` to enforce the single-writer invariant.
PRAGMAS_WRITER: Final[tuple[str, ...]] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=ON",
)

PRAGMAS_READER: Final[tuple[str, ...]] = (
    "PRAGMA query_only=1",
    "PRAGMA busy_timeout=5000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=ON",
)


# Canonical table-creation order. Foreign keys flow forward: sessions
# is the root, attribution_event references sessions, event_outcome_link
# and attribution_score reference attribution_event + outcome_event.
SCHEMA_DDL: Final[tuple[str, ...]] = (
    # --- 01-schema.sql -----------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id      TEXT PRIMARY KEY,
        stream_url      TEXT NOT NULL,
        started_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        ended_at        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id                      TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id                      TEXT NOT NULL,
        timestamp_utc                   TEXT NOT NULL,
        au12_intensity                  REAL,
        f0_valid_measure                INTEGER,
        f0_valid_baseline               INTEGER,
        perturbation_valid_measure      INTEGER,
        perturbation_valid_baseline     INTEGER,
        voiced_coverage_measure_s       REAL,
        voiced_coverage_baseline_s      REAL,
        f0_mean_measure_hz              REAL,
        f0_mean_baseline_hz             REAL,
        f0_delta_semitones              REAL,
        jitter_mean_measure             REAL,
        jitter_mean_baseline            REAL,
        jitter_delta                    REAL,
        shimmer_mean_measure            REAL,
        shimmer_mean_baseline           REAL,
        shimmer_delta                   REAL,
        created_at                      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id      TEXT NOT NULL,
        timestamp_utc   TEXT NOT NULL,
        text            TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id      TEXT NOT NULL,
        timestamp_utc   TEXT NOT NULL,
        reasoning       TEXT,
        is_match        INTEGER,
        confidence      REAL,
        created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL REFERENCES sessions(session_id),
        unique_id       TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        timestamp_utc   TEXT NOT NULL,
        gift_value      INTEGER,
        is_combo        INTEGER DEFAULT 0,
        created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # WS4 P1: experiments is the local read-only cache. WS5 P2 will
    # populate it from the cloud ExperimentBundle; until then, the
    # bootstrap seeds the four greeting-line variants directly. The
    # ``UNIQUE(experiment_id, arm)`` constraint is new in v4.0 — it
    # tightens the v3.4 schema (which relied on PostgreSQL's ON
    # CONFLICT DO NOTHING + a non-unique index) so re-bootstrapping
    # the SQLite store is idempotent on the seed insert.
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id   TEXT NOT NULL,
        label           TEXT,
        arm             TEXT NOT NULL,
        greeting_text   TEXT,
        alpha_param     REAL NOT NULL DEFAULT 1.0,
        beta_param      REAL NOT NULL DEFAULT 1.0,
        enabled         INTEGER NOT NULL DEFAULT 1,
        end_dated_at    TEXT,
        updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (experiment_id, arm)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS encounter_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id          TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id          TEXT NOT NULL,
        experiment_id       TEXT NOT NULL,
        arm                 TEXT NOT NULL,
        timestamp_utc       TEXT NOT NULL,
        gated_reward        REAL NOT NULL,
        p90_intensity       REAL NOT NULL,
        semantic_gate       INTEGER NOT NULL,
        n_frames_in_window  INTEGER NOT NULL,
        au12_baseline_pre   REAL,
        stimulus_time       REAL,
        created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT REFERENCES sessions(session_id),
        source_url      TEXT NOT NULL,
        scraped_at_utc  TEXT NOT NULL,
        data            TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # --- 03-physiology.sql -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS physiology_log (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id              TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id              TEXT NOT NULL,
        subject_role            TEXT NOT NULL CHECK (subject_role IN ('streamer', 'operator')),
        rmssd_ms                REAL,
        heart_rate_bpm          INTEGER,
        freshness_s             REAL NOT NULL,
        is_stale                INTEGER NOT NULL,
        provider                TEXT NOT NULL,
        source_kind             TEXT NOT NULL CHECK (source_kind IN ('ibi','session')),
        derivation_method       TEXT NOT NULL,
        window_s                INTEGER NOT NULL CHECK (window_s > 0),
        validity_ratio          REAL NOT NULL CHECK (validity_ratio BETWEEN 0.0 AND 1.0),
        is_valid                INTEGER NOT NULL,
        source_timestamp_utc    TEXT NOT NULL,
        created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comodulation_log (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id              TEXT NOT NULL REFERENCES sessions(session_id),
        window_start_utc        TEXT NOT NULL,
        window_end_utc          TEXT NOT NULL,
        window_minutes          INTEGER NOT NULL,
        co_modulation_index     REAL,
        n_paired_observations   INTEGER NOT NULL,
        coverage_ratio          REAL NOT NULL,
        streamer_rmssd_mean     REAL,
        operator_rmssd_mean     REAL,
        created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # --- 05-attribution.sql ------------------------------------------
    # The PostgreSQL regex CHECKs (segment_id ~ '^[0-9a-f]{64}$',
    # expected_rule_text_hash ~ '^[0-9a-f]{64}$') drop here; Pydantic
    # enforces those patterns on the write path. Enum CHECKs survive.
    """
    CREATE TABLE IF NOT EXISTS attribution_event (
        event_id                    TEXT PRIMARY KEY,
        session_id                  TEXT NOT NULL REFERENCES sessions(session_id),
        segment_id                  TEXT NOT NULL,
        event_type                  TEXT NOT NULL CHECK (event_type IN ('greeting_interaction')),
        event_time_utc              TEXT NOT NULL,
        stimulus_time_utc           TEXT,
        selected_arm_id             TEXT NOT NULL,
        expected_rule_text_hash     TEXT NOT NULL,
        semantic_method             TEXT NOT NULL CHECK (
            semantic_method IN ('cross_encoder', 'llm_gray_band', 'azure_llm_legacy')
        ),
        semantic_method_version     TEXT NOT NULL,
        semantic_p_match            REAL CHECK (
            semantic_p_match IS NULL OR (semantic_p_match BETWEEN 0.0 AND 1.0)
        ),
        semantic_reason_code        TEXT,
        reward_path_version         TEXT NOT NULL,
        bandit_decision_snapshot    TEXT NOT NULL,
        evidence_flags              TEXT NOT NULL DEFAULT '[]',
        finality                    TEXT NOT NULL CHECK (
            finality IN ('online_provisional', 'offline_final')
        ),
        schema_version              TEXT NOT NULL,
        created_at                  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (session_id, segment_id, event_type, reward_path_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcome_event (
        outcome_id          TEXT PRIMARY KEY,
        session_id          TEXT NOT NULL REFERENCES sessions(session_id),
        outcome_type        TEXT NOT NULL CHECK (outcome_type IN ('creator_follow')),
        outcome_value       REAL NOT NULL,
        outcome_time_utc    TEXT NOT NULL,
        source_system       TEXT NOT NULL,
        source_event_ref    TEXT,
        confidence          REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
        finality            TEXT NOT NULL CHECK (
            finality IN ('online_provisional', 'offline_final')
        ),
        schema_version      TEXT NOT NULL,
        created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_outcome_link (
        link_id             TEXT PRIMARY KEY,
        event_id            TEXT NOT NULL REFERENCES attribution_event(event_id),
        outcome_id          TEXT NOT NULL REFERENCES outcome_event(outcome_id),
        lag_s               REAL NOT NULL CHECK (lag_s >= 0.0),
        horizon_s           REAL NOT NULL CHECK (horizon_s > 0.0),
        link_rule_version   TEXT NOT NULL,
        eligibility_flags   TEXT NOT NULL DEFAULT '[]',
        finality            TEXT NOT NULL CHECK (
            finality IN ('online_provisional', 'offline_final')
        ),
        schema_version      TEXT NOT NULL,
        created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CHECK (lag_s <= horizon_s),
        UNIQUE (event_id, outcome_id, link_rule_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attribution_score (
        score_id                TEXT PRIMARY KEY,
        event_id                TEXT NOT NULL REFERENCES attribution_event(event_id),
        outcome_id              TEXT REFERENCES outcome_event(outcome_id),
        attribution_method      TEXT NOT NULL,
        method_version          TEXT NOT NULL,
        score_raw               REAL,
        score_normalized        REAL,
        confidence              REAL CHECK (
            confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)
        ),
        evidence_flags          TEXT NOT NULL DEFAULT '[]',
        finality                TEXT NOT NULL CHECK (
            finality IN ('online_provisional', 'offline_final')
        ),
        schema_version          TEXT NOT NULL,
        created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # --- WS4 P2 — process heartbeats and capture-child manifest ------
    # ``process_heartbeat`` carries one row per v4 desktop process,
    # keyed by canonical process name. Each child writes a 1 Hz
    # ``INSERT OR REPLACE`` from its own short-lived SQLite connection;
    # WAL mode + ``busy_timeout=5000`` absorb the cross-process write
    # contention. The table fuels both the operator console health
    # rollup and the next startup's recovery sweep ("which processes
    # were running last time?").
    """
    CREATE TABLE IF NOT EXISTS process_heartbeat (
        process_name        TEXT PRIMARY KEY,
        pid                 INTEGER NOT NULL,
        started_at_utc      TEXT NOT NULL,
        last_heartbeat_utc  TEXT NOT NULL
    )
    """,
    # ``capture_pid_manifest`` records every external child
    # (scrcpy / adb / ffmpeg) that capture_supervisor spawns through
    # :class:`SupervisedProcess`. Win32 Job Objects auto-clean these
    # on parent crash; POSIX has no kernel equivalent and relies on
    # this manifest for the recovery sweep to terminate orphans on
    # the next startup.
    """
    CREATE TABLE IF NOT EXISTS capture_pid_manifest (
        pid                 INTEGER PRIMARY KEY,
        process_kind        TEXT NOT NULL CHECK (
            process_kind IN ('scrcpy', 'adb', 'ffmpeg')
        ),
        parent_process      TEXT NOT NULL,
        spawned_at_utc      TEXT NOT NULL
    )
    """,
)


# All v3.4 indexes survive verbatim. SQLite supports the same
# CREATE INDEX IF NOT EXISTS syntax.
INDEX_DDL: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_evaluations_session ON evaluations(session_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_lookup ON experiments(experiment_id, arm)",
    "CREATE INDEX IF NOT EXISTS idx_encounter_log_experiment ON encounter_log(experiment_id, arm)",
    "CREATE INDEX IF NOT EXISTS idx_encounter_log_session "
    "ON encounter_log(session_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_physiology_session "
    "ON physiology_log(session_id, subject_role, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_physiology_segment ON physiology_log(session_id, segment_id)",
    "CREATE INDEX IF NOT EXISTS idx_comod_session ON comodulation_log(session_id, window_end_utc)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_event_session ON attribution_event(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_event_segment ON attribution_event(segment_id)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_event_time ON attribution_event(event_time_utc)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_event_finality ON attribution_event(finality)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_event_session ON outcome_event(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_event_time ON outcome_event(outcome_time_utc)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_event_finality ON outcome_event(finality)",
    "CREATE INDEX IF NOT EXISTS idx_event_outcome_link_event ON event_outcome_link(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_outcome_link_outcome ON event_outcome_link(outcome_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_outcome_link_finality ON event_outcome_link(finality)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_score_event ON attribution_score(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_score_outcome ON attribution_score(outcome_id)",
    "CREATE INDEX IF NOT EXISTS idx_attribution_score_finality ON attribution_score(finality)",
    "CREATE INDEX IF NOT EXISTS idx_process_heartbeat_freshness "
    "ON process_heartbeat(last_heartbeat_utc)",
    "CREATE INDEX IF NOT EXISTS idx_capture_pid_manifest_parent "
    "ON capture_pid_manifest(parent_process)",
)


# Local seed for the four §4.E.1 greeting variants. Mirrors
# data/sql/02-seed-experiments.sql verbatim. WS5 P2 will replace this
# block with a cloud-synced ExperimentBundle ingest; until then the
# operator console's Experiments page renders these rows directly.
SEED_EXPERIMENTS: Final[tuple[tuple[str, str, str, str, float, float, int], ...]] = (
    (
        "greeting_line_v1",
        "Greeting Line V1",
        "warm_welcome",
        "Hey! Thanks for streaming, you're awesome!",
        1.0,
        1.0,
        1,
    ),
    (
        "greeting_line_v1",
        "Greeting Line V1",
        "direct_question",
        "Hi! What's the best advice you've gotten today?",
        1.0,
        1.0,
        1,
    ),
    (
        "greeting_line_v1",
        "Greeting Line V1",
        "compliment_content",
        "Love the energy on this stream! How long have you been live?",
        1.0,
        1.0,
        1,
    ),
    (
        "greeting_line_v1",
        "Greeting Line V1",
        "simple_hello",
        "Hello! Just joined, happy to be here!",
        1.0,
        1.0,
        1,
    ),
)


SEED_EXPERIMENTS_INSERT: Final[str] = (
    "INSERT OR IGNORE INTO experiments "
    "(experiment_id, label, arm, greeting_text, alpha_param, beta_param, enabled) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


def _apply_pragmas(conn: sqlite3.Connection, pragmas: Iterable[str]) -> None:
    for pragma in pragmas:
        conn.execute(pragma)


def apply_writer_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the WS4 P1 writer-side PRAGMA bundle to ``conn``."""
    _apply_pragmas(conn, PRAGMAS_WRITER)


def apply_reader_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the WS4 P1 reader-side PRAGMA bundle (incl. ``query_only=1``)."""
    _apply_pragmas(conn, PRAGMAS_READER)


def bootstrap_schema(conn: sqlite3.Connection, *, seed_experiments: bool = True) -> None:
    """Run all DDL + indexes (and optionally the experiments seed) idempotently.

    Caller is responsible for opening ``conn`` in writer mode. The
    transaction is committed on the way out so subsequent reader
    connections observe the schema immediately.
    """
    apply_writer_pragmas(conn)
    for stmt in SCHEMA_DDL:
        conn.execute(stmt)
    for stmt in INDEX_DDL:
        conn.execute(stmt)
    if seed_experiments:
        conn.executemany(SEED_EXPERIMENTS_INSERT, SEED_EXPERIMENTS)
    conn.commit()
    logger.info(
        "sqlite schema bootstrapped: %d tables, %d indexes, seed=%s",
        len(SCHEMA_DDL),
        len(INDEX_DDL),
        seed_experiments,
    )
