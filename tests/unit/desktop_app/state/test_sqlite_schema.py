"""SQLite schema bootstrap and parity tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from services.desktop_app.state.sqlite_schema import (
    PRAGMAS_READER,
    PRAGMAS_WRITER,
    SEED_EXPERIMENTS,
    apply_reader_pragmas,
    apply_writer_pragmas,
    bootstrap_schema,
)

EXPECTED_TABLES: set[str] = {
    "sessions",
    "metrics",
    "transcripts",
    "evaluations",
    "events",
    "experiments",
    "encounter_log",
    "context",
    "physiology_log",
    "comodulation_log",
    "attribution_event",
    "outcome_event",
    "event_outcome_link",
    "attribution_score",
    "pending_uploads",
    "analytics_message_ledger",
    "capture_status",
    "live_session_state",
}

# Subset of v3.4 column names per table that MUST survive the port —
# anything code-path-load-bearing. Full per-column type assertions are
# excessive; this catches accidental drops and renames.
RAW_MEDIA_COLUMN_TOKENS = ("audio", "frame", "image", "video", "voiceprint", "biometric")
RAW_MEDIA_ALLOWED_COLUMNS = {
    ("sessions", "stream_url"),
    ("metrics", "au12_intensity"),
    ("encounter_log", "n_frames_in_window"),
    ("live_session_state", "calibration_frames_accumulated"),
    ("live_session_state", "calibration_frames_required"),
    ("live_session_state", "latest_au12_intensity"),
    ("live_session_state", "latest_au12_timestamp_s"),
    ("pending_uploads", "payload_json"),
    ("pending_uploads", "payload_sha256"),
    ("pending_uploads", "payload_redacted_at_utc"),
    ("outcome_event", "confidence"),
    ("attribution_score", "confidence"),
    ("attribution_score", "attribution_method"),
}

EXPECTED_COLUMNS: dict[str, set[str]] = {
    "sessions": {
        "session_id",
        "stream_url",
        "experiment_id",
        "started_at",
        "ended_at",
    },
    "metrics": {
        "session_id",
        "segment_id",
        "timestamp_utc",
        "au12_intensity",
        "f0_valid_measure",
        "f0_valid_baseline",
        "perturbation_valid_measure",
        "perturbation_valid_baseline",
        "voiced_coverage_measure_s",
        "voiced_coverage_baseline_s",
        "f0_mean_measure_hz",
        "f0_mean_baseline_hz",
        "f0_delta_semitones",
        "jitter_mean_measure",
        "jitter_mean_baseline",
        "jitter_delta",
        "shimmer_mean_measure",
        "shimmer_mean_baseline",
        "shimmer_delta",
    },
    "encounter_log": {
        "session_id",
        "segment_id",
        "experiment_id",
        "arm",
        "timestamp_utc",
        "gated_reward",
        "p90_intensity",
        "semantic_gate",
        "n_frames_in_window",
        "au12_baseline_pre",
        "stimulus_time",
    },
    "physiology_log": {
        "session_id",
        "segment_id",
        "subject_role",
        "rmssd_ms",
        "heart_rate_bpm",
        "freshness_s",
        "is_stale",
        "provider",
        "source_kind",
        "derivation_method",
        "window_s",
        "validity_ratio",
        "is_valid",
        "source_timestamp_utc",
    },
    "comodulation_log": {
        "session_id",
        "window_start_utc",
        "window_end_utc",
        "window_minutes",
        "co_modulation_index",
        "n_paired_observations",
        "coverage_ratio",
        "streamer_rmssd_mean",
        "operator_rmssd_mean",
    },
    "capture_status": {
        "status_key",
        "state",
        "label",
        "detail",
        "operator_action_hint",
        "updated_at_utc",
    },
    "live_session_state": {
        "session_id",
        "active_arm",
        "stimulus_definition",
        "is_calibrating",
        "calibration_frames_accumulated",
        "calibration_frames_required",
        "face_present",
        "latest_au12_intensity",
        "latest_au12_timestamp_s",
        "status",
        "updated_at_utc",
    },
    "attribution_event": {
        "event_id",
        "session_id",
        "segment_id",
        "event_type",
        "event_time_utc",
        "stimulus_time_utc",
        "selected_arm_id",
        "expected_rule_text_hash",
        "semantic_method",
        "semantic_method_version",
        "semantic_p_match",
        "semantic_reason_code",
        "reward_path_version",
        "bandit_decision_snapshot",
        "evidence_flags",
        "finality",
        "schema_version",
    },
    "outcome_event": {
        "outcome_id",
        "session_id",
        "outcome_type",
        "outcome_value",
        "outcome_time_utc",
        "source_system",
        "source_event_ref",
        "confidence",
        "finality",
        "schema_version",
    },
    "event_outcome_link": {
        "link_id",
        "event_id",
        "outcome_id",
        "lag_s",
        "horizon_s",
        "link_rule_version",
        "eligibility_flags",
        "finality",
        "schema_version",
    },
    "attribution_score": {
        "score_id",
        "event_id",
        "outcome_id",
        "attribution_method",
        "method_version",
        "score_raw",
        "score_normalized",
        "confidence",
        "evidence_flags",
        "finality",
        "schema_version",
    },
    "pending_uploads": {
        "upload_id",
        "endpoint",
        "payload_type",
        "dedupe_key",
        "payload_json",
        "payload_sha256",
        "payload_redacted_at_utc",
        "created_at_utc",
        "next_attempt_at_utc",
        "attempt_count",
        "locked_at_utc",
        "last_error",
        "status",
    },
    "analytics_message_ledger": {
        "message_id",
        "segment_id",
        "client_id",
        "arm",
        "processed_at_utc",
    },
}


def _bootstrap(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    bootstrap_schema(conn)
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_bootstrap_creates_every_expected_table(tmp_path: Path) -> None:
    conn = _bootstrap(tmp_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    actual = {row[0] for row in rows}
    assert EXPECTED_TABLES.issubset(actual)
    conn.close()


@pytest.mark.parametrize("table,expected_cols", sorted(EXPECTED_COLUMNS.items()))
def test_table_has_expected_columns(tmp_path: Path, table: str, expected_cols: set[str]) -> None:
    conn = _bootstrap(tmp_path)
    actual = _table_columns(conn, table)
    missing = expected_cols.difference(actual)
    assert not missing, f"{table}: missing columns {sorted(missing)}"
    conn.close()


def test_desktop_schema_has_no_raw_media_persistence_columns(tmp_path: Path) -> None:
    conn = _bootstrap(tmp_path)
    rows = conn.execute(
        """
        SELECT m.name AS table_name, p.name AS column_name
        FROM sqlite_master m
        JOIN pragma_table_info(m.name) p
        WHERE m.type = 'table' AND m.name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    conn.close()

    forbidden = [
        (table_name, column_name)
        for table_name, column_name in rows
        if (table_name, column_name) not in RAW_MEDIA_ALLOWED_COLUMNS
        and any(token in column_name.lower() for token in RAW_MEDIA_COLUMN_TOKENS)
    ]
    assert forbidden == []


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    """Running bootstrap_schema twice must not raise or duplicate seed rows."""
    conn = _bootstrap(tmp_path)
    bootstrap_schema(conn)
    bootstrap_schema(conn)

    rows = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
    assert rows[0] == len(SEED_EXPERIMENTS)
    conn.close()


def test_bootstrap_migrates_existing_pending_uploads_table(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE pending_uploads (
            upload_id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            payload_type TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            next_attempt_at_utc TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            locked_at_utc TEXT,
            last_error TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            UNIQUE (payload_type, dedupe_key)
        )
        """
    )

    bootstrap_schema(conn)

    columns = _table_columns(conn, "pending_uploads")
    assert {"payload_sha256", "payload_redacted_at_utc"}.issubset(columns)
    conn.close()


def test_bootstrap_migrates_existing_live_session_state_and_attribution_tables(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            stream_url TEXT NOT NULL,
            started_at TEXT NOT NULL,
            stimulus_definition TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_session_state (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
            active_arm TEXT,
            is_calibrating INTEGER NOT NULL,
            calibration_frames_accumulated INTEGER NOT NULL,
            calibration_frames_required INTEGER NOT NULL,
            face_present INTEGER NOT NULL,
            latest_au12_intensity REAL,
            latest_au12_timestamp_s REAL,
            status TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attribution_event (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            segment_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_time_utc TEXT NOT NULL,
            stimulus_time_utc TEXT,
            stimulus_modality TEXT,
            selected_arm_id TEXT NOT NULL,
            expected_rule_text_hash TEXT NOT NULL,
            semantic_method TEXT NOT NULL,
            semantic_method_version TEXT NOT NULL,
            semantic_p_match REAL,
            semantic_reason_code TEXT,
            reward_path_version TEXT NOT NULL,
            bandit_decision_snapshot TEXT NOT NULL,
            evidence_flags TEXT NOT NULL DEFAULT '[]',
            finality TEXT NOT NULL,
            schema_version TEXT NOT NULL
        )
        """
    )

    bootstrap_schema(conn)

    assert "stimulus_definition" in _table_columns(conn, "live_session_state")
    attribution_columns = _table_columns(conn, "attribution_event")
    assert {
        "stimulus_id",
        "stimulus_modality",
        "selected_arm_id",
        "expected_rule_text_hash",
        "expected_response_rule_text_hash",
        "semantic_method",
        "semantic_method_version",
        "semantic_p_match",
        "semantic_reason_code",
        "matched_response_time_utc",
        "response_registration_status",
        "response_reason_code",
        "reward_path_version",
        "bandit_decision_snapshot",
        "evidence_flags",
        "schema_version",
        "created_at",
    }.issubset(attribution_columns)
    conn.close()


def test_bootstrap_migrates_existing_experiments_table(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            label TEXT,
            arm TEXT NOT NULL,
            greeting_text TEXT,
            alpha_param REAL NOT NULL DEFAULT 1.0,
            beta_param REAL NOT NULL DEFAULT 1.0,
            UNIQUE (experiment_id, arm)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO experiments (experiment_id, label, arm, greeting_text, alpha_param, beta_param)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "greeting_line_v1",
            "Greeting Line V1",
            "warm_welcome",
            "Say hello to the creator",
            2.0,
            3.0,
        ),
    )

    bootstrap_schema(conn)

    columns = _table_columns(conn, "experiments")
    assert {"stimulus_definition", "enabled", "end_dated_at", "updated_at"}.issubset(columns)
    row = conn.execute(
        "SELECT arm, stimulus_definition, alpha_param, beta_param "
        "FROM experiments WHERE experiment_id = ? AND arm = ?",
        ("greeting_line_v1", "warm_welcome"),
    ).fetchone()
    assert row is not None
    assert row[0] == "warm_welcome"
    assert '"stimulus_modality":"spoken_greeting"' in row[1]
    assert '"text":"Say hello to the creator"' in row[1]
    assert row[2] == pytest.approx(2.0, abs=1e-12)
    assert row[3] == pytest.approx(3.0, abs=1e-12)
    conn.close()


def test_analytics_message_identity_index_is_unique(tmp_path: Path) -> None:
    conn = _bootstrap(tmp_path)
    indexes = conn.execute("PRAGMA index_list(analytics_message_ledger)").fetchall()
    unique_indexes = {row[1] for row in indexes if row[2] == 1}
    assert "idx_analytics_message_identity" in unique_indexes
    columns = conn.execute("PRAGMA index_info(idx_analytics_message_identity)").fetchall()
    assert [row[2] for row in columns] == ["segment_id", "client_id", "arm"]
    conn.close()


def test_seed_experiments_inserts_four_arms(tmp_path: Path) -> None:
    conn = _bootstrap(tmp_path)
    rows = conn.execute(
        "SELECT arm FROM experiments WHERE experiment_id='greeting_line_v1' ORDER BY arm"
    ).fetchall()
    arms = sorted(row[0] for row in rows)
    assert arms == ["compliment_content", "direct_question", "simple_hello", "warm_welcome"]
    conn.close()


def test_writer_pragmas_set_wal(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "x.sqlite"), isolation_level=None)
    apply_writer_pragmas(conn)
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.lower() == "wal"
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert busy_timeout == 5000
    conn.close()


def test_reader_pragmas_enforce_query_only(tmp_path: Path) -> None:
    """A query_only=1 connection must reject INSERTs."""
    bootstrap_conn = _bootstrap(tmp_path)
    bootstrap_conn.close()

    reader = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    apply_reader_pragmas(reader)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.execute(
            "INSERT INTO sessions (session_id, stream_url) VALUES (?, ?)",
            ("00000000-0000-4000-8000-000000000000", "test"),
        )
    reader.close()


def test_pragmas_writer_includes_foreign_keys() -> None:
    """foreign_keys must be enabled on writer connections so REFERENCES work."""
    assert any("foreign_keys=ON" in p for p in PRAGMAS_WRITER)


def test_pragmas_reader_includes_query_only() -> None:
    assert any("query_only=1" in p for p in PRAGMAS_READER)
