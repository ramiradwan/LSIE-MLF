"""WS4 P1 — SQLite schema bootstrap + parity tests."""

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
}

# Subset of v3.4 column names per table that MUST survive the port —
# anything code-path-load-bearing. Full per-column type assertions are
# excessive; this catches accidental drops and renames.
EXPECTED_COLUMNS: dict[str, set[str]] = {
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
        "created_at_utc",
        "next_attempt_at_utc",
        "attempt_count",
        "locked_at_utc",
        "last_error",
        "status",
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


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    """Running bootstrap_schema twice must not raise or duplicate seed rows."""
    conn = _bootstrap(tmp_path)
    bootstrap_schema(conn)
    bootstrap_schema(conn)

    rows = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
    assert rows[0] == len(SEED_EXPERIMENTS)
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
