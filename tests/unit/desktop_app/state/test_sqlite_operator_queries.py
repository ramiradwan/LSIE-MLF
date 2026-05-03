"""SQLite-flavored operator query layer tests.

Each test bootstraps a fresh SQLite store via :class:`SqliteWriter`,
optionally seeds rows through the writer's enqueue/flush path, then
opens a :class:`sqlite3.Connection` cursor scoped to the same file
(query-only) and exercises one of the public ``fetch_*`` functions.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from services.desktop_app.state import sqlite_operator_queries as q
from services.desktop_app.state.sqlite_reader import SqliteReader
from services.desktop_app.state.sqlite_writer import SqliteWriter

SESSION_A = UUID("00000000-0000-4000-8000-000000000001")
SESSION_B = UUID("00000000-0000-4000-8000-000000000002")


@contextmanager
def _cursor(reader: SqliteReader) -> Iterator[sqlite3.Cursor]:
    with reader.connection() as conn:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()


def _bootstrap(tmp_path: Path) -> Path:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    return db


def _seed_session(
    writer: SqliteWriter,
    session_id: UUID,
    *,
    stream_url: str,
    started_at: str,
    ended_at: str | None = None,
) -> None:
    payload: dict[str, str] = {
        "session_id": str(session_id),
        "stream_url": stream_url,
        "started_at": started_at,
    }
    if ended_at is not None:
        payload["ended_at"] = ended_at
    writer.enqueue("sessions", payload)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_fetch_recent_sessions_empty_db(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        assert q.fetch_recent_sessions(cur, limit=10) == []


def test_fetch_recent_sessions_orders_newest_first(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="older", started_at="2026-01-01 00:00:00")
        _seed_session(writer, SESSION_B, stream_url="newer", started_at="2026-05-01 00:00:00")
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_recent_sessions(cur, limit=10)
    assert [row["session_id"] for row in rows] == [str(SESSION_B), str(SESSION_A)]
    assert rows[0]["last_segment_completed_at_utc"] is None
    assert rows[0]["latest_reward"] is None


def test_fetch_session_by_id_returns_row(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        row = q.fetch_session_by_id(cur, SESSION_A)
    assert row is not None
    assert row["session_id"] == str(SESSION_A)
    assert row["ended_at"] is None
    # duration_s computed from started_at to NOW()
    assert isinstance(row["duration_s"], float)
    assert row["duration_s"] >= 0.0


def test_fetch_session_by_id_unknown_returns_none(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        assert q.fetch_session_by_id(cur, SESSION_A) is None


def test_fetch_active_session_picks_unended(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(
            writer,
            SESSION_A,
            stream_url="ended",
            started_at="2026-01-01 00:00:00",
            ended_at="2026-01-01 01:00:00",
        )
        _seed_session(writer, SESSION_B, stream_url="active", started_at="2026-05-01 00:00:00")
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        row = q.fetch_active_session(cur)
    assert row is not None
    assert row["session_id"] == str(SESSION_B)


def test_fetch_active_session_none_when_all_ended(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(
            writer,
            SESSION_A,
            stream_url="ended",
            started_at="2026-01-01 00:00:00",
            ended_at="2026-01-01 01:00:00",
        )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        assert q.fetch_active_session(cur) is None


# ---------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------


def test_fetch_session_encounters_empty(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        assert q.fetch_session_encounters(cur, SESSION_A, limit=10, before_utc=None) == []


def test_fetch_session_encounters_returns_reward_explanation(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        writer.enqueue(
            "encounter_log",
            {
                "session_id": str(SESSION_A),
                "segment_id": "a" * 64,
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": "2026-04-01 12:01:00",
                "gated_reward": 0.42,
                "p90_intensity": 0.6,
                "semantic_gate": 1,
                "n_frames_in_window": 30,
                "au12_baseline_pre": 0.05,
                "stimulus_time": 1714737660.0,
            },
        )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_session_encounters(cur, SESSION_A, limit=10, before_utc=None)

    assert len(rows) == 1
    row = rows[0]
    assert row["arm"] == "warm_welcome"
    assert row["gated_reward"] == pytest.approx(0.42)
    assert row["semantic_gate"] == 1
    assert row["n_frames_in_window"] == 30
    # acoustic + attribution lateral joins resolve to NULL when the
    # corresponding rows do not exist yet.
    assert row["metrics_row_id"] is None
    assert row["semantic_method"] is None
    assert row["soft_reward_candidate"] is None


def test_fetch_session_encounters_filters_before_utc(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        for ts in ("2026-04-01 12:00:30", "2026-04-01 12:01:00", "2026-04-01 12:01:30"):
            writer.enqueue(
                "encounter_log",
                {
                    "session_id": str(SESSION_A),
                    "segment_id": ts.replace(" ", "_") + "_" + "a" * 32,
                    "experiment_id": "greeting_line_v1",
                    "arm": "warm_welcome",
                    "timestamp_utc": ts,
                    "gated_reward": 0.0,
                    "p90_intensity": 0.0,
                    "semantic_gate": 0,
                    "n_frames_in_window": 0,
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    cutoff = datetime(2026, 4, 1, 12, 1, 0, tzinfo=UTC)
    with _cursor(reader) as cur:
        rows = q.fetch_session_encounters(cur, SESSION_A, limit=10, before_utc=cutoff)
    assert len(rows) == 1
    assert rows[0]["timestamp_utc"] == "2026-04-01 12:00:30"


def test_fetch_latest_encounter_returns_newest(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        for ts in ("2026-04-01 12:00:30", "2026-04-01 12:01:30"):
            writer.enqueue(
                "encounter_log",
                {
                    "session_id": str(SESSION_A),
                    "segment_id": ts.replace(" ", "_") + "_" + "b" * 32,
                    "experiment_id": "greeting_line_v1",
                    "arm": "warm_welcome",
                    "timestamp_utc": ts,
                    "gated_reward": 1.0 if ts.endswith("30") and "01:" in ts else 0.0,
                    "p90_intensity": 0.7,
                    "semantic_gate": 1,
                    "n_frames_in_window": 30,
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        row = q.fetch_latest_encounter(cur, SESSION_A)
    assert row is not None
    assert row["timestamp_utc"] == "2026-04-01 12:01:30"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def test_fetch_experiment_arms_returns_seed_with_rollup_nulls(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_experiment_arms(cur, "greeting_line_v1")
    assert len(rows) == 4
    # No encounter_log rows yet, so the rollup LEFT JOIN gives NULL
    # selection_count / recent_reward_mean / recent_semantic_pass_rate.
    for row in rows:
        assert row["selection_count"] is None
        assert row["recent_reward_mean"] is None
        assert row["recent_semantic_pass_rate"] is None
        assert row["enabled"] == 1
        assert row["greeting_text"]


def test_fetch_experiment_arms_aggregates_rollup(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        for i, gate in enumerate((1, 0)):
            writer.enqueue(
                "encounter_log",
                {
                    "session_id": str(SESSION_A),
                    "segment_id": f"{i:064d}",
                    "experiment_id": "greeting_line_v1",
                    "arm": "warm_welcome",
                    "timestamp_utc": f"2026-04-01 12:0{i}:00",
                    "gated_reward": float(gate),
                    "p90_intensity": 0.5,
                    "semantic_gate": gate,
                    "n_frames_in_window": 30,
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_experiment_arms(cur, "greeting_line_v1")
    by_arm = {row["arm"]: row for row in rows}
    warm = by_arm["warm_welcome"]
    assert warm["selection_count"] == 2
    assert warm["recent_reward_mean"] == pytest.approx(0.5)
    assert warm["recent_semantic_pass_rate"] == pytest.approx(0.5)


def test_fetch_active_arm_for_experiment(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        # No encounter rows → no active arm.
        assert q.fetch_active_arm_for_experiment(cur, "greeting_line_v1") is None


# ---------------------------------------------------------------------------
# Physiology
# ---------------------------------------------------------------------------


def test_fetch_latest_physiology_rows_picks_one_per_role(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        # Older + newer streamer rows; only operator's single row.
        for ts, role, rmssd in (
            ("2026-04-01 12:00:00", "streamer", 30.0),
            ("2026-04-01 12:01:00", "streamer", 32.0),
            ("2026-04-01 12:00:30", "operator", 40.0),
        ):
            writer.enqueue(
                "physiology_log",
                {
                    "session_id": str(SESSION_A),
                    "segment_id": "p" * 64,
                    "subject_role": role,
                    "rmssd_ms": rmssd,
                    "heart_rate_bpm": 70,
                    "freshness_s": 1.0,
                    "is_stale": 0,
                    "provider": "oura",
                    "source_kind": "ibi",
                    "derivation_method": "rmssd_v1",
                    "window_s": 300,
                    "validity_ratio": 1.0,
                    "is_valid": 1,
                    "source_timestamp_utc": ts,
                    "created_at": ts,
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_latest_physiology_rows(cur, SESSION_A)
    by_role = {row["subject_role"]: row for row in rows}
    assert set(by_role.keys()) == {"streamer", "operator"}
    assert by_role["streamer"]["rmssd_ms"] == pytest.approx(32.0)
    assert by_role["operator"]["rmssd_ms"] == pytest.approx(40.0)


def test_fetch_latest_comodulation_row_returns_none_when_empty(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        assert q.fetch_latest_comodulation_row(cur, SESSION_A) is None


def test_fetch_latest_comodulation_row_picks_newest(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        for end_ts, idx in (
            ("2026-04-01 12:01:00", 0.1),
            ("2026-04-01 12:11:00", 0.55),
        ):
            writer.enqueue(
                "comodulation_log",
                {
                    "session_id": str(SESSION_A),
                    "window_start_utc": "2026-04-01 12:00:00",
                    "window_end_utc": end_ts,
                    "window_minutes": 10,
                    "co_modulation_index": idx,
                    "n_paired_observations": 8,
                    "coverage_ratio": 0.8,
                    "streamer_rmssd_mean": 32.0,
                    "operator_rmssd_mean": 40.0,
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        row = q.fetch_latest_comodulation_row(cur, SESSION_A)
    assert row is not None
    assert row["co_modulation_index"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Health heuristics
# ---------------------------------------------------------------------------


def test_fetch_subsystem_pulse_empty_returns_nulls(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        pulse = q.fetch_subsystem_pulse(cur)
    assert pulse == {
        "last_metric_at": None,
        "last_physio_at": None,
        "last_comod_at": None,
        "last_encounter_at": None,
    }


def test_fetch_recent_stale_physiology_filters_is_stale(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        _seed_session(writer, SESSION_A, stream_url="x", started_at="2026-04-01 12:00:00")
        for ts, stale in (
            ("2026-04-01 12:00:00", 0),
            ("2026-04-01 12:01:00", 1),
        ):
            writer.enqueue(
                "physiology_log",
                {
                    "session_id": str(SESSION_A),
                    "segment_id": "p" * 64,
                    "subject_role": "streamer",
                    "rmssd_ms": 30.0,
                    "heart_rate_bpm": 70,
                    "freshness_s": 12.5,
                    "is_stale": stale,
                    "provider": "oura",
                    "source_kind": "ibi",
                    "derivation_method": "rmssd_v1",
                    "window_s": 300,
                    "validity_ratio": 1.0,
                    "is_valid": 1,
                    "source_timestamp_utc": ts,
                    # `created_at` defaults to CURRENT_TIMESTAMP so the
                    # `>= datetime('now', '-1 hour')` filter keeps both rows.
                },
            )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_recent_stale_physiology(cur, since_utc=None, limit=10)
    assert len(rows) == 1
    assert rows[0]["freshness_s"] == pytest.approx(12.5)


def test_fetch_recent_stale_physiology_respects_since_utc(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    far_future = datetime.now(UTC) + timedelta(days=1)
    with _cursor(reader) as cur:
        rows = q.fetch_recent_stale_physiology(cur, since_utc=far_future, limit=10)
    assert rows == []


def test_fetch_recently_ended_sessions_returns_only_ended(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        # Use SQLite's CURRENT_TIMESTAMP-style format and pin ended_at to "now"
        # so the `>= datetime('now','-1 hour')` filter keeps the row.
        now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        _seed_session(
            writer,
            SESSION_A,
            stream_url="ended-now",
            started_at="2026-01-01 00:00:00",
            ended_at=now_utc,
        )
        _seed_session(writer, SESSION_B, stream_url="open", started_at=now_utc)
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    with _cursor(reader) as cur:
        rows = q.fetch_recently_ended_sessions(cur, since_utc=None, limit=10)
    assert len(rows) == 1
    assert rows[0]["session_id"] == str(SESSION_A)
