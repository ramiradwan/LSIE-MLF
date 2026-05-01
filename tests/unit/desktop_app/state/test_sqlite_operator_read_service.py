"""WS4 P1b — SqliteOperatorReadService end-to-end smoke tests.

These tests exercise the full path:

  ``OperatorReadService`` public surface  ──▶  SQLite query backend
                                          ──▶  ``SqliteReader`` cursor
                                          ──▶  Pydantic DTO

They verify that the same DTO assembly logic that the FastAPI route
layer drives in production renders correct payloads on top of the
desktop's local SQLite store, and that empty-DB cases still produce
valid (degenerate) DTOs rather than raising.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from packages.schemas.operator_console import (
    EncounterState,
    HealthState,
)
from services.desktop_app.state.sqlite_operator_read_service import (
    SqliteOperatorReadService,
)
from services.desktop_app.state.sqlite_reader import SqliteReader
from services.desktop_app.state.sqlite_writer import SqliteWriter

SESSION_A = UUID("00000000-0000-4000-8000-000000000001")


def _build_service(db: Path, *, now: datetime | None = None) -> SqliteOperatorReadService:
    reader = SqliteReader(db)
    service = SqliteOperatorReadService(reader)
    if now is not None:
        service._clock = lambda: now  # noqa: SLF001 — test injection
    return service


def test_get_overview_with_no_sessions_yields_seed_experiments(tmp_path: Path) -> None:
    """Fresh install: no sessions, but the four seed arms must be reachable."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()

    service = _build_service(db)
    overview = service.get_overview()

    assert overview.active_session is None
    assert overview.latest_encounter is None
    assert overview.experiment_summary is None
    assert overview.physiology is None
    # The freshness rollup over an empty store classifies every
    # subsystem as UNKNOWN, so the overall state degrades to UNKNOWN.
    assert overview.health.overall_state is HealthState.UNKNOWN

    detail = service.get_experiment_detail("greeting_line_v1")
    assert detail is not None
    assert {arm.arm_id for arm in detail.arms} == {
        "warm_welcome",
        "direct_question",
        "compliment_content",
        "simple_hello",
    }


def test_list_sessions_renders_seeded_rows(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://1",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    sessions = service.list_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0].session_id == SESSION_A
    assert sessions[0].status == "active"
    assert sessions[0].started_at_utc == datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def test_list_encounters_carries_reward_explanation(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "test://reward",
                "started_at": "2026-04-01 12:00:00",
            },
        )
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

    service = _build_service(db)
    rows = service.list_encounters(SESSION_A, limit=10, before_utc=None)
    assert len(rows) == 1
    enc = rows[0]
    assert enc.state is EncounterState.COMPLETED
    assert enc.gated_reward == pytest.approx(0.42)
    assert enc.p90_intensity == pytest.approx(0.6)
    assert enc.semantic_gate == 1
    assert enc.n_frames_in_window == 30
    assert enc.au12_baseline_pre == pytest.approx(0.05)
    # No metrics / attribution rows yet → optional summaries omitted.
    assert enc.observational_acoustic is None
    assert enc.semantic_evaluation is None
    assert enc.attribution is None


def test_get_session_physiology_renders_null_comod(tmp_path: Path) -> None:
    """§7C null-valid co-modulation must surface through the SQLite path."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    snap = service.get_session_physiology(SESSION_A)
    assert snap is not None
    assert snap.streamer is None
    assert snap.operator is None
    assert snap.comodulation.co_modulation_index is None
    assert snap.comodulation.null_reason == "no co-modulation window computed yet"


def test_get_health_uses_no_op_probes(tmp_path: Path) -> None:
    """Desktop runtime ships no Postgres/Redis; probes must be skipped."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()

    service = _build_service(db)
    snapshot = asyncio.run(service.get_health())
    assert snapshot.subsystem_probes == {}
    # Fresh DB → all subsystems classified UNKNOWN by §12 freshness.
    assert all(row.state is HealthState.UNKNOWN for row in snapshot.subsystems)


def test_get_overview_with_encounter_renders_experiment(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
            },
        )
        writer.enqueue(
            "encounter_log",
            {
                "session_id": str(SESSION_A),
                "segment_id": "b" * 64,
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "timestamp_utc": "2026-04-01 12:01:00",
                "gated_reward": 1.0,
                "p90_intensity": 0.7,
                "semantic_gate": 1,
                "n_frames_in_window": 30,
            },
        )
        writer.flush()
    finally:
        writer.close()

    # Pin "now" to a time inside the stale-physiology hour-window so the
    # default §12 freshness rollup classifies the encounter row as fresh.
    service = _build_service(db, now=datetime(2026, 4, 1, 12, 1, 1, tzinfo=UTC))
    overview = service.get_overview()

    assert overview.active_session is not None
    assert overview.active_session.session_id == SESSION_A
    assert overview.latest_encounter is not None
    assert overview.latest_encounter.gated_reward == pytest.approx(1.0)
    assert overview.experiment_summary is not None
    assert overview.experiment_summary.experiment_id == "greeting_line_v1"
    assert overview.experiment_summary.active_arm_id == "warm_welcome"


def test_list_alerts_returns_session_ended_alert(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        # ended_at within the last hour to satisfy the SQL filter.
        ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
                "ended_at": ended_at,
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    alerts = service.list_alerts(limit=10, since_utc=None)
    assert len(alerts) == 1
    assert alerts[0].kind.value == "session_ended"
    assert alerts[0].session_id == SESSION_A


def test_since_utc_filter_excludes_old_alerts(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        writer.enqueue(
            "sessions",
            {
                "session_id": str(SESSION_A),
                "stream_url": "x",
                "started_at": "2026-04-01 12:00:00",
                "ended_at": ended_at,
            },
        )
        writer.flush()
    finally:
        writer.close()

    service = _build_service(db)
    far_future = datetime.now(UTC) + timedelta(days=1)
    alerts = service.list_alerts(limit=10, since_utc=far_future)
    assert alerts == []
