from __future__ import annotations

import uuid
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import SupportsFloat, cast

import pytest

from packages.schemas.cloud import PosteriorDelta
from services.cloud_api.db import bootstrap, connection, schema
from services.cloud_api.repos import telemetry

SEGMENT_ID = "b" * 64
DECISION_CONTEXT_HASH = "c" * 64


class ReadinessCursor:
    def __init__(self, *, fail: bool = False) -> None:
        self.executed_sql: list[str] = []
        self.fail = fail

    def __enter__(self) -> ReadinessCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def execute(self, sql: str) -> None:
        if self.fail:
            raise RuntimeError("postgres://user:secret@db/app SELECT private_table")
        self.executed_sql.append(sql)

    def fetchone(self) -> tuple[int]:
        return (1,)


class ReadinessConnection:
    def __init__(self, *, fail: bool = False) -> None:
        self.cursor_instance = ReadinessCursor(fail=fail)

    def cursor(self) -> ReadinessCursor:
        return self.cursor_instance


class ReadinessPool:
    def __init__(self, conn: ReadinessConnection) -> None:
        self.conn = conn
        self.returned: list[ReadinessConnection] = []

    def getconn(self) -> ReadinessConnection:
        return self.conn

    def putconn(self, conn: ReadinessConnection) -> None:
        self.returned.append(conn)


class BootstrapCursor:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.executed_sql: list[str] = []
        self.closed = False
        self.fail_on = fail_on

    def execute(self, sql: str) -> None:
        if sql == self.fail_on:
            raise RuntimeError("postgres://user:secret@db/app CREATE TABLE private")
        self.executed_sql.append(sql)

    def close(self) -> None:
        self.closed = True


class BootstrapConnection:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.cursor_instance = BootstrapCursor(fail_on=fail_on)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> BootstrapCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class BootstrapPool:
    def __init__(self, conn: BootstrapConnection) -> None:
        self.conn = conn
        self.returned: list[BootstrapConnection] = []
        self.closed = False

    def getconn(self) -> BootstrapConnection:
        return self.conn

    def putconn(self, conn: BootstrapConnection) -> None:
        self.returned.append(conn)

    def closeall(self) -> None:
        self.closed = True


class PosteriorCursor:
    def __init__(
        self,
        *,
        update_matches: bool = True,
        stored_client_id: str = "desktop-a",
        stored_experiment_id: int = 101,
        stored_arm_id: str = "arm_a",
        stored_decision_context_hash: str = DECISION_CONTEXT_HASH,
        segment_exists: bool = True,
    ) -> None:
        self.rowcount = 0
        self._applied_keys: set[tuple[str, str, str]] = set()
        self.alpha = 1.0
        self.beta = 1.0
        self.update_matches = update_matches
        self.segment_exists = segment_exists
        self._selected_row: tuple[object, object] | None = None
        self._segment_payload = {
            "_bandit_decision_snapshot": {
                "experiment_id": stored_experiment_id,
                "selected_arm_id": stored_arm_id,
                "decision_context_hash": stored_decision_context_hash,
            }
        }
        self._stored_client_id = stored_client_id

    def execute(self, sql: str, params: dict[str, str | SupportsFloat | None]) -> None:
        if "SELECT payload, client_id" in sql:
            self.rowcount = 1 if self.segment_exists else 0
            if self.segment_exists:
                self._selected_row = (self._segment_payload, self._stored_client_id)
            else:
                self._selected_row = None
            return
        key = (str(params["segment_id"]), str(params["client_id"]), str(params["arm_id"]))
        if "INSERT INTO posterior_delta_log" in sql:
            if key in self._applied_keys:
                self.rowcount = 0
                return
            self._applied_keys.add(key)
            self.rowcount = 1
            return
        if "UPDATE experiments" in sql:
            if not self.update_matches:
                self.rowcount = 0
                return
            self.alpha += float(cast(SupportsFloat, params["delta_alpha"]))
            self.beta += float(cast(SupportsFloat, params["delta_beta"]))
            self.rowcount = 1
            return
        raise AssertionError(sql)

    def fetchone(self) -> tuple[object, object] | None:
        return self._selected_row


def _write_sql_file(path: Path, sql: str) -> Path:
    path.write_text(sql, encoding="utf-8")
    return path


def _posterior_delta(
    *,
    segment_id: str = SEGMENT_ID,
    client_id: str = "desktop-a",
    arm_id: str = "arm_a",
    experiment_id: int = 101,
    delta_alpha: float = 0.0,
    delta_beta: float = 0.0,
    decision_context_hash: str = DECISION_CONTEXT_HASH,
) -> PosteriorDelta:
    return PosteriorDelta(
        experiment_id=experiment_id,
        arm_id=arm_id,
        delta_alpha=delta_alpha,
        delta_beta=delta_beta,
        segment_id=segment_id,
        client_id=client_id,
        event_id=uuid.uuid4(),
        applied_at_utc=datetime.now(UTC),
        decision_context_hash=decision_context_hash,
    )


def test_sql_bootstrap_files_are_deterministic_and_end_with_cloud_sync() -> None:
    names = tuple(path.name for path in schema.SQL_BOOTSTRAP_FILES)

    assert names == (
        "01-schema.sql",
        "02-seed-experiments.sql",
        "03-encounter-log.sql",
        "03-physiology.sql",
        "04-metrics-observational-acoustics.sql",
        "05-attribution.sql",
        "06-cloud-sync.sql",
    )
    assert names[-1] == "06-cloud-sync.sql"


def test_telemetry_repository_uses_cloud_table_names() -> None:
    assert "INSERT INTO segment_telemetry" in telemetry._INSERT_SEGMENT_SQL
    assert "INSERT INTO posterior_delta_log" in telemetry._INSERT_POSTERIOR_DELTA_SQL
    assert "INSERT INTO attribution_event" in telemetry._INSERT_ATTRIBUTION_EVENT_SQL
    assert "SELECT payload, client_id" in telemetry._SELECT_SEGMENT_AUTH_SQL
    assert "ON CONFLICT (segment_id) DO NOTHING" in telemetry._INSERT_SEGMENT_SQL
    assert "ON CONFLICT DO NOTHING" in telemetry._INSERT_POSTERIOR_DELTA_SQL
    assert "UPDATE experiments" in telemetry._APPLY_POSTERIOR_DELTA_SQL
    assert "alpha_param = alpha_param + %(delta_alpha)s" in telemetry._APPLY_POSTERIOR_DELTA_SQL
    assert "beta_param = beta_param + %(delta_beta)s" in telemetry._APPLY_POSTERIOR_DELTA_SQL


def test_cloud_sync_sql_contains_idempotent_tables_and_constraints() -> None:
    assert "CREATE TABLE IF NOT EXISTS segment_telemetry" in schema.SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS posterior_delta_log" in schema.SCHEMA_SQL
    assert "UNIQUE (segment_id, client_id, arm_id)" in schema.SCHEMA_SQL
    assert "decision_context_hash TEXT NOT NULL CHECK" in schema.SCHEMA_SQL


def test_bootstrap_applies_sql_files_in_order_and_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_sql_file(tmp_path / "01-first.sql", "CREATE TABLE first")
    second = _write_sql_file(tmp_path / "02-second.sql", "CREATE TABLE second")
    conn = BootstrapConnection()
    pool = BootstrapPool(conn)
    monkeypatch.setattr(connection, "_pool", pool)

    applied = bootstrap.apply_bootstrap_files((first, second))

    assert applied == ("01-first.sql", "02-second.sql")
    assert conn.cursor_instance.executed_sql == ["CREATE TABLE first", "CREATE TABLE second"]
    assert conn.cursor_instance.closed is True
    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert pool.returned == [conn]


def test_bootstrap_rolls_back_and_returns_connection_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_sql_file(tmp_path / "01-first.sql", "CREATE TABLE first")
    second = _write_sql_file(tmp_path / "02-second.sql", "CREATE TABLE second")
    conn = BootstrapConnection(fail_on="CREATE TABLE second")
    pool = BootstrapPool(conn)
    monkeypatch.setattr(connection, "_pool", pool)

    with pytest.raises(RuntimeError, match="secret"):
        bootstrap.apply_bootstrap_files((first, second))

    assert conn.cursor_instance.executed_sql == ["CREATE TABLE first"]
    assert conn.cursor_instance.closed is True
    assert conn.commits == 0
    assert conn.rollbacks == 1
    assert pool.returned == [conn]


@pytest.mark.asyncio
async def test_bootstrap_entrypoint_emits_bounded_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql_file = _write_sql_file(tmp_path / "01-schema.sql", "CREATE TABLE bootstrap_success")
    conn = BootstrapConnection()
    pool = BootstrapPool(conn)
    stdout = StringIO()
    stderr = StringIO()

    async def init_test_pool(*, minconn: int, maxconn: int) -> None:
        assert minconn == 1
        assert maxconn == 1
        monkeypatch.setattr(connection, "_pool", pool)

    monkeypatch.setattr(bootstrap, "SQL_BOOTSTRAP_FILES", (sql_file,))
    monkeypatch.setattr(connection, "init_pool", init_test_pool)

    exit_code = await bootstrap.run_bootstrap(stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stdout.getvalue() == "cloud-db-bootstrap status=ok files_applied=1\n"
    assert stderr.getvalue() == ""
    assert pool.closed is True


@pytest.mark.asyncio
async def test_bootstrap_entrypoint_emits_bounded_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql_file = _write_sql_file(tmp_path / "01-schema.sql", "CREATE TABLE bootstrap_failure")
    conn = BootstrapConnection(fail_on="CREATE TABLE bootstrap_failure")
    pool = BootstrapPool(conn)
    stdout = StringIO()
    stderr = StringIO()

    async def init_test_pool(*, minconn: int, maxconn: int) -> None:
        del minconn, maxconn
        monkeypatch.setattr(connection, "_pool", pool)

    monkeypatch.setattr(bootstrap, "SQL_BOOTSTRAP_FILES", (sql_file,))
    monkeypatch.setattr(connection, "init_pool", init_test_pool)

    exit_code = await bootstrap.run_bootstrap(stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "cloud-db-bootstrap status=failed\n"
    assert "secret" not in stderr.getvalue()
    assert "CREATE TABLE" not in stderr.getvalue()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_readiness_helper_uses_existing_pool_and_returns_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = ReadinessConnection()
    pool = ReadinessPool(conn)
    monkeypatch.setattr(connection, "_pool", pool)

    ready = await connection.check_readiness()

    assert ready is True
    assert conn.cursor_instance.executed_sql == ["SELECT 1"]
    assert pool.returned == [conn]


@pytest.mark.asyncio
async def test_readiness_helper_returns_false_for_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = ReadinessConnection(fail=True)
    pool = ReadinessPool(conn)
    monkeypatch.setattr(connection, "_pool", pool)

    ready = await connection.check_readiness()

    assert ready is False
    assert pool.returned == [conn]


@pytest.mark.asyncio
async def test_readiness_helper_returns_false_without_pool() -> None:
    await connection.close_pool()

    assert await connection.check_readiness() is False


def test_posterior_delta_duplicate_applies_once() -> None:
    cur = PosteriorCursor()
    delta = _posterior_delta(delta_alpha=0.25, delta_beta=0.75)

    inserted = telemetry.insert_posterior_delta_batch(
        cur,
        [delta, delta],
        authenticated_client_id="desktop-a",
    )

    assert inserted == 1
    assert cur.alpha == pytest.approx(1.25)
    assert cur.beta == pytest.approx(1.75)


def test_posterior_delta_missing_arm_fails_without_reporting_insert() -> None:
    cur = PosteriorCursor(update_matches=False)
    delta = _posterior_delta(delta_alpha=0.25, delta_beta=0.75)

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="arm_a"):
        telemetry.insert_posterior_delta_batch(cur, [delta], authenticated_client_id="desktop-a")

    assert cur.alpha == pytest.approx(1.0)
    assert cur.beta == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("a" * 64, "c" * 64),
        ("c" * 64, "a" * 64),
    ],
)
def test_posterior_delta_application_is_commutative(first: str, second: str) -> None:
    cur = PosteriorCursor()

    inserted = telemetry.insert_posterior_delta_batch(
        cur,
        [
            _posterior_delta(segment_id=first, delta_alpha=0.2, delta_beta=0.3),
            _posterior_delta(segment_id=second, delta_alpha=0.4, delta_beta=0.1),
        ],
        authenticated_client_id="desktop-a",
    )

    assert inserted == 2
    assert cur.alpha == pytest.approx(1.6)
    assert cur.beta == pytest.approx(1.4)


def test_posterior_delta_rejects_unknown_segment_before_mutation() -> None:
    cur = PosteriorCursor(segment_exists=False)

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="segment not found"):
        telemetry.insert_posterior_delta_batch(
            cur,
            [_posterior_delta(delta_alpha=0.2, delta_beta=0.8)],
            authenticated_client_id="desktop-a",
        )

    assert cur.alpha == pytest.approx(1.0)
    assert cur.beta == pytest.approx(1.0)


def test_posterior_delta_rejects_cross_client_segment() -> None:
    cur = PosteriorCursor(stored_client_id="desktop-b")

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="authenticated client"):
        telemetry.insert_posterior_delta_batch(
            cur,
            [_posterior_delta(delta_alpha=0.2, delta_beta=0.8)],
            authenticated_client_id="desktop-a",
        )

    assert cur.alpha == pytest.approx(1.0)
    assert cur.beta == pytest.approx(1.0)


def test_posterior_delta_rejects_experiment_snapshot_mismatch() -> None:
    cur = PosteriorCursor(stored_experiment_id=202)

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="experiment_id"):
        telemetry.insert_posterior_delta_batch(
            cur,
            [_posterior_delta(experiment_id=101, delta_alpha=0.2, delta_beta=0.8)],
            authenticated_client_id="desktop-a",
        )


def test_posterior_delta_rejects_arm_snapshot_mismatch() -> None:
    cur = PosteriorCursor(stored_arm_id="arm_b")

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="arm_id"):
        telemetry.insert_posterior_delta_batch(
            cur,
            [_posterior_delta(arm_id="arm_a", delta_alpha=0.2, delta_beta=0.8)],
            authenticated_client_id="desktop-a",
        )


def test_posterior_delta_rejects_decision_hash_mismatch() -> None:
    cur = PosteriorCursor(stored_decision_context_hash="d" * 64)

    with pytest.raises(telemetry.PosteriorDeltaApplyError, match="decision_context_hash"):
        telemetry.insert_posterior_delta_batch(
            cur,
            [_posterior_delta(decision_context_hash=DECISION_CONTEXT_HASH)],
            authenticated_client_id="desktop-a",
        )
