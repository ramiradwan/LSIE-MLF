from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import SupportsFloat, cast

import pytest

from packages.schemas.cloud import PosteriorDelta
from services.cloud_api.db import schema
from services.cloud_api.repos import telemetry

SEGMENT_ID = "b" * 64
DECISION_CONTEXT_HASH = "c" * 64


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
