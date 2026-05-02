from __future__ import annotations

from services.cloud_api.db import schema
from services.cloud_api.repos import telemetry


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
    assert "ON CONFLICT (segment_id) DO NOTHING" in telemetry._INSERT_SEGMENT_SQL
    assert "ON CONFLICT (segment_id, client_id, arm_id) DO NOTHING" in (
        telemetry._INSERT_POSTERIOR_DELTA_SQL
    )


def test_cloud_sync_sql_contains_idempotent_tables_and_constraints() -> None:
    assert "CREATE TABLE IF NOT EXISTS segment_telemetry" in schema.SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS posterior_delta_log" in schema.SCHEMA_SQL
    assert "UNIQUE (segment_id, client_id, arm_id)" in schema.SCHEMA_SQL
    assert "decision_context_hash TEXT CHECK" in schema.SCHEMA_SQL
