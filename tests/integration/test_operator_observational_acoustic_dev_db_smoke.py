"""Manual dev-db smoke for operator encounter observational acoustic hydration.

This integration test is intentionally narrow: it looks for a real encounter
whose canonical §7D payload exhibits asymmetric validity (valid F0, invalid
perturbation) and verifies the operator read path preserves mixed `None` /
non-`None` fields on `observational_acoustic` instead of collapsing the object.

The test is safe in generic CI environments: if PostgreSQL connection details
are absent, unreachable, or no qualifying encounter exists in the target
backend, it skips with the exact blocker.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections.abc import Iterator
from contextlib import suppress
from typing import Any
from uuid import UUID

import pytest

from services.api.db import connection as db_connection
from services.api.services.operator_read_service import OperatorReadService

_CANONICAL_FIELDS: tuple[str, ...] = (
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
)

_CANDIDATE_SQL = """
SELECT
    e.id::text AS encounter_id,
    e.session_id::text AS session_id,
    e.segment_id,
    m.f0_valid_measure,
    m.f0_valid_baseline,
    m.perturbation_valid_measure,
    m.perturbation_valid_baseline,
    m.voiced_coverage_measure_s,
    m.voiced_coverage_baseline_s,
    m.f0_mean_measure_hz,
    m.f0_mean_baseline_hz,
    m.f0_delta_semitones,
    m.jitter_mean_measure,
    m.jitter_mean_baseline,
    m.jitter_delta,
    m.shimmer_mean_measure,
    m.shimmer_mean_baseline,
    m.shimmer_delta
FROM encounter_log e
JOIN LATERAL (
    SELECT
        metrics.id,
        metrics.f0_valid_measure,
        metrics.f0_valid_baseline,
        metrics.perturbation_valid_measure,
        metrics.perturbation_valid_baseline,
        metrics.voiced_coverage_measure_s,
        metrics.voiced_coverage_baseline_s,
        metrics.f0_mean_measure_hz,
        metrics.f0_mean_baseline_hz,
        metrics.f0_delta_semitones,
        metrics.jitter_mean_measure,
        metrics.jitter_mean_baseline,
        metrics.jitter_delta,
        metrics.shimmer_mean_measure,
        metrics.shimmer_mean_baseline,
        metrics.shimmer_delta,
        metrics.created_at
    FROM metrics
    WHERE metrics.session_id = e.session_id
      AND metrics.segment_id = e.segment_id
    ORDER BY metrics.created_at DESC, metrics.id DESC
    LIMIT 1
) m ON TRUE
WHERE m.f0_valid_measure IS TRUE
  AND COALESCE(m.perturbation_valid_measure, FALSE) IS FALSE
  AND m.f0_mean_measure_hz IS NOT NULL
  AND (
      m.jitter_mean_measure IS NULL
      OR m.jitter_mean_baseline IS NULL
      OR m.jitter_delta IS NULL
      OR m.shimmer_mean_measure IS NULL
      OR m.shimmer_mean_baseline IS NULL
      OR m.shimmer_delta IS NULL
  )
ORDER BY e.timestamp_utc DESC, e.created_at DESC
LIMIT 1
"""


def _row_to_dict(cursor: Any) -> dict[str, Any] | None:
    if cursor.description is None:
        return None
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row, strict=True))


@pytest.fixture
def initialized_pool() -> Iterator[None]:
    required_env = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        pytest.skip(f"dev-db smoke blocked: missing env vars {', '.join(sorted(missing))}")

    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError as exc:
        pytest.skip(f"dev-db smoke blocked: cannot reach PostgreSQL at {host}:{port} ({exc})")

    try:
        asyncio.run(db_connection.init_pool(minconn=1, maxconn=2))
        yield
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dev-db smoke blocked during pool init: {exc}")
    finally:
        with suppress(Exception):
            asyncio.run(db_connection.close_pool())


@pytest.mark.integration
def test_operator_read_preserves_asymmetric_observational_acoustic_nullability(
    initialized_pool: None,
) -> None:
    """Real DB smoke: mixed nullability survives through the operator read path."""
    candidate: dict[str, Any] | None = None

    conn = db_connection.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(_CANDIDATE_SQL)
            candidate = _row_to_dict(cursor)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dev-db smoke blocked during candidate query: {exc}")
    finally:
        db_connection.put_connection(conn)

    if candidate is None:
        pytest.skip(
            "dev-db smoke blocked: no encounter found with valid F0 and invalid perturbation"
        )

    service = OperatorReadService()
    encounters = service.list_encounters(UUID(candidate["session_id"]), limit=5000)
    encounter = next(
        (row for row in encounters if row.encounter_id == candidate["encounter_id"]),
        None,
    )
    assert encounter is not None, (
        "candidate encounter was not returned by OperatorReadService.list_encounters"
    )
    assert encounter.segment_timestamp_utc is not None
    assert encounter.observational_acoustic is not None

    observed_payload = encounter.observational_acoustic.model_dump(mode="json")
    expected_payload = {field: candidate[field] for field in _CANONICAL_FIELDS}

    assert observed_payload == expected_payload
    assert observed_payload["f0_valid_measure"] is True
    assert observed_payload["perturbation_valid_measure"] is False
    assert any(observed_payload[field] is None for field in _CANONICAL_FIELDS)
    assert any(observed_payload[field] is not None for field in _CANONICAL_FIELDS)

    print(
        "observed_operator_observational_acoustic=",
        json.dumps(
            {
                "encounter_id": candidate["encounter_id"],
                "session_id": candidate["session_id"],
                "segment_id": candidate["segment_id"],
                "observational_acoustic": observed_payload,
            },
            sort_keys=True,
        ),
    )
