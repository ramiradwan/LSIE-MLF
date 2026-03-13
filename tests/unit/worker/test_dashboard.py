"""
Tests for services/worker/dashboard.py — Phase 7.3 validation.

Verifies dashboard module structure against:
  §4.E.1 — Operational metrics visualization
  §11 — Variable Extraction Matrix queries
  §2 step 7 — Parameterized queries
"""

from __future__ import annotations

from services.worker.dashboard import (
    _ACOUSTIC_SQL,
    _AU12_SQL,
    _EXPERIMENTS_SQL,
    _SESSIONS_SQL,
    _get_db_dsn,
)


class TestDashboardQueries:
    """§2 step 7 — Parameterized query validation."""

    def test_sessions_sql_has_left_join(self) -> None:
        """Session overview includes metric counts."""
        assert "LEFT JOIN metrics" in _SESSIONS_SQL
        assert "COUNT" in _SESSIONS_SQL

    def test_au12_sql_parameterized(self) -> None:
        """§2 step 7 — AU12 query uses parameterized session_id."""
        assert "%(session_id)s" in _AU12_SQL
        assert "au12_intensity" in _AU12_SQL

    def test_acoustic_sql_parameterized(self) -> None:
        """§2 step 7 — Acoustic query uses parameterized session_id."""
        assert "%(session_id)s" in _ACOUSTIC_SQL
        assert "pitch_f0" in _ACOUSTIC_SQL
        assert "jitter" in _ACOUSTIC_SQL
        assert "shimmer" in _ACOUSTIC_SQL

    def test_experiments_sql_queries_arms(self) -> None:
        """§4.E.1 — Experiments query includes alpha/beta parameters."""
        assert "alpha_param" in _EXPERIMENTS_SQL
        assert "beta_param" in _EXPERIMENTS_SQL
        assert "experiment_id" in _EXPERIMENTS_SQL

    def test_au12_ordered_by_timestamp(self) -> None:
        """§11 — AU12 time-series ordered ascending."""
        assert "ORDER BY timestamp_utc ASC" in _AU12_SQL

    def test_acoustic_ordered_by_timestamp(self) -> None:
        """§11 — Acoustic time-series ordered ascending."""
        assert "ORDER BY timestamp_utc ASC" in _ACOUSTIC_SQL


class TestGetDbDsn:
    """Database DSN construction."""

    def test_dsn_format(self) -> None:
        """DSN contains required PostgreSQL parameters."""
        dsn = _get_db_dsn()
        assert "host=" in dsn
        assert "port=" in dsn
        assert "dbname=" in dsn
        assert "user=" in dsn
        assert "password=" in dsn

    def test_dsn_defaults(self) -> None:
        """DSN uses default values when env vars not set."""
        dsn = _get_db_dsn()
        assert "host=postgres" in dsn
        assert "port=5432" in dsn
