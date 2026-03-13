"""
Tests for services/worker/pipeline/analytics.py — Phase 2.3–2.4 validation.

Verifies MetricsStore and ThompsonSamplingEngine against:
  §2 step 7 — Parameterized queries, isolation levels
  §4.E.1 — Thompson Sampling with Beta distributions
  §12.1 Module E — Buffer 1000 records, retry 5s, CSV fallback
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Install mock psycopg2 before any analytics imports
_mock_psycopg2 = MagicMock()
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED = 1
_mock_psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE = 6


@pytest.fixture(autouse=True)
def _patch_psycopg2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure psycopg2 mock is in sys.modules for every test."""
    monkeypatch.setitem(sys.modules, "psycopg2", _mock_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", _mock_psycopg2.pool)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", _mock_psycopg2.extensions)


from services.worker.pipeline.analytics import (  # noqa: E402
    DB_BUFFER_MAX,
    MetricsStore,
    ThompsonSamplingEngine,
)


@pytest.fixture()
def mock_conn() -> MagicMock:
    """Create a mock psycopg2 connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture()
def store(mock_conn: MagicMock) -> MetricsStore:
    """MetricsStore with mocked pool returning mock_conn."""
    s = MetricsStore()
    mock_pool = MagicMock()
    s._pool = mock_pool
    s._psycopg2 = _mock_psycopg2
    mock_pool.getconn.return_value = mock_conn
    return s


@pytest.fixture()
def sample_metrics() -> dict[str, Any]:
    """Sample metrics dict matching §2 step 6 payload."""
    return {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "segment_id": "seg-001",
        "timestamp_utc": "2026-03-13T12:00:00+00:00",
        "au12_intensity": 2.5,
        "pitch_f0": 180.0,
        "jitter": 0.02,
        "shimmer": 0.05,
        "transcription": "hello world",
        "semantic": {
            "reasoning": "greeting detected",
            "is_match": True,
            "confidence": 0.95,
        },
    }


class TestMetricsStore:
    """§4.E / §2 step 7 — MetricsStore tests."""

    def test_connect_creates_pool(self) -> None:
        """§2 step 7 — connect() creates ThreadedConnectionPool."""
        s = MetricsStore()
        with patch.dict(
            "os.environ",
            {
                "POSTGRES_USER": "test",
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_DB": "testdb",
            },
        ):
            s.connect()
            assert s._pool is not None

    def test_get_conn_raises_without_connect(self) -> None:
        """_get_conn raises RuntimeError if not connected."""
        s = MetricsStore()
        with pytest.raises(RuntimeError, match="not connected"):
            s._get_conn()

    def test_insert_metrics_executes_parameterized_queries(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
        sample_metrics: dict[str, Any],
    ) -> None:
        """§2 step 7 — Parameterized INSERT for metrics, transcripts, evaluations."""
        store.insert_metrics(sample_metrics)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # 3 inserts: metrics + transcript + evaluation
        assert cursor.execute.call_count == 3
        mock_conn.commit.assert_called_once()

    def test_insert_metrics_sets_read_committed(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
        sample_metrics: dict[str, Any],
    ) -> None:
        """§2 step 7 — READ COMMITTED isolation for metric inserts."""
        store.insert_metrics(sample_metrics)
        mock_conn.set_isolation_level.assert_called_with(
            _mock_psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED
        )

    def test_insert_metrics_only_numeric_when_no_text(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """Only metrics INSERT when no transcription or semantic."""
        minimal = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "seg-001",
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "au12_intensity": 1.0,
        }
        store.insert_metrics(minimal)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count == 1

    def test_insert_metrics_buffers_on_failure(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
        sample_metrics: dict[str, Any],
    ) -> None:
        """§12.1 Module E — Buffer on DB failure."""
        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = (
            _mock_psycopg2.OperationalError("connection lost")
        )
        store.insert_metrics(sample_metrics)
        assert len(store._buffer) == 1
        assert store._buffer[0] is sample_metrics

    def test_buffer_overflow_to_csv(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """§12.4 Module E — CSV fallback when buffer hits 1000."""
        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = (
            _mock_psycopg2.OperationalError("connection lost")
        )
        with patch(
            "services.worker.pipeline.analytics.CSV_FALLBACK_DIR",
            str(tmp_path),
        ):
            # Fill buffer to DB_BUFFER_MAX
            for _ in range(DB_BUFFER_MAX):
                store.insert_metrics(sample_metrics)

            # Buffer should be cleared after overflow
            assert len(store._buffer) == 0
            # CSV file should exist
            csv_files = list(tmp_path.glob("overflow_*.csv"))
            assert len(csv_files) == 1

            # Verify CSV content
            with open(csv_files[0], encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == DB_BUFFER_MAX

    def test_rollback_on_exception(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """Connection rolled back on exception."""
        mock_conn.cursor.return_value.__enter__.return_value.execute.side_effect = (
            _mock_psycopg2.InterfaceError("bad cursor")
        )
        store.insert_metrics({"session_id": "x", "segment_id": "y", "timestamp_utc": "z"})
        mock_conn.rollback.assert_called()

    def test_close(self, store: MetricsStore) -> None:
        """close() calls closeall on pool."""
        pool_ref = store._pool
        store.close()
        pool_ref.closeall.assert_called_once()  # type: ignore[union-attr]
        assert store._pool is None


class TestMetricsStoreExperiments:
    """§4.E.1 — Experiment arm read/write via MetricsStore."""

    def test_get_experiment_arms(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§4.E.1 — Fetches arms with alpha/beta params."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("arm_a", 5.0, 3.0),
            ("arm_b", 2.0, 8.0),
        ]
        arms = store.get_experiment_arms("exp-1")
        assert len(arms) == 2
        assert arms[0] == {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0}
        assert arms[1] == {"arm": "arm_b", "alpha_param": 2.0, "beta_param": 8.0}

    def test_update_experiment_arm_serializable(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§2 step 7 — SERIALIZABLE isolation for experiment updates."""
        store.update_experiment_arm("exp-1", "arm_a", 6.0, 3.0)
        mock_conn.set_isolation_level.assert_called_with(
            _mock_psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE
        )
        mock_conn.commit.assert_called_once()


class TestThompsonSamplingEngine:
    """§4.E.1 — Thompson Sampling tests."""

    @pytest.fixture(autouse=True)
    def _mock_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install mock scipy.stats.beta into sys.modules."""
        import random

        mock_scipy = MagicMock()
        mock_scipy_stats = MagicMock()

        def _mock_beta_rvs(a: float, b: float) -> float:
            """Approximate Beta sampling: higher a/(a+b) → higher sample."""
            mean = a / (a + b)
            return mean + random.uniform(-0.05, 0.05)

        mock_scipy_stats.beta.rvs = _mock_beta_rvs
        monkeypatch.setitem(sys.modules, "scipy", mock_scipy)
        monkeypatch.setitem(sys.modules, "scipy.stats", mock_scipy_stats)

    def test_select_arm_returns_string(self) -> None:
        """§4.E.1 — select_arm returns arm name."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 10.0, "beta_param": 1.0},
            {"arm": "arm_b", "alpha_param": 1.0, "beta_param": 10.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        selected = engine.select_arm("exp-1")
        assert selected in {"arm_a", "arm_b"}

    def test_select_arm_favors_high_alpha(self) -> None:
        """§4.E.1 — Arm with high alpha/low beta should be selected most often."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "winner", "alpha_param": 100.0, "beta_param": 1.0},
            {"arm": "loser", "alpha_param": 1.0, "beta_param": 100.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)

        selections = [engine.select_arm("exp-1") for _ in range(100)]
        winner_count = selections.count("winner")
        # With these extreme params, winner should be selected nearly always
        assert winner_count > 90

    def test_select_arm_empty_raises(self) -> None:
        """select_arm raises ValueError if no arms exist."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = []
        engine = ThompsonSamplingEngine(mock_store)
        with pytest.raises(ValueError, match="No arms found"):
            engine.select_arm("exp-1")

    def test_update_success_increments_alpha(self) -> None:
        """§4.E.1 — reward >= 0.5 increments alpha."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.8)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 6.0, 3.0)

    def test_update_failure_increments_beta(self) -> None:
        """§4.E.1 — reward < 0.5 increments beta."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.2)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 5.0, 4.0)

    def test_update_unknown_arm_raises(self) -> None:
        """update raises ValueError for unknown arm."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 1.0, "beta_param": 1.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        with pytest.raises(ValueError, match="not found"):
            engine.update("exp-1", "arm_x", reward=1.0)

    def test_update_boundary_reward(self) -> None:
        """§4.E.1 — reward == 0.5 counts as success (increments alpha)."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 1.0, "beta_param": 1.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.5)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 2.0, 1.0)
