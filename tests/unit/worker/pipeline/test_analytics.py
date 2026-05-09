"""
Tests for services/worker/pipeline/analytics.py — Phase 2.3–2.4 validation.

Verifies MetricsStore and ThompsonSamplingEngine against:
  §2 step 7 — Parameterized queries, isolation levels
  §4.E.1 — Thompson Sampling with Beta distributions
  §12.1 Module E — Buffer 1000 records, retry 5s, CSV fallback
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType
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
def _patch_psycopg2(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Ensure psycopg2 mock and shared attribution buffer are reset per test."""
    monkeypatch.setitem(sys.modules, "psycopg2", _mock_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", _mock_psycopg2.pool)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", _mock_psycopg2.extensions)
    store_cls = globals().get("MetricsStore")
    if store_cls is not None and hasattr(store_cls, "_shared_attribution_buffer"):
        store_cls._shared_attribution_buffer.clear()
    yield
    store_cls = globals().get("MetricsStore")
    if store_cls is not None and hasattr(store_cls, "_shared_attribution_buffer"):
        store_cls._shared_attribution_buffer.clear()


from services.worker.pipeline.analytics import (  # noqa: E402
    _METRICS_DB_FIELDS,
    _METRICS_OVERFLOW_CORE_FIELDS,
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
    """Sample metrics dict matching the canonical Module D → E payload."""
    return {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "segment_id": "seg-001",
        "timestamp_utc": "2026-03-13T12:00:00+00:00",
        "au12_intensity": 2.5,
        "f0_valid_measure": False,
        "f0_valid_baseline": True,
        "perturbation_valid_measure": False,
        "perturbation_valid_baseline": True,
        "voiced_coverage_measure_s": 0.0,
        "voiced_coverage_baseline_s": 1.5,
        "f0_mean_measure_hz": None,
        "f0_mean_baseline_hz": 175.0,
        "f0_delta_semitones": None,
        "jitter_mean_measure": None,
        "jitter_mean_baseline": 0.015,
        "jitter_delta": None,
        "shimmer_mean_measure": None,
        "shimmer_mean_baseline": 0.025,
        "shimmer_delta": None,
        "transcription": "hello world",
        "semantic": {
            "reasoning": "cross_encoder_high_match",
            "is_match": True,
            "confidence_score": 0.95,
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
        metrics_sql, metrics_params = cursor.execute.call_args_list[0].args
        assert "f0_valid_measure" in metrics_sql
        assert tuple(metrics_params.keys()) == _METRICS_DB_FIELDS
        assert metrics_params["f0_valid_measure"] is False
        assert metrics_params["f0_valid_baseline"] is True
        assert metrics_params["f0_mean_measure_hz"] is None
        assert metrics_params["jitter_mean_measure"] is None
        _, evaluation_params = cursor.execute.call_args_list[2].args
        assert evaluation_params["confidence"] == pytest.approx(0.95)
        assert "confidence" not in sample_metrics["semantic"]
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
                assert tuple(reader.fieldnames or ()) == _METRICS_OVERFLOW_CORE_FIELDS
                rows = list(reader)
                assert len(rows) == DB_BUFFER_MAX
                assert rows[0]["f0_valid_measure"] == "false"
                assert rows[0]["f0_valid_baseline"] == "true"
                assert rows[0]["f0_mean_measure_hz"] == "null"
                assert json.loads(rows[0]["semantic"])
                assert json.loads(rows[0]["semantic"]) == sample_metrics["semantic"]

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

    def test_overflow_csv_ignores_unapproved_extra_fields(
        self,
        store: MetricsStore,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Overflow CSV uses only the approved field allowlist."""
        metrics_with_extras = {
            **sample_metrics,
            "raw_audio": [0.1, 0.2, 0.3],
            "voiceprint_embedding": [1.0, 2.0, 3.0],
            "reconstructive_voiceprint": {"coefficients": [0.4, 0.5, 0.6]},
            "unexpected": {"debug": True},
        }

        with patch(
            "services.worker.pipeline.analytics.CSV_FALLBACK_DIR",
            str(tmp_path),
        ):
            store._overflow_to_csv([metrics_with_extras])

        csv_files = list(tmp_path.glob("overflow_*.csv"))
        assert len(csv_files) == 1

        with open(csv_files[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert tuple(reader.fieldnames or ()) == _METRICS_OVERFLOW_CORE_FIELDS
            assert "raw_audio" not in (reader.fieldnames or ())
            assert "voiceprint_embedding" not in (reader.fieldnames or ())
            assert "reconstructive_voiceprint" not in (reader.fieldnames or ())
            rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            assert "raw_audio" not in row
            assert "voiceprint_embedding" not in row
            assert "reconstructive_voiceprint" not in row
            assert "unexpected" not in row
            assert row["session_id"] == sample_metrics["session_id"]
            assert row["transcription"] == sample_metrics["transcription"]
            assert row["f0_valid_measure"] == "false"
            assert row["f0_valid_baseline"] == "true"
            assert row["f0_mean_measure_hz"] == "null"
            assert json.loads(row["semantic"]) == sample_metrics["semantic"]

    def _get_inference_module(self) -> Any:
        """Import the inference task module with the Celery decorator patched out."""
        import importlib

        mock_app = MagicMock()
        mock_app.task.return_value = lambda f: f
        celery_mod = ModuleType("celery")
        celery_mod.Task = object  # type: ignore[attr-defined]
        celery_app_mod = ModuleType("services.worker.celery_app")
        celery_app_mod.celery_app = mock_app  # type: ignore[attr-defined]

        with (
            patch.dict(
                sys.modules,
                {
                    "celery": celery_mod,
                    "services.worker.celery_app": celery_app_mod,
                },
            ),
            patch("services.worker.celery_app.celery_app", mock_app),
        ):
            mod_name = "services.worker.tasks.inference"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            return importlib.import_module(mod_name)

    def _make_inference_payload(self, **overrides: Any) -> dict[str, Any]:
        """Create a minimal Module C -> D payload for end-to-end analytics tests."""
        payload: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "_segment_id": "seg-analytics-001",
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "_audio_data": b"\x00" * 3200,
            "_stimulus_time": 100.0,
            "segments": [],
        }
        payload.update(overrides)
        return payload

    def test_module_d_emits_and_module_e_persists_canonical_acoustic_fields(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """Module D emits canonical acoustics and Module E persists the same values."""
        from packages.ml_core.acoustic import AcousticMetrics

        mod = self._get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        mock_acoustic = MagicMock()
        mock_acoustic.analyze.return_value = AcousticMetrics(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=2.0,
            voiced_coverage_baseline_s=1.5,
            f0_mean_measure_hz=220.0,
            f0_mean_baseline_hz=180.0,
            f0_delta_semitones=3.468,
            jitter_mean_measure=0.011,
            jitter_mean_baseline=0.009,
            jitter_delta=0.002,
            shimmer_mean_measure=0.021,
            shimmer_mean_baseline=0.018,
            shimmer_delta=0.003,
        )
        canonical_fields = (
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

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", return_value=mock_acoustic),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(MagicMock(), self._make_inference_payload())

        dispatched_metrics = mock_persist.delay.call_args.args[0]
        assert all(field in result for field in canonical_fields)
        assert all(field in dispatched_metrics for field in canonical_fields)
        for field in canonical_fields:
            assert result[field] == dispatched_metrics[field]
        assert result["f0_valid_measure"] is True
        assert result["f0_valid_baseline"] is True
        assert result["perturbation_valid_measure"] is True
        assert result["perturbation_valid_baseline"] is True
        assert result["f0_mean_measure_hz"] == pytest.approx(220.0)
        assert result["f0_mean_baseline_hz"] == pytest.approx(180.0)
        assert result["f0_delta_semitones"] == pytest.approx(3.468)
        assert result["jitter_mean_measure"] == pytest.approx(0.011)
        assert result["shimmer_mean_measure"] == pytest.approx(0.021)
        assert "_audio_data" not in result
        assert "_audio_data" not in dispatched_metrics
        assert "raw_audio" not in result
        assert "raw_audio" not in dispatched_metrics
        assert "voiceprint_embedding" not in result
        assert "voiceprint_embedding" not in dispatched_metrics
        assert "reconstructive_voiceprint" not in result
        assert "reconstructive_voiceprint" not in dispatched_metrics

        store.insert_metrics(
            {
                **dispatched_metrics,
                "raw_audio": [0.1, 0.2, 0.3],
                "voiceprint_embedding": [1.0, 2.0, 3.0],
                "reconstructive_voiceprint": {"embedding": [0.4, 0.5, 0.6]},
            }
        )

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        metrics_sql, metrics_params = cursor.execute.call_args_list[0].args
        assert cursor.execute.call_count == 1
        assert "f0_valid_measure" in metrics_sql
        assert tuple(metrics_params.keys()) == _METRICS_DB_FIELDS
        for field in canonical_fields:
            assert metrics_params[field] == dispatched_metrics[field]
        assert "raw_audio" not in metrics_params
        assert "voiceprint_embedding" not in metrics_params
        assert "reconstructive_voiceprint" not in metrics_params
        mock_conn.commit.assert_called_once()

    def test_sparse_invalid_acoustic_windows_persist_null_and_boolean_semantics(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """Sparse speech windows stay non-fatal and persist false/null acoustic semantics."""
        from packages.ml_core.acoustic import AcousticMetrics

        mod = self._get_inference_module()
        mock_persist = MagicMock()
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = ""
        mock_acoustic = MagicMock()
        mock_acoustic.analyze.return_value = AcousticMetrics(
            f0_valid_measure=False,
            f0_valid_baseline=True,
            perturbation_valid_measure=False,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=0.25,
            voiced_coverage_baseline_s=1.5,
            f0_mean_measure_hz=None,
            f0_mean_baseline_hz=175.0,
            f0_delta_semitones=None,
            jitter_mean_measure=None,
            jitter_mean_baseline=0.015,
            jitter_delta=None,
            shimmer_mean_measure=None,
            shimmer_mean_baseline=0.025,
            shimmer_delta=None,
        )

        with (
            patch.object(mod, "persist_metrics", mock_persist),
            patch("packages.ml_core.transcription.TranscriptionEngine", return_value=mock_engine),
            patch("packages.ml_core.acoustic.AcousticAnalyzer", return_value=mock_acoustic),
            patch("subprocess.run"),
            patch("os.remove"),
            patch("tempfile.NamedTemporaryFile") as mock_tmpfile,
        ):
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.raw"
            mock_tmpfile.return_value = mock_file

            result = mod.process_segment(MagicMock(), self._make_inference_payload())

        dispatched_metrics = mock_persist.delay.call_args.args[0]
        assert result["f0_valid_measure"] is False
        assert result["f0_valid_baseline"] is True
        assert result["perturbation_valid_measure"] is False
        assert result["perturbation_valid_baseline"] is True
        assert result["voiced_coverage_measure_s"] == pytest.approx(0.25)
        assert result["voiced_coverage_baseline_s"] == pytest.approx(1.5)
        assert result["f0_mean_measure_hz"] is None
        assert result["f0_mean_baseline_hz"] == pytest.approx(175.0)
        assert result["f0_delta_semitones"] is None
        assert result["jitter_mean_measure"] is None
        assert result["jitter_mean_baseline"] == pytest.approx(0.015)
        assert result["jitter_delta"] is None
        assert result["shimmer_mean_measure"] is None
        assert result["shimmer_mean_baseline"] == pytest.approx(0.025)
        assert result["shimmer_delta"] is None
        assert dispatched_metrics["f0_valid_measure"] is False
        assert dispatched_metrics["perturbation_valid_measure"] is False
        assert dispatched_metrics["f0_mean_measure_hz"] is None
        assert dispatched_metrics["jitter_mean_measure"] is None
        assert dispatched_metrics["shimmer_mean_measure"] is None

        store.insert_metrics(dispatched_metrics)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        _, metrics_params = cursor.execute.call_args_list[0].args
        assert cursor.execute.call_count == 1
        assert metrics_params["f0_valid_measure"] is False
        assert metrics_params["f0_valid_baseline"] is True
        assert metrics_params["perturbation_valid_measure"] is False
        assert metrics_params["perturbation_valid_baseline"] is True
        assert metrics_params["voiced_coverage_measure_s"] == pytest.approx(0.25)
        assert metrics_params["voiced_coverage_baseline_s"] == pytest.approx(1.5)
        assert metrics_params["f0_mean_measure_hz"] is None
        assert metrics_params["f0_mean_baseline_hz"] == pytest.approx(175.0)
        assert metrics_params["f0_delta_semitones"] is None
        assert metrics_params["jitter_mean_measure"] is None
        assert metrics_params["jitter_mean_baseline"] == pytest.approx(0.015)
        assert metrics_params["jitter_delta"] is None
        assert metrics_params["shimmer_mean_measure"] is None
        assert metrics_params["shimmer_mean_baseline"] == pytest.approx(0.025)
        assert metrics_params["shimmer_delta"] is None
        mock_conn.commit.assert_called_once()

    def test_encounter_log_persists_canonical_au12_baseline_pre(
        self,
        mock_conn: MagicMock,
    ) -> None:
        """§7B/§11.5.6 — persistence uses canonical au12_baseline_pre."""
        mod = self._get_inference_module()
        from services.worker.pipeline.reward import RewardResult

        reward_result = RewardResult(
            gated_reward=0.42,
            p90_intensity=0.42,
            semantic_gate=1,
            n_frames_in_window=12,
            au12_baseline_pre=0.123,
        )
        store = MagicMock()
        store._get_conn.return_value = mock_conn
        metrics = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "seg-baseline",
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
        }

        mod._log_encounter(
            store,
            metrics,
            experiment_id="exp-1",
            arm="arm-a",
            result=reward_result,
            stimulus_time=100.0,
        )

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args.args
        assert "au12_baseline_pre" in sql
        assert params["au12_baseline_pre"] == pytest.approx(reward_result.au12_baseline_pre)
        assert set(params) == {
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
        }
        mock_conn.commit.assert_called_once()
        store._put_conn.assert_called_once_with(mock_conn)

    @pytest.mark.audit_item("13.24")
    @pytest.mark.audit_item("13.26")
    @pytest.mark.audit_item("13.28")
    @pytest.mark.audit_item("13.29")
    def test_attribution_ledger_persistence_uses_deterministic_upserts(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§7E/§13.24 — Persist all attribution rows with replay-stable IDs."""
        from datetime import UTC, datetime

        from packages.ml_core.attribution import (
            ATTRIBUTION_EVENT_TYPE_GREETING,
            DEFAULT_ATTRIBUTION_METHOD_VERSION,
            DEFAULT_LINK_RULE_VERSION,
            DEFAULT_REWARD_PATH_VERSION,
            attribution_event_id,
            attribution_score_id,
            build_attribution_ledger_records,
            event_outcome_link_id,
            outcome_event_id,
        )
        from services.worker.pipeline.reward import RewardResult

        created_at = datetime(2026, 3, 13, 12, 2, tzinfo=UTC)
        metrics: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "a" * 64,
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "semantic": {
                "reasoning": "cross_encoder_high_match",
                "is_match": True,
                "confidence_score": 0.75,
                "semantic_method": "cross_encoder",
                "semantic_method_version": "ce-v1.2.3",
            },
            "_active_arm": "arm-a",
            "_expected_greeting": "hello welcome",
            "_stimulus_time": 1773403200.0,
            "_au12_series": [
                {"timestamp_s": 1773403195.5, "intensity": 0.10},
                {"timestamp_s": 1773403197.5, "intensity": 0.12},
                {"timestamp_s": 1773403200.5, "intensity": 0.20},
                {"timestamp_s": 1773403201.0, "intensity": 0.40},
                {"timestamp_s": 1773403202.0, "intensity": 0.60},
            ],
            "_bandit_decision_snapshot": {
                "selection_method": "thompson_sampling",
                "selection_time_utc": "2026-03-13T12:00:00+00:00",
                "experiment_id": 1,
                "policy_version": "policy-v1",
                "selected_arm_id": "arm-a",
                "candidate_arm_ids": ["arm-a", "arm-b"],
                "posterior_by_arm": {
                    "arm-a": {"alpha": 2.0, "beta": 3.0},
                    "arm-b": {"alpha": 4.0, "beta": 5.0},
                },
                "sampled_theta_by_arm": {"arm-a": 0.4, "arm-b": 0.2},
                "expected_greeting": "hello welcome",
                "decision_context_hash": "b" * 64,
                "random_seed": 123,
            },
            "outcome_event": {
                "outcome_type": "creator_follow",
                "outcome_value": 1.0,
                "outcome_time_utc": "2026-03-13T12:01:00+00:00",
                "source_system": "tiktok_webcast",
                "source_event_ref": "follow-123",
                "confidence": 1.0,
            },
        }
        reward_result = RewardResult(
            gated_reward=0.5,
            p90_intensity=0.5,
            semantic_gate=1,
            n_frames_in_window=3,
            au12_baseline_pre=0.11,
        )

        ledger = build_attribution_ledger_records(
            metrics,
            reward_result=reward_result,
            comodulation_result={"co_modulation_index": 0.25},
            created_at=created_at,
        )
        replay_ledger = build_attribution_ledger_records(
            dict(metrics),
            reward_result=reward_result,
            comodulation_result={"co_modulation_index": 0.25},
            created_at=created_at,
        )

        assert ledger is not None
        assert replay_ledger is not None
        assert len(ledger.outcomes) == 1
        assert len(ledger.links) == 1
        assert len(ledger.scores) == 6
        expected_event_id = attribution_event_id(
            session_id=metrics["session_id"],
            segment_id=metrics["segment_id"],
            event_type=ATTRIBUTION_EVENT_TYPE_GREETING,
            reward_path_version=DEFAULT_REWARD_PATH_VERSION,
        )
        expected_outcome_id = outcome_event_id(
            session_id=metrics["session_id"],
            outcome_type="creator_follow",
            outcome_time_utc=ledger.outcomes[0].outcome_time_utc,
            source_system="tiktok_webcast",
            source_event_ref="follow-123",
        )
        expected_link_id = event_outcome_link_id(
            event_id=expected_event_id,
            outcome_id=expected_outcome_id,
            link_rule_version=DEFAULT_LINK_RULE_VERSION,
        )
        assert ledger.event.event_id == expected_event_id
        assert ledger.outcomes[0].outcome_id == expected_outcome_id
        assert ledger.links[0].link_id == expected_link_id
        assert ledger.links[0].lag_s == replay_ledger.links[0].lag_s == 60.0
        assert ledger.links[0].horizon_s == replay_ledger.links[0].horizon_s
        assert ledger.links[0].link_rule_version == DEFAULT_LINK_RULE_VERSION
        assert ledger.event.event_id == replay_ledger.event.event_id
        assert ledger.outcomes[0].outcome_id == replay_ledger.outcomes[0].outcome_id
        assert ledger.links[0].link_id == replay_ledger.links[0].link_id
        assert [score.score_id for score in ledger.scores] == [
            score.score_id for score in replay_ledger.scores
        ]
        assert ledger.scores[0].score_id == attribution_score_id(
            event_id=expected_event_id,
            outcome_id=expected_outcome_id,
            attribution_method="soft_reward_candidate",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
        )

        store.persist_attribution_ledger(ledger)
        store.persist_attribution_ledger(replay_ledger)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        calls = cursor.execute.call_args_list
        assert len(calls) == 18
        first_pass = calls[:9]
        replay_pass = calls[9:]
        table_expectations = (
            ("attribution_event", "ON CONFLICT (event_id)"),
            ("outcome_event", "ON CONFLICT (outcome_id)"),
            ("event_outcome_link", "ON CONFLICT (link_id)"),
        )
        for call_args, (table_name, conflict_clause) in zip(
            first_pass[:3], table_expectations, strict=True
        ):
            sql, _ = call_args.args
            assert f"INSERT INTO {table_name}" in sql
            assert conflict_clause in sql
        assert "INSERT INTO attribution_score" in first_pass[3].args[0]
        assert "ON CONFLICT (score_id)" in first_pass[3].args[0]

        event_params = first_pass[0].args[1]
        assert event_params["event_id"] == str(expected_event_id)
        assert event_params["semantic_method"] == "cross_encoder"
        assert event_params["semantic_method_version"] == "ce-v1.2.3"
        assert event_params["semantic_p_match"] == pytest.approx(0.75)
        assert event_params["finality"] == "online_provisional"
        assert event_params["schema_version"] == "v3.4"
        assert isinstance(event_params["bandit_decision_snapshot"], str)
        assert json.loads(event_params["bandit_decision_snapshot"])["selected_arm_id"] == "arm-a"
        assert first_pass[1].args[1]["outcome_id"] == str(expected_outcome_id)
        assert first_pass[1].args[1]["finality"] == "online_provisional"
        assert first_pass[1].args[1]["schema_version"] == "v3.4"
        assert first_pass[2].args[1]["link_id"] == str(expected_link_id)
        assert first_pass[2].args[1]["lag_s"] == 60.0
        assert first_pass[2].args[1]["link_rule_version"] == DEFAULT_LINK_RULE_VERSION
        assert first_pass[2].args[1]["finality"] == "online_provisional"
        assert first_pass[2].args[1]["schema_version"] == "v3.4"
        assert first_pass[3].args[1]["score_id"] == str(ledger.scores[0].score_id)
        assert first_pass[3].args[1]["finality"] == "online_provisional"
        assert first_pass[3].args[1]["schema_version"] == "v3.4"

        first_ids = [call_args.args[1].get("event_id") for call_args in first_pass]
        first_ids += [first_pass[1].args[1]["outcome_id"], first_pass[2].args[1]["link_id"]]
        first_ids += [call_args.args[1]["score_id"] for call_args in first_pass[3:]]
        replay_ids = [call_args.args[1].get("event_id") for call_args in replay_pass]
        replay_ids += [replay_pass[1].args[1]["outcome_id"], replay_pass[2].args[1]["link_id"]]
        replay_ids += [call_args.args[1]["score_id"] for call_args in replay_pass[3:]]
        assert first_ids == replay_ids
        assert mock_conn.commit.call_count == 2

    def test_attribution_ledger_missing_outcome_persists_event_and_null_link_scores(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§7E.2/§7E.6 — Missing outcomes/null links are non-fatal states."""
        from datetime import UTC, datetime

        from packages.ml_core.attribution import build_attribution_ledger_records
        from services.worker.pipeline.reward import RewardResult

        metrics: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "c" * 64,
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "semantic": {
                "reasoning": "gray_band_llm_match",
                "is_match": True,
                "confidence_score": 0.61,
                "semantic_method": "llm_gray_band",
                "semantic_method_version": "gray-v2",
            },
            "_active_arm": "arm-b",
            "_expected_greeting": "hello welcome",
            "_stimulus_time": 1773403200.0,
            "_au12_series": [],
            "_bandit_decision_snapshot": {
                "selection_method": "thompson_sampling",
                "selection_time_utc": "2026-03-13T12:00:00+00:00",
                "experiment_id": 2,
                "policy_version": "policy-v1",
                "selected_arm_id": "arm-b",
                "candidate_arm_ids": ["arm-b"],
                "posterior_by_arm": {"arm-b": {"alpha": 1.0, "beta": 1.0}},
                "sampled_theta_by_arm": {"arm-b": 0.5},
                "expected_greeting": "hello welcome",
                "decision_context_hash": "b" * 64,
                "random_seed": 123,
            },
        }
        reward_result = RewardResult(
            gated_reward=0.0,
            p90_intensity=0.0,
            semantic_gate=1,
            n_frames_in_window=0,
            au12_baseline_pre=None,
        )

        ledger = build_attribution_ledger_records(
            metrics,
            reward_result=reward_result,
            created_at=datetime(2026, 3, 13, 12, 3, tzinfo=UTC),
        )

        assert ledger is not None
        assert ledger.outcomes == ()
        assert ledger.links == ()
        assert len(ledger.scores) == 6
        assert {score.outcome_id for score in ledger.scores} == {None}

        store.persist_attribution_ledger(ledger)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        calls = cursor.execute.call_args_list
        assert len(calls) == 7
        assert "INSERT INTO attribution_event" in calls[0].args[0]
        assert all("outcome_event" not in call_args.args[0] for call_args in calls)
        assert all("event_outcome_link" not in call_args.args[0] for call_args in calls)
        assert all("INSERT INTO attribution_score" in call_args.args[0] for call_args in calls[1:])
        assert {call_args.args[1]["outcome_id"] for call_args in calls[1:]} == {None}
        mock_conn.commit.assert_called_once()

    def _sample_attribution_ledger(
        self,
        *,
        finality: str = "online_provisional",
        sync_peak_corr: float | None = None,
        sync_peak_lag: float | None = None,
        comodulation_result: dict[str, Any] | None = None,
    ) -> Any:
        """Build a valid deterministic attribution ledger for persistence tests."""
        from datetime import UTC, datetime

        from packages.ml_core.attribution import build_attribution_ledger_records
        from services.worker.pipeline.reward import RewardResult

        metrics: dict[str, Any] = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "segment_id": "e" * 64,
            "timestamp_utc": "2026-03-13T12:00:00+00:00",
            "semantic": {
                "reasoning": "cross_encoder_high_match",
                "is_match": True,
                "confidence_score": 0.75,
                "semantic_method": "cross_encoder",
                "semantic_method_version": "ce-v1.2.3",
            },
            "_active_arm": "arm-a",
            "_expected_greeting": "hello welcome",
            "_stimulus_time": 1773403200.0,
            "_au12_series": [
                {"timestamp_s": 1773403195.5, "intensity": 0.10},
                {"timestamp_s": 1773403197.5, "intensity": 0.12},
                {"timestamp_s": 1773403200.5, "intensity": 0.20},
                {"timestamp_s": 1773403201.0, "intensity": 0.40},
                {"timestamp_s": 1773403202.0, "intensity": 0.60},
            ],
            "_bandit_decision_snapshot": {
                "selection_method": "thompson_sampling",
                "selection_time_utc": "2026-03-13T12:00:00+00:00",
                "experiment_id": 1,
                "policy_version": "policy-v1",
                "selected_arm_id": "arm-a",
                "candidate_arm_ids": ["arm-a", "arm-b"],
                "posterior_by_arm": {
                    "arm-a": {"alpha": 2.0, "beta": 3.0},
                    "arm-b": {"alpha": 4.0, "beta": 5.0},
                },
                "sampled_theta_by_arm": {"arm-a": 0.4, "arm-b": 0.2},
                "expected_greeting": "hello welcome",
                "decision_context_hash": "b" * 64,
                "random_seed": 123,
            },
            "outcome_event": {
                "outcome_type": "creator_follow",
                "outcome_value": 1.0,
                "outcome_time_utc": "2026-03-13T12:01:00+00:00",
                "source_system": "tiktok_webcast",
                "source_event_ref": "follow-123",
                "confidence": 1.0,
            },
        }
        if sync_peak_corr is not None:
            metrics["sync_peak_corr"] = sync_peak_corr
        if sync_peak_lag is not None:
            metrics["sync_peak_lag"] = sync_peak_lag

        reward_result = RewardResult(
            gated_reward=0.5,
            p90_intensity=0.5,
            semantic_gate=1,
            n_frames_in_window=3,
            au12_baseline_pre=0.11,
        )
        ledger = build_attribution_ledger_records(
            metrics,
            reward_result=reward_result,
            comodulation_result=comodulation_result,
            finality=finality,
            created_at=datetime(2026, 3, 13, 12, 2, tzinfo=UTC),
        )
        assert ledger is not None
        return ledger

    def test_attribution_outage_buffers_and_overflows_to_csv(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """§2.7/§7E — Attribution CSV-overflows as soon as capacity is reached."""
        ledger = self._sample_attribution_ledger()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = _mock_psycopg2.OperationalError("connection lost")

        with (
            patch("services.worker.pipeline.analytics.DB_BUFFER_MAX", 2),
            patch("services.worker.pipeline.analytics.CSV_FALLBACK_DIR", str(tmp_path)),
        ):
            store.persist_attribution_ledger(ledger)
            assert store._attribution_buffer == [ledger]
            assert len(store._attribution_buffer) < 2
            assert list(tmp_path.glob("attribution_overflow_*.csv")) == []

            store.persist_attribution_ledger(ledger)
            assert store._attribution_buffer == []

        csv_files = list(tmp_path.glob("attribution_overflow_*.csv"))
        assert len(csv_files) == 1
        with open(csv_files[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["record_type", "payload_json"]
            rows = list(reader)
        assert len(rows) == 2
        assert {row["record_type"] for row in rows} == {"attribution_ledger"}
        payload = json.loads(rows[0]["payload_json"])
        assert payload["event"]["event_id"] == str(ledger.event.event_id)
        assert len(payload["scores"]) == 6

    @pytest.mark.audit_item("13.28")
    def test_attribution_flush_replays_with_upserts_without_duplicate_identities(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§13.28 — Buffered attribution replay uses deterministic upserts."""
        ledger = self._sample_attribution_ledger()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = _mock_psycopg2.OperationalError("connection lost")

        store.persist_attribution_ledger(ledger)
        assert store._attribution_buffer == [ledger]

        cursor.execute.reset_mock()
        cursor.execute.side_effect = None
        mock_conn.commit.reset_mock()
        mock_conn.rollback.reset_mock()

        store.persist_attribution_ledger(ledger)

        assert store._attribution_buffer == []
        calls = cursor.execute.call_args_list
        assert len(calls) == 18
        current_pass = calls[:9]
        flushed_pass = calls[9:]
        assert all("ON CONFLICT" in call_args.args[0] for call_args in calls)
        current_ids = [call_args.args[1].get("event_id") for call_args in current_pass]
        current_ids += [current_pass[1].args[1]["outcome_id"], current_pass[2].args[1]["link_id"]]
        current_ids += [call_args.args[1]["score_id"] for call_args in current_pass[3:]]
        flushed_ids = [call_args.args[1].get("event_id") for call_args in flushed_pass]
        flushed_ids += [flushed_pass[1].args[1]["outcome_id"], flushed_pass[2].args[1]["link_id"]]
        flushed_ids += [call_args.args[1]["score_id"] for call_args in flushed_pass[3:]]
        assert current_ids == flushed_ids
        assert mock_conn.commit.call_count == 2

    @pytest.mark.audit_item("13.24")
    @pytest.mark.audit_item("13.28")
    def test_attribution_score_id_stable_across_finality_transition(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§7E.6/§13.28 — online_provisional → offline_final updates same score rows."""
        online = self._sample_attribution_ledger(finality="online_provisional")
        offline = self._sample_attribution_ledger(finality="offline_final")

        assert [score.score_id for score in online.scores] == [
            score.score_id for score in offline.scores
        ]
        assert {score.finality for score in online.scores} == {"online_provisional"}
        assert {score.finality for score in offline.scores} == {"offline_final"}

        store.persist_attribution_ledger(online)
        store.persist_attribution_ledger(offline)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        calls = cursor.execute.call_args_list
        first_score = calls[3].args[1]
        final_score = calls[12].args[1]
        assert first_score["score_id"] == final_score["score_id"]
        assert first_score["finality"] == "online_provisional"
        assert final_score["finality"] == "offline_final"
        assert "finality = EXCLUDED.finality" in calls[12].args[0]

    def test_synchrony_scores_do_not_synthesize_zero_lag_comodulation_proxy(self) -> None:
        """§7E.5/§11.5.13-14 — unavailable lag-aware inputs persist as NULL."""
        ledger = self._sample_attribution_ledger(comodulation_result={"co_modulation_index": 0.95})
        scores = {score.attribution_method: score for score in ledger.scores}

        assert scores["sync_peak_corr"].score_raw is None
        assert scores["sync_peak_corr"].score_normalized is None
        assert scores["sync_peak_lag"].score_raw is None
        assert scores["sync_peak_corr"].evidence_flags == []
        assert scores["sync_peak_lag"].evidence_flags == []
        assert "zero_lag_comodulation_proxy" not in scores["sync_peak_lag"].evidence_flags

    def test_synchrony_scores_persist_real_upstream_peak_outputs(self) -> None:
        """§7E.5 — Real upstream lag-scan peak correlation/lag are persisted."""
        ledger = self._sample_attribution_ledger(sync_peak_corr=0.42, sync_peak_lag=3.0)
        scores = {score.attribution_method: score for score in ledger.scores}

        assert scores["sync_peak_corr"].score_raw == pytest.approx(0.42)
        assert scores["sync_peak_corr"].score_normalized == pytest.approx(0.42)
        assert scores["sync_peak_lag"].score_raw == pytest.approx(3.0)
        assert scores["sync_peak_corr"].evidence_flags == ["lag_scan_result"]
        assert scores["sync_peak_lag"].evidence_flags == ["lag_scan_result"]

    def test_attribution_buffer_survives_task_local_store_disposal_and_flushes(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """§2.7/§7E — Buffered ledgers survive discarded task-local stores."""
        ledger = self._sample_attribution_ledger()
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = _mock_psycopg2.OperationalError("connection lost")

        store.persist_attribution_ledger(ledger)
        assert store._attribution_buffer == [ledger]

        store.close()
        assert store._attribution_buffer == [ledger]

        next_store = MetricsStore()
        next_pool = MagicMock()
        next_store._pool = next_pool
        next_store._psycopg2 = _mock_psycopg2
        next_pool.getconn.return_value = mock_conn
        cursor.execute.reset_mock()
        cursor.execute.side_effect = None
        mock_conn.commit.reset_mock()
        mock_conn.rollback.reset_mock()

        next_store.persist_attribution_ledger(ledger)

        assert next_store._attribution_buffer == []
        assert store._attribution_buffer == []
        calls = cursor.execute.call_args_list
        assert len(calls) == 18
        assert all("ON CONFLICT" in call_args.args[0] for call_args in calls)
        assert mock_conn.commit.call_count == 2


class TestMetricsStoreExperiments:
    """§4.E.1 — Experiment arm read/write via MetricsStore."""

    def test_get_experiment_arms_filters_disabled_arms_when_status_columns_exist(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        """Scheduler input excludes disabled/end-dated arms."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("arm_a", 5.0, 3.0),
            ("arm_b", 2.0, 8.0),
        ]

        with patch.object(store, "_experiment_arm_status_columns_available", return_value=True):
            arms = store.get_experiment_arms("exp-1")

        assert len(arms) == 2
        assert arms[0] == {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0}
        assert arms[1] == {"arm": "arm_b", "alpha_param": 2.0, "beta_param": 8.0}
        sql = cursor.execute.call_args[0][0]
        assert "COALESCE(enabled, TRUE) = TRUE" in sql
        assert "end_dated_at IS NULL" in sql

    def test_get_experiment_arms_falls_back_when_status_columns_missing(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("arm_a", 5.0, 3.0)]

        with patch.object(store, "_experiment_arm_status_columns_available", return_value=False):
            store.get_experiment_arms("exp-1")

        sql = cursor.execute.call_args[0][0]
        assert "COALESCE(enabled, TRUE) = TRUE" not in sql
        assert "end_dated_at IS NULL" not in sql

    def test_get_experiment_arm_fetches_single_row_without_status_filter(
        self,
        store: MetricsStore,
        mock_conn: MagicMock,
    ) -> None:
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = ("arm_a", 5.0, 3.0)

        arm = store.get_experiment_arm("exp-1", "arm_a")

        assert arm == {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0}
        sql = cursor.execute.call_args[0][0]
        assert "AND arm = %(arm)s" in sql
        assert "end_dated_at IS NULL" not in sql

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
        mock_store.get_experiment_arm.return_value = {
            "arm": "arm_a",
            "alpha_param": 5.0,
            "beta_param": 3.0,
        }
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.8)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 5.8, 3.2)

    def test_update_failure_increments_beta(self) -> None:
        """§4.E.1 — reward < 0.5 increments beta."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arm.return_value = {
            "arm": "arm_a",
            "alpha_param": 5.0,
            "beta_param": 3.0,
        }
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.2)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 5.2, 3.8)

    def test_update_unknown_arm_raises(self) -> None:
        """update raises ValueError for unknown arm."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arm.return_value = None
        engine = ThompsonSamplingEngine(mock_store)
        with pytest.raises(ValueError, match="not found"):
            engine.update("exp-1", "arm_x", reward=1.0)

    def test_update_boundary_reward(self) -> None:
        """§4.E.1 — reward == 0.5 results in equal fractional update."""
        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arm.return_value = {
            "arm": "arm_a",
            "alpha_param": 1.0,
            "beta_param": 1.0,
        }
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.5)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 1.5, 1.5)
