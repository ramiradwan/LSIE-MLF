"""
Module E persistence and Thompson Sampling analytics (§4.E).

Aggregates Module D inference metrics, writes derived outputs to the Persistent
Store, persists physiology/co-modulation and §7E attribution records, and applies
Thompson Sampling posterior updates for §7B rewards.

Persistent Store writes use parameterized SQL with DOUBLE PRECISION and
TIMESTAMPTZ fields where required (§2 step 7). On write failure, records are
buffered up to the configured in-memory limit and persistent failures overflow
to CSV (§12.5).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from packages.schemas.data_tiers import DataTier, mark_data_tier

if TYPE_CHECKING:
    from psycopg2 import pool

logger = logging.getLogger(__name__)

# §12.5 Module E: buffer up to 1000 records before disk overflow
DB_BUFFER_MAX: int = 1000
DB_RETRY_INTERVAL: int = 5  # seconds
CSV_FALLBACK_DIR: str = "/data/processed/failed_tasks/"

# §2.7 / §11.4 — metrics table write order must match the canonical
# observational acoustic schema order in Persistent Store.
_METRICS_REQUIRED_FIELDS: tuple[str, ...] = (
    "session_id",
    "segment_id",
    "timestamp_utc",
)

_METRICS_OPTIONAL_FIELDS: tuple[str, ...] = (
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
)

_METRICS_DB_FIELDS: tuple[str, ...] = _METRICS_REQUIRED_FIELDS + _METRICS_OPTIONAL_FIELDS

_METRICS_OVERFLOW_CORE_FIELDS: tuple[str, ...] = (
    *_METRICS_DB_FIELDS,
    "transcription",
    "semantic",
)

# §2 step 7 — Parameterized INSERT for metrics table
_INSERT_METRICS_SQL: str = mark_data_tier(
    """
    INSERT INTO metrics (
        session_id, segment_id, timestamp_utc,
        au12_intensity,
        f0_valid_measure, f0_valid_baseline,
        perturbation_valid_measure, perturbation_valid_baseline,
        voiced_coverage_measure_s, voiced_coverage_baseline_s,
        f0_mean_measure_hz, f0_mean_baseline_hz, f0_delta_semitones,
        jitter_mean_measure, jitter_mean_baseline, jitter_delta,
        shimmer_mean_measure, shimmer_mean_baseline, shimmer_delta
    )
    VALUES (
        %(session_id)s, %(segment_id)s, %(timestamp_utc)s,
        %(au12_intensity)s,
        %(f0_valid_measure)s, %(f0_valid_baseline)s,
        %(perturbation_valid_measure)s, %(perturbation_valid_baseline)s,
        %(voiced_coverage_measure_s)s, %(voiced_coverage_baseline_s)s,
        %(f0_mean_measure_hz)s, %(f0_mean_baseline_hz)s, %(f0_delta_semitones)s,
        %(jitter_mean_measure)s, %(jitter_mean_baseline)s, %(jitter_delta)s,
        %(shimmer_mean_measure)s, %(shimmer_mean_baseline)s, %(shimmer_delta)s
    )
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Module E derived acoustic/AU12 metrics INSERT",
)  # §5.2.3 Permanent Analytical Storage


def _metrics_insert_params(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build normalized metrics INSERT params without coercing null/bool values."""
    params = {field: metrics[field] for field in _METRICS_REQUIRED_FIELDS}
    params.update({field: metrics.get(field) for field in _METRICS_OPTIONAL_FIELDS})
    return mark_data_tier(
        params,
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
        purpose="Normalized derived metrics row parameters",
    )  # §5.2.3 Permanent Analytical Storage


def _transcript_insert_params(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build normalized transcript INSERT parameters from a metrics record."""
    return mark_data_tier(
        {
            "session_id": metrics["session_id"],
            "segment_id": metrics["segment_id"],
            "timestamp_utc": metrics["timestamp_utc"],
            "text": metrics["transcription"],
        },
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
        purpose="Normalized derived transcript row parameters",
    )  # §5.2.3 Permanent Analytical Storage


def _evaluation_insert_params(metrics: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    """Build normalized bounded evaluation INSERT parameters."""
    return mark_data_tier(
        {
            "session_id": metrics["session_id"],
            "segment_id": metrics["segment_id"],
            "timestamp_utc": metrics["timestamp_utc"],
            "reasoning": semantic.get("reasoning"),
            "is_match": semantic.get("is_match"),
            "confidence": semantic.get("confidence_score"),
        },
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
        purpose="Normalized bounded evaluation row parameters",
    )  # §5.2.3 Permanent Analytical Storage


def _overflow_fieldnames(_records: list[dict[str, Any]]) -> list[str]:
    """Return the fixed approved CSV headers for overflow persistence."""
    return list(_METRICS_OVERFLOW_CORE_FIELDS)


def _csv_serialize_value(value: Any) -> Any:
    """Serialize overflow values without collapsing None to empty strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _overflow_record(record: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    """Materialize a CSV row while preserving every requested field."""
    return {field: _csv_serialize_value(record.get(field)) for field in fieldnames}


# §2 step 7 — Parameterized INSERT for transcripts table
_INSERT_TRANSCRIPT_SQL: str = mark_data_tier(
    """
    INSERT INTO transcripts (session_id, segment_id, timestamp_utc, text)
    VALUES (%(session_id)s, %(segment_id)s, %(timestamp_utc)s, %(text)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Module E derived transcript INSERT",
)  # §5.2.3 Permanent Analytical Storage

# §2 step 7 — Parameterized INSERT for evaluations table
_INSERT_EVALUATION_SQL: str = mark_data_tier(
    """
    INSERT INTO evaluations (session_id, segment_id, timestamp_utc,
                             reasoning, is_match, confidence)
    VALUES (%(session_id)s, %(segment_id)s, %(timestamp_utc)s,
            %(reasoning)s, %(is_match)s, %(confidence)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Module E bounded semantic evaluation INSERT",
)  # §5.2.3 Permanent Analytical Storage

# §4.E.2 — Parameterized INSERT for per-segment physiology snapshots
_INSERT_PHYSIOLOGY_SQL: str = mark_data_tier(
    """
    INSERT INTO physiology_log
        (session_id, segment_id, subject_role, rmssd_ms, heart_rate_bpm,
         freshness_s, is_stale, provider, source_kind, derivation_method,
         window_s, validity_ratio, is_valid, source_timestamp_utc)
    VALUES
        (%(session_id)s, %(segment_id)s, %(subject_role)s, %(rmssd_ms)s,
         %(heart_rate_bpm)s, %(freshness_s)s, %(is_stale)s, %(provider)s,
         %(source_kind)s, %(derivation_method)s, %(window_s)s,
         %(validity_ratio)s, %(is_valid)s, %(source_timestamp_utc)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Module E scalar physiological metrics INSERT",
)  # §5.2.3 Permanent Analytical Storage


def _physiology_insert_params(
    session_id: str,
    segment_id: str,
    subject_role: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Normalize transient physiology context into scalar Permanent Store fields.

    §5.3 forbids raw Oura/provider/chunk payload bodies in the Persistent Store;
    this whitelist only emits scalar/timestamp derivatives named by the
    physiology_log INSERT schema.
    """
    source_kind = snapshot.get("source_kind")
    derivation_method = snapshot.get("derivation_method")
    window_s = snapshot.get("window_s")
    if window_s is None:
        window_s = snapshot.get("window_length_s")
    validity_ratio = snapshot.get("validity_ratio")
    is_valid = snapshot.get("is_valid")
    return mark_data_tier(
        {
            "session_id": session_id,
            "segment_id": segment_id,
            "subject_role": subject_role,
            "rmssd_ms": snapshot.get("rmssd_ms"),
            "heart_rate_bpm": snapshot.get("heart_rate_bpm"),
            "freshness_s": snapshot["freshness_s"],
            "is_stale": snapshot["is_stale"],
            "provider": snapshot["provider"],
            "source_kind": source_kind,
            "derivation_method": derivation_method,
            "window_s": window_s,
            "validity_ratio": validity_ratio,
            "is_valid": is_valid,
            "source_timestamp_utc": snapshot.get("source_timestamp_utc"),
        },
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
        purpose="Normalized scalar physiology_log row parameters",
    )  # §5.2.3 Permanent Analytical Storage


# §7C — Parameterized INSERT for rolling co-modulation analytics
_INSERT_COMODULATION_SQL: str = mark_data_tier(
    """
    INSERT INTO comodulation_log
        (session_id, window_start_utc, window_end_utc, window_minutes,
         co_modulation_index, n_paired_observations, coverage_ratio,
         streamer_rmssd_mean, operator_rmssd_mean)
    VALUES
        (%(session_id)s, %(window_start_utc)s, %(window_end_utc)s,
         %(window_minutes)s, %(co_modulation_index)s,
         %(n_paired_observations)s, %(coverage_ratio)s,
         %(streamer_rmssd_mean)s, %(operator_rmssd_mean)s)
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Module E co-modulation analytics INSERT",
)  # §5.2.3 Permanent Analytical Storage

# §7C — Minimum paired observations for valid co-modulation output
MIN_COMOD_PAIRS: int = 4

# §7C — Rolling co-modulation analysis window
COMOD_WINDOW_MINUTES: int = 10

# §7C.2 — Resampling rule for co-modulation alignment
COMOD_RESAMPLE_RULE: str = "1min"

_QUERY_RECENT_PHYSIOLOGY_SQL: str = """
    SELECT subject_role, rmssd_ms, source_timestamp_utc
    FROM physiology_log
    WHERE session_id = %(session_id)s
      AND source_timestamp_utc > %(window_start_utc)s
      AND source_timestamp_utc <= %(window_end_utc)s
      AND rmssd_ms IS NOT NULL
      AND is_valid = TRUE
      AND is_stale = FALSE
    ORDER BY source_timestamp_utc ASC
"""

# §4.E.1 — SELECT experiment arms for Thompson Sampling scheduler input
_SELECT_ACTIVE_ARMS_SQL: str = """
    SELECT arm, alpha_param, beta_param FROM experiments
    WHERE experiment_id = %(experiment_id)s
      AND COALESCE(enabled, TRUE) = TRUE
      AND end_dated_at IS NULL
    ORDER BY arm ASC
"""

# Legacy fallback when rollout-safe arm-status columns are not present yet.
_SELECT_ARMS_SQL: str = """
    SELECT arm, alpha_param, beta_param FROM experiments
    WHERE experiment_id = %(experiment_id)s
    ORDER BY arm ASC
"""

# Historical-arm lookup for posterior updates. Intentionally unfiltered so a
# reward that arrives after an arm is disabled still updates the row that was
# active at selection time.
_SELECT_ARM_SQL: str = """
    SELECT arm, alpha_param, beta_param FROM experiments
    WHERE experiment_id = %(experiment_id)s
      AND arm = %(arm)s
"""

_EXPERIMENT_ARM_STATUS_COLUMNS_SQL: str = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'experiments'
      AND column_name IN ('enabled', 'end_dated_at')
"""

# §4.E.1 — UPDATE experiment arm posterior (SERIALIZABLE isolation)
_UPDATE_ARM_SQL: str = """
    UPDATE experiments SET alpha_param = %(alpha)s, beta_param = %(beta)s,
                           updated_at = NOW()
    WHERE experiment_id = %(experiment_id)s AND arm = %(arm)s
"""


def _import_psycopg2() -> Any:
    """Lazy import psycopg2 for ML Worker and API Server containers."""
    import psycopg2 as _psycopg2

    return _psycopg2


class MetricsStore:
    """
    §4.E / §2 step 7 — Persistent Store interface for Module E.

    Uses a psycopg2-binary connection pool and parameterized SQL to write
    metrics, transcripts, evaluations, physiology/co-modulation analytics,
    attribution ledger records, and Thompson Sampling posterior updates.

    Isolation levels (§2 step 7):
      - SERIALIZABLE for experiment updates
      - READ COMMITTED for metric inserts and derived analytics writes

    On write failure, records are buffered up to the configured in-memory limit
    and persistent failures overflow to CSV (§12.5).
    """

    _shared_attribution_buffer: ClassVar[list[Any]] = []

    def __init__(self) -> None:
        self._pool: pool.ThreadedConnectionPool | None = None
        self._buffer: list[dict[str, Any]] = []
        self._attribution_buffer: list[Any] = MetricsStore._shared_attribution_buffer
        self._psycopg2: Any = None

    def connect(
        self,
        minconn: int = 2,
        maxconn: int = 10,
    ) -> None:
        """
        Initialize psycopg2 connection pool.

        §2 step 7 — Connection parameters from environment variables.
        """
        self._psycopg2 = _import_psycopg2()
        from psycopg2 import pool as pg_pool

        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=os.environ.get("POSTGRES_HOST", "postgres"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            dbname=os.environ["POSTGRES_DB"],
        )

    def _get_conn(self) -> Any:
        """Obtain a connection from the pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("MetricsStore not connected. Call connect() first.")
        return self._pool.getconn()

    def _put_conn(self, conn: Any) -> None:
        """Return a connection to the pool."""
        if self._pool is not None:
            self._pool.putconn(conn)

    def _is_db_error(self, exc: BaseException) -> bool:
        """Check for psycopg2 connection errors or RuntimeError."""
        if isinstance(exc, RuntimeError):
            return True
        if self._psycopg2 is not None:
            return isinstance(
                exc,
                (self._psycopg2.OperationalError, self._psycopg2.InterfaceError),
            )
        return False

    def insert_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Insert inference metrics into the Persistent Store.

        §2 step 7 — Parameterized queries, DOUBLE PRECISION, TIMESTAMPTZ.
        §12.5.1 Module E — On failure, buffer up to 1000 records, retry 5s,
        then overflow to CSV.
        """
        try:
            self._write_single(metrics)
            if self._buffer:
                self._flush_buffer()
        except Exception as exc:
            if not self._is_db_error(exc):
                raise
            logger.warning(
                "DB write failed, buffering record (%d in buffer)",
                len(self._buffer) + 1,
            )
            self._buffer.append(metrics)
            if len(self._buffer) >= DB_BUFFER_MAX:
                self._overflow_to_csv(self._buffer[:])
                self._buffer.clear()

    def _write_single(self, metrics: dict[str, Any]) -> None:
        """
        Write a single metrics record to the Persistent Store.

        §2 step 7 — READ COMMITTED isolation for metric inserts.
        """
        psycopg2 = _import_psycopg2()
        conn = self._get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)
            with conn.cursor() as cur:
                cur.execute(_INSERT_METRICS_SQL, _metrics_insert_params(metrics))

                if metrics.get("transcription"):
                    cur.execute(
                        _INSERT_TRANSCRIPT_SQL,
                        _transcript_insert_params(metrics),
                    )

                if metrics.get("semantic") is not None:
                    semantic = metrics["semantic"]
                    cur.execute(
                        _INSERT_EVALUATION_SQL,
                        _evaluation_insert_params(metrics, semantic),
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def _flush_buffer(self) -> None:
        """
        Flush buffered records to the Persistent Store or CSV fallback.

        §12.5.1 Module E — retry every 5s, overflow to CSV on persistent failure.
        """
        remaining: list[dict[str, Any]] = []
        for record in self._buffer:
            try:
                self._write_single(record)
            except Exception as exc:
                if not self._is_db_error(exc):
                    raise
                remaining.append(record)

        if remaining:
            logger.warning(
                "Flush: %d records failed, retrying in %ds",
                len(remaining),
                DB_RETRY_INTERVAL,
            )
            time.sleep(DB_RETRY_INTERVAL)
            still_failing: list[dict[str, Any]] = []
            for record in remaining:
                try:
                    self._write_single(record)
                except Exception as exc:
                    if not self._is_db_error(exc):
                        raise
                    still_failing.append(record)

            if still_failing:
                self._overflow_to_csv(still_failing)

        self._buffer.clear()

    def _overflow_to_csv(self, records: list[dict[str, Any]]) -> None:
        """
        Write overflow records to CSV fallback storage.

        §12.5.4 Module E — CSV fallback when the Persistent Store is unreachable.
        """
        fallback_dir = Path(CSV_FALLBACK_DIR)
        fallback_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        csv_path = fallback_dir / f"overflow_{timestamp}.csv"

        fieldnames = _overflow_fieldnames(records)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_overflow_record(record, fieldnames) for record in records)

        logger.error("Overflow: wrote %d records to %s", len(records), csv_path)

    def _experiment_arm_status_columns_available(self, conn: Any) -> bool:
        """Return True once `enabled` + `end_dated_at` exist on `experiments`."""
        with conn.cursor() as cur:
            cur.execute(_EXPERIMENT_ARM_STATUS_COLUMNS_SQL)
            columns = {row[0] for row in cur.fetchall()}
        return {"enabled", "end_dated_at"}.issubset(columns)

    def get_experiment_arms(self, experiment_id: str) -> list[dict[str, Any]]:
        """
        Fetch scheduler-eligible arms for an experiment.

        Disabled or end-dated arms are excluded once the additive arm-status
        columns are present; older schemas transparently fall back to the
        legacy unfiltered query so rollout order stays safe.
        """
        conn = self._get_conn()
        try:
            sql = (
                _SELECT_ACTIVE_ARMS_SQL
                if self._experiment_arm_status_columns_available(conn)
                else _SELECT_ARMS_SQL
            )
            with conn.cursor() as cur:
                cur.execute(sql, {"experiment_id": experiment_id})
                rows = cur.fetchall()
            return [{"arm": row[0], "alpha_param": row[1], "beta_param": row[2]} for row in rows]
        finally:
            self._put_conn(conn)

    def get_experiment_arm(self, experiment_id: str, arm: str) -> dict[str, Any] | None:
        """Fetch one arm record without scheduler-status filtering."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    _SELECT_ARM_SQL,
                    {
                        "experiment_id": experiment_id,
                        "arm": arm,
                    },
                )
                row = cur.fetchone()
            if row is None:
                return None
            return {"arm": row[0], "alpha_param": row[1], "beta_param": row[2]}
        finally:
            self._put_conn(conn)

    def update_experiment_arm(
        self,
        experiment_id: str,
        arm: str,
        alpha: float,
        beta: float,
    ) -> None:
        """
        Update an experiment arm's posterior in the Persistent Store.

        §2 step 7 — SERIALIZABLE isolation for experiment updates.
        """
        psycopg2 = _import_psycopg2()
        conn = self._get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE)
            with conn.cursor() as cur:
                cur.execute(
                    _UPDATE_ARM_SQL,
                    {
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "alpha": alpha,
                        "beta": beta,
                    },
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def persist_physiology_snapshot(
        self,
        session_id: str,
        segment_id: str,
        subject_role: str,
        snapshot: dict[str, Any],
    ) -> None:
        """§4.E.2 — Persist a normalized scalar physiological snapshot."""
        params = _physiology_insert_params(session_id, segment_id, subject_role, snapshot)

        psycopg2 = _import_psycopg2()
        conn = self._get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_PHYSIOLOGY_SQL,
                    params,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def compute_comodulation(self, session_id: str) -> dict[str, Any]:
        """
        §4.E.2 / §7C — Compute and persist rolling co-modulation analytics.

        Uses 1-minute mean resampling for each subject role over the rolling
        10-minute window, persists every non-error invocation, and returns the
        persisted result payload even when Pearson correlation cannot be emitted.
        """
        import pandas as pd
        from scipy.stats import pearsonr

        window_end_utc = datetime.now(UTC)
        window_start_utc = window_end_utc - timedelta(minutes=COMOD_WINDOW_MINUTES)
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    _QUERY_RECENT_PHYSIOLOGY_SQL,
                    {
                        "session_id": session_id,
                        "window_start_utc": window_start_utc,
                        "window_end_utc": window_end_utc,
                    },
                )
                rows = cur.fetchall()

            result: dict[str, Any] = {
                "session_id": session_id,
                "window_start_utc": window_start_utc,
                "window_end_utc": window_end_utc,
                "window_minutes": COMOD_WINDOW_MINUTES,
                "co_modulation_index": None,
                "n_paired_observations": 0,
                "coverage_ratio": 0.0,
                "streamer_rmssd_mean": None,
                "operator_rmssd_mean": None,
            }

            empty_series = pd.Series(dtype="float64")
            streamer_1m = empty_series
            operator_1m = empty_series
            aligned = pd.DataFrame(columns=["streamer", "operator"])

            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["subject_role", "rmssd_ms", "source_timestamp_utc"],
                )
                df["source_timestamp_utc"] = pd.to_datetime(
                    df["source_timestamp_utc"],
                    utc=True,
                    errors="coerce",
                )
                df["rmssd_ms"] = pd.to_numeric(df["rmssd_ms"], errors="coerce")
                df = df.dropna(subset=["subject_role", "rmssd_ms", "source_timestamp_utc"])
                df = df[df["subject_role"].isin(["streamer", "operator"])]
                df = df[
                    (df["source_timestamp_utc"] > pd.Timestamp(window_start_utc))
                    & (df["source_timestamp_utc"] <= pd.Timestamp(window_end_utc))
                ]

                if not df.empty:
                    streamer = (
                        df[df["subject_role"] == "streamer"]
                        .set_index("source_timestamp_utc")["rmssd_ms"]
                        .sort_index()
                    )
                    operator = (
                        df[df["subject_role"] == "operator"]
                        .set_index("source_timestamp_utc")["rmssd_ms"]
                        .sort_index()
                    )

                    streamer_1m = streamer.resample(COMOD_RESAMPLE_RULE).mean().dropna()
                    operator_1m = operator.resample(COMOD_RESAMPLE_RULE).mean().dropna()
                    aligned = pd.concat(
                        [
                            streamer_1m.rename("streamer"),
                            operator_1m.rename("operator"),
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                else:
                    logger.info(
                        "Co-modulation unavailable: session=%s has no parseable "
                        "eligible RMSSD samples",
                        session_id,
                    )
            else:
                logger.info(
                    "Co-modulation unavailable: session=%s has no eligible "
                    "non-stale physiology rows",
                    session_id,
                )

            n_pairs = int(len(aligned))
            coverage_ratio = float(n_pairs / max(len(streamer_1m), len(operator_1m), 1))
            result["n_paired_observations"] = n_pairs
            result["coverage_ratio"] = coverage_ratio
            result["streamer_rmssd_mean"] = (
                float(streamer_1m.mean()) if not streamer_1m.empty else None
            )
            result["operator_rmssd_mean"] = (
                float(operator_1m.mean()) if not operator_1m.empty else None
            )

            if n_pairs >= MIN_COMOD_PAIRS:
                streamer_variance = float(aligned["streamer"].var(ddof=0))
                operator_variance = float(aligned["operator"].var(ddof=0))
                if streamer_variance > 0.0 and operator_variance > 0.0:
                    r_value, _ = pearsonr(aligned["streamer"], aligned["operator"])
                    if r_value == r_value:
                        result["co_modulation_index"] = float(r_value)
                    else:
                        logger.info(
                            "Co-modulation unavailable: session=%s Pearson "
                            "correlation is undefined",
                            session_id,
                        )
                else:
                    logger.info(
                        "Co-modulation unavailable: session=%s aligned series have zero variance",
                        session_id,
                    )
            else:
                logger.info(
                    "Co-modulation unavailable: session=%s aligned_observations=%d minimum=%d",
                    session_id,
                    n_pairs,
                    MIN_COMOD_PAIRS,
                )

            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_COMODULATION_SQL,
                    mark_data_tier(
                        result,
                        DataTier.PERMANENT,
                        spec_ref="§5.2.3",
                        purpose="Normalized co-modulation analytics row parameters",
                    ),
                )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    def persist_attribution_ledger(self, ledger: Any) -> None:
        """§4.E.3 / §7E — Persist attribution ledger records idempotently.

        All four ledger tables use deterministic UUID primary keys and
        replay-safe ``INSERT ... ON CONFLICT`` upserts. Missing outcomes and
        null linkage are represented by empty ``ledger.outcomes`` /
        ``ledger.links`` collections and are therefore valid no-op states for
        those tables.

        §2.7 outage handling mirrors metric persistence for attribution writes:
        unreachable Persistent Store errors are buffered in memory up to the
        configured in-memory limit, retried/flushed after successful writes with
        the configured retry spacing, and overflowed to CSV rather than silently
        dropped. The attribution buffer is process-level so per-task MetricsStore
        disposal cannot drop ledgers that have not yet reached the CSV overflow
        threshold.
        """
        if ledger is None:
            return

        try:
            self._write_attribution_ledger(ledger)
            if self._attribution_buffer:
                self._flush_attribution_buffer()
        except Exception as exc:
            if not self._is_db_error(exc):
                raise
            logger.warning(
                "Attribution ledger write failed, buffering ledger (%d in buffer)",
                len(self._attribution_buffer) + 1,
            )
            self._attribution_buffer.append(ledger)
            if len(self._attribution_buffer) >= DB_BUFFER_MAX:
                self._overflow_attribution_to_csv(self._attribution_buffer[:])
                self._attribution_buffer.clear()

    def _write_attribution_ledger(self, ledger: Any) -> None:
        """Write one attribution ledger directly with replay-safe upserts."""
        if ledger is None:
            return

        insert_event_sql = mark_data_tier(
            """
    INSERT INTO attribution_event (
        event_id, session_id, segment_id, event_type, event_time_utc,
        stimulus_time_utc, stimulus_id, stimulus_modality,
        selected_arm_id, expected_rule_text_hash, expected_response_rule_text_hash,
        semantic_method, semantic_method_version, semantic_p_match,
        semantic_reason_code, matched_response_time_utc,
        response_registration_status, response_reason_code, reward_path_version,
        bandit_decision_snapshot, evidence_flags, finality,
        schema_version, created_at
    ) VALUES (
        %(event_id)s, %(session_id)s, %(segment_id)s, %(event_type)s,
        %(event_time_utc)s, %(stimulus_time_utc)s, %(stimulus_id)s,
        %(stimulus_modality)s, %(selected_arm_id)s,
        %(expected_rule_text_hash)s, %(expected_response_rule_text_hash)s,
        %(semantic_method)s, %(semantic_method_version)s, %(semantic_p_match)s,
        %(semantic_reason_code)s, %(matched_response_time_utc)s,
        %(response_registration_status)s, %(response_reason_code)s,
        %(reward_path_version)s, %(bandit_decision_snapshot)s::jsonb,
        %(evidence_flags)s, %(finality)s, %(schema_version)s, %(created_at)s
    )
    ON CONFLICT (event_id) DO UPDATE SET
        session_id = EXCLUDED.session_id,
        segment_id = EXCLUDED.segment_id,
        event_type = EXCLUDED.event_type,
        event_time_utc = EXCLUDED.event_time_utc,
        stimulus_time_utc = EXCLUDED.stimulus_time_utc,
        stimulus_id = EXCLUDED.stimulus_id,
        stimulus_modality = EXCLUDED.stimulus_modality,
        selected_arm_id = EXCLUDED.selected_arm_id,
        expected_rule_text_hash = EXCLUDED.expected_rule_text_hash,
        expected_response_rule_text_hash = EXCLUDED.expected_response_rule_text_hash,
        semantic_method = EXCLUDED.semantic_method,
        semantic_method_version = EXCLUDED.semantic_method_version,
        semantic_p_match = EXCLUDED.semantic_p_match,
        semantic_reason_code = EXCLUDED.semantic_reason_code,
        matched_response_time_utc = EXCLUDED.matched_response_time_utc,
        response_registration_status = EXCLUDED.response_registration_status,
        response_reason_code = EXCLUDED.response_reason_code,
        reward_path_version = EXCLUDED.reward_path_version,
        bandit_decision_snapshot = EXCLUDED.bandit_decision_snapshot,
        evidence_flags = EXCLUDED.evidence_flags,
        finality = EXCLUDED.finality,
        schema_version = EXCLUDED.schema_version
""",
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Attribution event analytics INSERT",
        )  # §5.2.3 Permanent Analytical Storage
        insert_outcome_sql = mark_data_tier(
            """
    INSERT INTO outcome_event (
        outcome_id, session_id, outcome_type, outcome_value,
        outcome_time_utc, source_system, source_event_ref,
        confidence, finality, schema_version, created_at
    ) VALUES (
        %(outcome_id)s, %(session_id)s, %(outcome_type)s,
        %(outcome_value)s, %(outcome_time_utc)s, %(source_system)s,
        %(source_event_ref)s, %(confidence)s, %(finality)s,
        %(schema_version)s, %(created_at)s
    )
    ON CONFLICT (outcome_id) DO UPDATE SET
        session_id = EXCLUDED.session_id,
        outcome_type = EXCLUDED.outcome_type,
        outcome_value = EXCLUDED.outcome_value,
        outcome_time_utc = EXCLUDED.outcome_time_utc,
        source_system = EXCLUDED.source_system,
        source_event_ref = EXCLUDED.source_event_ref,
        confidence = EXCLUDED.confidence,
        finality = EXCLUDED.finality,
        schema_version = EXCLUDED.schema_version
""",
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Attribution outcome analytics INSERT",
        )  # §5.2.3 Permanent Analytical Storage
        insert_link_sql = mark_data_tier(
            """
    INSERT INTO event_outcome_link (
        link_id, event_id, outcome_id, lag_s, horizon_s,
        link_rule_version, eligibility_flags, finality,
        schema_version, created_at
    ) VALUES (
        %(link_id)s, %(event_id)s, %(outcome_id)s, %(lag_s)s,
        %(horizon_s)s, %(link_rule_version)s,
        %(eligibility_flags)s, %(finality)s, %(schema_version)s,
        %(created_at)s
    )
    ON CONFLICT (link_id) DO UPDATE SET
        event_id = EXCLUDED.event_id,
        outcome_id = EXCLUDED.outcome_id,
        lag_s = EXCLUDED.lag_s,
        horizon_s = EXCLUDED.horizon_s,
        link_rule_version = EXCLUDED.link_rule_version,
        eligibility_flags = EXCLUDED.eligibility_flags,
        finality = EXCLUDED.finality,
        schema_version = EXCLUDED.schema_version
""",
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Attribution event/outcome link INSERT",
        )  # §5.2.3 Permanent Analytical Storage
        insert_score_sql = mark_data_tier(
            """
    INSERT INTO attribution_score (
        score_id, event_id, outcome_id, attribution_method,
        method_version, score_raw, score_normalized, confidence,
        evidence_flags, finality, schema_version, created_at
    ) VALUES (
        %(score_id)s, %(event_id)s, %(outcome_id)s,
        %(attribution_method)s, %(method_version)s, %(score_raw)s,
        %(score_normalized)s, %(confidence)s, %(evidence_flags)s,
        %(finality)s, %(schema_version)s, %(created_at)s
    )
    ON CONFLICT (score_id) DO UPDATE SET
        event_id = EXCLUDED.event_id,
        outcome_id = EXCLUDED.outcome_id,
        attribution_method = EXCLUDED.attribution_method,
        method_version = EXCLUDED.method_version,
        score_raw = EXCLUDED.score_raw,
        score_normalized = EXCLUDED.score_normalized,
        confidence = EXCLUDED.confidence,
        evidence_flags = EXCLUDED.evidence_flags,
        finality = EXCLUDED.finality,
        schema_version = EXCLUDED.schema_version
""",
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Attribution score analytics INSERT",
        )  # §5.2.3 Permanent Analytical Storage

        event_params = ledger.event.model_dump(mode="json")
        event_params["bandit_decision_snapshot"] = json.dumps(
            event_params["bandit_decision_snapshot"],
            sort_keys=True,
            separators=(",", ":"),
        )
        outcome_params = [outcome.model_dump(mode="json") for outcome in ledger.outcomes]
        link_params = [link.model_dump(mode="json") for link in ledger.links]
        score_params = [score.model_dump(mode="json") for score in ledger.scores]

        psycopg2 = _import_psycopg2()
        conn = self._get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)
            with conn.cursor() as cur:
                cur.execute(
                    insert_event_sql,
                    mark_data_tier(
                        event_params,
                        DataTier.PERMANENT,
                        spec_ref="§5.2.3",
                        purpose="Normalized AttributionEvent row parameters",
                    ),
                )
                for params in outcome_params:
                    cur.execute(
                        insert_outcome_sql,
                        mark_data_tier(
                            params,
                            DataTier.PERMANENT,
                            spec_ref="§5.2.3",
                            purpose="Normalized OutcomeEvent row parameters",
                        ),
                    )
                for params in link_params:
                    cur.execute(
                        insert_link_sql,
                        mark_data_tier(
                            params,
                            DataTier.PERMANENT,
                            spec_ref="§5.2.3",
                            purpose="Normalized EventOutcomeLink row parameters",
                        ),
                    )
                for params in score_params:
                    cur.execute(
                        insert_score_sql,
                        mark_data_tier(
                            params,
                            DataTier.PERMANENT,
                            spec_ref="§5.2.3",
                            purpose="Normalized AttributionScore row parameters",
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def _attribution_overflow_payload(self, ledger: Any) -> str:
        """Serialize one buffered attribution ledger for CSV fallback replay."""
        return json.dumps(
            {
                "event": ledger.event.model_dump(mode="json"),
                "outcomes": [outcome.model_dump(mode="json") for outcome in ledger.outcomes],
                "links": [link.model_dump(mode="json") for link in ledger.links],
                "scores": [score.model_dump(mode="json") for score in ledger.scores],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _overflow_attribution_to_csv(self, ledgers: list[Any]) -> None:
        """Write attribution overflow ledgers to CSV fallback storage."""
        fallback_dir = Path(CSV_FALLBACK_DIR)
        fallback_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        csv_path = fallback_dir / f"attribution_overflow_{timestamp}.csv"
        fieldnames = ["record_type", "payload_json"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ledger in ledgers:
                writer.writerow(
                    {
                        "record_type": "attribution_ledger",
                        "payload_json": self._attribution_overflow_payload(ledger),
                    }
                )

        logger.error("Attribution overflow: wrote %d ledgers to %s", len(ledgers), csv_path)

    def _flush_attribution_buffer(self) -> None:
        """Retry buffered attribution ledgers, then overflow persistent failures."""
        remaining: list[Any] = []
        for ledger in self._attribution_buffer:
            try:
                self._write_attribution_ledger(ledger)
            except Exception as exc:
                if not self._is_db_error(exc):
                    raise
                remaining.append(ledger)

        if remaining:
            logger.warning(
                "Attribution flush: %d ledgers failed, retrying in %ds",
                len(remaining),
                DB_RETRY_INTERVAL,
            )
            time.sleep(DB_RETRY_INTERVAL)
            still_failing: list[Any] = []
            for ledger in remaining:
                try:
                    self._write_attribution_ledger(ledger)
                except Exception as exc:
                    if not self._is_db_error(exc):
                        raise
                    still_failing.append(ledger)

            if still_failing:
                self._overflow_attribution_to_csv(still_failing)

        self._attribution_buffer.clear()


class ThompsonSamplingEngine:
    """
    §4.E.1 — Thompson Sampling arm selection and posterior updates.

    Reads scheduler-eligible arms through MetricsStore, samples Beta(alpha,
    beta) for arm selection, and applies fractional Beta-Bernoulli updates for
    bounded rewards in [0, 1]. MetricsStore performs Persistent Store access;
    this class does not run inference or compute rewards.
    """

    def __init__(self, store: MetricsStore) -> None:
        self.store = store

    def select_arm(self, experiment_id: str) -> str:
        """
        Select the next arm via Thompson Sampling.

        §4.E.1 — For each arm, draw from Beta(alpha, beta).
        Return the arm with the highest sample.
        """
        from scipy.stats import beta as beta_dist

        arms = self.store.get_experiment_arms(experiment_id)
        if not arms:
            raise ValueError(f"No arms found for experiment '{experiment_id}'")

        best_arm = ""
        best_sample = -1.0
        for arm_data in arms:
            sample: float = float(beta_dist.rvs(arm_data["alpha_param"], arm_data["beta_param"]))
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_data["arm"]

        return best_arm

    def update(self, experiment_id: str, arm: str, reward: float) -> None:
        """
        Update posterior with observed reward.

        §4.E.1 — Fractional Beta-Bernoulli update α += r_t, β += (1 − r_t)
        for continuous rewards constrained to the interval [0, 1].
        """
        # Validate reward bounds before arm lookup
        if not 0.0 <= reward <= 1.0:
            raise ValueError(
                f"Reward must be in [0.0, 1.0], got {reward:.4f}. "
                "Ensure the reward pipeline produces bounded output."
            )

        arm_data = self.store.get_experiment_arm(experiment_id, arm)
        if arm_data is None:
            raise ValueError(f"Arm '{arm}' not found for experiment '{experiment_id}'")

        alpha: float = float(arm_data["alpha_param"])
        beta_val: float = float(arm_data["beta_param"])

        # Fractional Beta–Bernoulli pseudo-count update
        alpha += reward
        beta_val += 1.0 - reward

        self.store.update_experiment_arm(experiment_id, arm, alpha, beta_val)
