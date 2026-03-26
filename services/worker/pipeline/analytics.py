"""
Experimentation & Analytics — §4.E Module E

Aggregates inference metrics, persists to Persistent Store,
and runs adaptive experimentation via Thompson Sampling.

§2 step 7 — Parameterized INSERT, DOUBLE PRECISION, TIMESTAMPTZ.
§12.1 Module E — Buffer 1000 records, retry every 5s, CSV fallback.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg2 import pool

logger = logging.getLogger(__name__)

# §12.1 Module E: buffer up to 1000 records before disk overflow
DB_BUFFER_MAX: int = 1000
DB_RETRY_INTERVAL: int = 5  # seconds
CSV_FALLBACK_DIR: str = "/data/processed/failed_tasks/"

# §2 step 7 — Parameterized INSERT for metrics table
_INSERT_METRICS_SQL: str = """
    INSERT INTO metrics (session_id, segment_id, timestamp_utc,
                         au12_intensity, pitch_f0, jitter, shimmer)
    VALUES (%(session_id)s, %(segment_id)s, %(timestamp_utc)s,
            %(au12_intensity)s, %(pitch_f0)s, %(jitter)s, %(shimmer)s)
"""

# §2 step 7 — Parameterized INSERT for transcripts table
_INSERT_TRANSCRIPT_SQL: str = """
    INSERT INTO transcripts (session_id, segment_id, timestamp_utc, text)
    VALUES (%(session_id)s, %(segment_id)s, %(timestamp_utc)s, %(text)s)
"""

# §2 step 7 — Parameterized INSERT for evaluations table
_INSERT_EVALUATION_SQL: str = """
    INSERT INTO evaluations (session_id, segment_id, timestamp_utc,
                             reasoning, is_match, confidence)
    VALUES (%(session_id)s, %(segment_id)s, %(timestamp_utc)s,
            %(reasoning)s, %(is_match)s, %(confidence)s)
"""

# §4.E.1 — SELECT experiment arms for Thompson Sampling
_SELECT_ARMS_SQL: str = """
    SELECT arm, alpha_param, beta_param FROM experiments
    WHERE experiment_id = %(experiment_id)s
"""

# §4.E.1 — UPDATE experiment arm posterior (SERIALIZABLE isolation)
_UPDATE_ARM_SQL: str = """
    UPDATE experiments SET alpha_param = %(alpha)s, beta_param = %(beta)s,
                           updated_at = NOW()
    WHERE experiment_id = %(experiment_id)s AND arm = %(arm)s
"""


def _import_psycopg2() -> Any:
    """Lazy import psycopg2 — only available inside worker/api containers."""
    import psycopg2 as _psycopg2

    return _psycopg2


class MetricsStore:
    """
    §4.E / §2 step 7 — Persistent Store interface.

    Uses psycopg2-binary connection pool. SQL INSERT with parameterized
    queries storing metrics as DOUBLE PRECISION and timestamps as TIMESTAMPTZ.

    Isolation levels (§2 step 7):
      - SERIALIZABLE for experiment updates
      - READ COMMITTED for metric inserts

    Failure: buffer 1000 records, retry every 5s, overflow to CSV.
    """

    def __init__(self) -> None:
        self._pool: pool.ThreadedConnectionPool | None = None
        self._buffer: list[dict[str, Any]] = []
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
        """Check if exception is a psycopg2 DB error or RuntimeError."""
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
        §12.1 Module E — On failure, buffer up to 1000 records, retry 5s,
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
        Write a single metrics record to the database.

        §2 step 7 — READ COMMITTED isolation for metric inserts.
        """
        psycopg2 = _import_psycopg2()
        conn = self._get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_METRICS_SQL,
                    {
                        "session_id": metrics["session_id"],
                        "segment_id": metrics["segment_id"],
                        "timestamp_utc": metrics["timestamp_utc"],
                        "au12_intensity": metrics.get("au12_intensity"),
                        "pitch_f0": metrics.get("pitch_f0"),
                        "jitter": metrics.get("jitter"),
                        "shimmer": metrics.get("shimmer"),
                    },
                )

                if metrics.get("transcription"):
                    cur.execute(
                        _INSERT_TRANSCRIPT_SQL,
                        {
                            "session_id": metrics["session_id"],
                            "segment_id": metrics["segment_id"],
                            "timestamp_utc": metrics["timestamp_utc"],
                            "text": metrics["transcription"],
                        },
                    )

                if metrics.get("semantic") is not None:
                    semantic = metrics["semantic"]
                    cur.execute(
                        _INSERT_EVALUATION_SQL,
                        {
                            "session_id": metrics["session_id"],
                            "segment_id": metrics["segment_id"],
                            "timestamp_utc": metrics["timestamp_utc"],
                            "reasoning": semantic.get("reasoning"),
                            "is_match": semantic.get("is_match"),
                            "confidence": semantic.get("confidence"),
                        },
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def _flush_buffer(self) -> None:
        """
        Flush buffered records to the database or CSV fallback.

        §12.1 Module E — retry every 5s, overflow to CSV on persistent failure.
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

        §12.4 Module E — CSV fallback when DB is unreachable.
        """
        fallback_dir = Path(CSV_FALLBACK_DIR)
        fallback_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        csv_path = fallback_dir / f"overflow_{timestamp}.csv"

        fieldnames = [
            "session_id",
            "segment_id",
            "timestamp_utc",
            "au12_intensity",
            "pitch_f0",
            "jitter",
            "shimmer",
            "transcription",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        logger.error("Overflow: wrote %d records to %s", len(records), csv_path)

    def get_experiment_arms(self, experiment_id: str) -> list[dict[str, Any]]:
        """
        Fetch all arms for an experiment from the Persistent Store.

        §4.E.1 — Returns list of {arm, alpha_param, beta_param}.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SELECT_ARMS_SQL, {"experiment_id": experiment_id})
                rows = cur.fetchall()
            return [{"arm": row[0], "alpha_param": row[1], "beta_param": row[2]} for row in rows]
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

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


class ThompsonSamplingEngine:
    """
    §4.E.1 — Adaptive experimentation using Thompson Sampling.

    Dynamically evaluates greeting rules and behavioral prompts.
    Experiment state persisted to Persistent Store via MetricsStore.

    Uses SciPy Beta distributions:
      - select_arm(): sample from Beta(alpha, beta) for each arm, pick max
      - update(): fractional Beta-Bernoulli update α += r_t, β += (1 − r_t)
        for continuous rewards in [0,1]
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

        arms = self.store.get_experiment_arms(experiment_id)

        for arm_data in arms:
            if arm_data["arm"] == arm:
                alpha: float = float(arm_data["alpha_param"])
                beta_val: float = float(arm_data["beta_param"])

                # Fractional Beta–Bernoulli pseudo-count update
                alpha += reward
                beta_val += 1.0 - reward

                self.store.update_experiment_arm(experiment_id, arm, alpha, beta_val)
                return

        raise ValueError(f"Arm '{arm}' not found for experiment '{experiment_id}'")
