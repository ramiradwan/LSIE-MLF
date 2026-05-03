"""SQLite-backed experiment admin service for the desktop API shell."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from services.api.services.experiment_admin_service import (
    ExperimentAlreadyExistsError,
    ExperimentArmAlreadyExistsError,
    ExperimentArmNotFoundError,
    ExperimentMutationValidationError,
    ExperimentNotFoundError,
)
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema

_BETA_PRIOR_ALPHA: float = 1.0
_BETA_PRIOR_BETA: float = 1.0


class SqliteExperimentAdminService:
    """Handle experiment admin writes against desktop SQLite state."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._db_path = db_path
        self._clock = clock

    def create_experiment(self, request: ExperimentCreateRequest) -> ExperimentAdminResponse:
        with self._connection() as conn:
            if _experiment_exists(conn, request.experiment_id):
                raise ExperimentAlreadyExistsError(
                    f"experiment '{request.experiment_id}' already exists"
                )
            for arm in request.arms:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id, label, arm, greeting_text,
                        alpha_param, beta_param, enabled, end_dated_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?)
                    """,
                    (
                        request.experiment_id,
                        request.label,
                        arm.arm,
                        arm.greeting_text,
                        _BETA_PRIOR_ALPHA,
                        _BETA_PRIOR_BETA,
                        _iso_utc(self._clock()),
                    ),
                )
            rows = _fetch_experiment_rows(conn, request.experiment_id)
        return ExperimentAdminResponse(
            experiment_id=request.experiment_id,
            label=request.label,
            arms=[_build_arm_response(row) for row in rows],
        )

    def add_arm(
        self,
        experiment_id: str,
        request: ExperimentArmCreateRequest,
    ) -> ExperimentArmAdminResponse:
        with self._connection() as conn:
            identity = _fetch_experiment_identity(conn, experiment_id)
            if identity is None:
                raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found")
            existing = _fetch_arm_row(conn, experiment_id, request.arm)
            if existing is not None:
                raise ExperimentArmAlreadyExistsError(
                    f"arm '{request.arm}' already exists for experiment '{experiment_id}'"
                )
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, label, arm, greeting_text,
                    alpha_param, beta_param, enabled, end_dated_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?)
                """,
                (
                    experiment_id,
                    str(identity["label"] or experiment_id),
                    request.arm,
                    request.greeting_text,
                    _BETA_PRIOR_ALPHA,
                    _BETA_PRIOR_BETA,
                    _iso_utc(self._clock()),
                ),
            )
            row = _fetch_arm_row(conn, experiment_id, request.arm)
        if row is None:
            raise ExperimentArmNotFoundError(
                f"arm '{request.arm}' not found for experiment '{experiment_id}' after insert"
            )
        return _build_arm_response(row)

    def patch_arm(
        self,
        experiment_id: str,
        arm_id: str,
        request: ExperimentArmPatchRequest,
    ) -> ExperimentArmAdminResponse:
        if request.enabled is True:
            raise ExperimentMutationValidationError(
                "enabled=true is not supported; use enabled=false to disable an arm"
            )
        with self._connection() as conn:
            row = _fetch_arm_row(conn, experiment_id, arm_id)
            if row is None:
                _raise_missing_experiment_or_arm(conn, experiment_id, arm_id)
            assert row is not None
            greeting_text = request.greeting_text or str(row["greeting_text"] or arm_id)
            enabled = int(row["enabled"] if row["enabled"] is not None else 1)
            end_dated_at = row["end_dated_at"]
            if request.enabled is False:
                enabled = 0
                if end_dated_at is None:
                    end_dated_at = _iso_utc(self._clock())
            conn.execute(
                """
                UPDATE experiments
                SET greeting_text = ?, enabled = ?, end_dated_at = ?, updated_at = ?
                WHERE experiment_id = ? AND arm = ?
                """,
                (
                    greeting_text,
                    enabled,
                    end_dated_at,
                    _iso_utc(self._clock()),
                    experiment_id,
                    arm_id,
                ),
            )
            updated = _fetch_arm_row(conn, experiment_id, arm_id)
        if updated is None:
            raise ExperimentArmNotFoundError(
                f"arm '{arm_id}' not found for experiment '{experiment_id}' after update"
            )
        return _build_arm_response(updated)

    def delete_arm(self, experiment_id: str, arm_id: str) -> ExperimentArmDeleteResponse:
        with self._connection() as conn:
            row = _fetch_arm_row(conn, experiment_id, arm_id)
            if row is None:
                _raise_missing_experiment_or_arm(conn, experiment_id, arm_id)
            assert row is not None
            selection_count = _fetch_arm_selection_count(conn, experiment_id, arm_id)
            if _requires_posterior_preservation(row, selection_count):
                end_dated_at = row["end_dated_at"] or _iso_utc(self._clock())
                conn.execute(
                    """
                    UPDATE experiments
                    SET enabled = 0, end_dated_at = ?, updated_at = ?
                    WHERE experiment_id = ? AND arm = ?
                    """,
                    (end_dated_at, _iso_utc(self._clock()), experiment_id, arm_id),
                )
                updated = _fetch_arm_row(conn, experiment_id, arm_id)
                if updated is None:
                    raise ExperimentArmNotFoundError(
                        f"arm '{arm_id}' not found for experiment "
                        f"'{experiment_id}' after delete guard"
                    )
                return ExperimentArmDeleteResponse(
                    experiment_id=experiment_id,
                    arm=arm_id,
                    deleted=False,
                    posterior_preserved=True,
                    reason="arm has posterior history; disabled instead of hard-deleting",
                    arm_state=_build_arm_response(updated),
                )
            conn.execute(
                """
                DELETE FROM experiments
                WHERE experiment_id = ? AND arm = ?
                """,
                (experiment_id, arm_id),
            )
        return ExperimentArmDeleteResponse(
            experiment_id=experiment_id,
            arm=arm_id,
            deleted=True,
            posterior_preserved=False,
            reason="unused arm hard-deleted",
            arm_state=None,
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            bootstrap_schema(conn)
            apply_writer_pragmas(conn)
            yield conn
        finally:
            conn.close()


def _experiment_exists(conn: sqlite3.Connection, experiment_id: str) -> bool:
    return _fetch_experiment_identity(conn, experiment_id) is not None


def _fetch_experiment_identity(
    conn: sqlite3.Connection,
    experiment_id: str,
) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT experiment_id, label
        FROM experiments
        WHERE experiment_id = ?
        LIMIT 1
        """,
        (experiment_id,),
    ).fetchone()
    return cast("sqlite3.Row | None", row)


def _fetch_experiment_rows(
    conn: sqlite3.Connection,
    experiment_id: str,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT experiment_id, label, arm, greeting_text, alpha_param, beta_param,
                   enabled, end_dated_at, updated_at
            FROM experiments
            WHERE experiment_id = ?
            ORDER BY arm
            """,
            (experiment_id,),
        ).fetchall()
    )


def _fetch_arm_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    arm_id: str,
) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT experiment_id, label, arm, greeting_text, alpha_param, beta_param,
               enabled, end_dated_at, updated_at
        FROM experiments
        WHERE experiment_id = ? AND arm = ?
        """,
        (experiment_id, arm_id),
    ).fetchone()
    return cast("sqlite3.Row | None", row)


def _fetch_arm_selection_count(
    conn: sqlite3.Connection,
    experiment_id: str,
    arm_id: str,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS selection_count
        FROM encounter_log
        WHERE experiment_id = ? AND arm = ?
        """,
        (experiment_id, arm_id),
    ).fetchone()
    return int(row["selection_count"] if row is not None else 0)


def _raise_missing_experiment_or_arm(
    conn: sqlite3.Connection,
    experiment_id: str,
    arm_id: str,
) -> None:
    if _fetch_experiment_identity(conn, experiment_id) is None:
        raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found")
    raise ExperimentArmNotFoundError(f"arm '{arm_id}' not found for experiment '{experiment_id}'")


def _requires_posterior_preservation(row: sqlite3.Row, selection_count: int) -> bool:
    if selection_count > 0:
        return True
    alpha = float(row["alpha_param"]) if row["alpha_param"] is not None else None
    beta = float(row["beta_param"]) if row["beta_param"] is not None else None
    if alpha is None or beta is None:
        return True
    return alpha != _BETA_PRIOR_ALPHA or beta != _BETA_PRIOR_BETA


def _build_arm_response(row: sqlite3.Row) -> ExperimentArmAdminResponse:
    return ExperimentArmAdminResponse(
        experiment_id=str(row["experiment_id"]),
        label=str(row["label"] or row["experiment_id"]),
        arm=str(row["arm"]),
        greeting_text=str(row["greeting_text"] or row["arm"]),
        alpha_param=float(row["alpha_param"]),
        beta_param=float(row["beta_param"]),
        enabled=bool(row["enabled"]),
        end_dated_at=_parse_datetime(row["end_dated_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["SqliteExperimentAdminService"]
