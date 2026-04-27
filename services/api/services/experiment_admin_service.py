"""Experiment admin write service.

This service owns the additive mutation surface behind
`/api/v1/experiments/*` while keeping posterior-owned numeric state
read-only. New arms are always initialized at Beta(1,1); patches may
only change `greeting_text` and/or disable an arm. Explicit DELETE
requests hard-delete only unused prior arms and soft-disable arms that
must preserve posterior history.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from services.api.db.connection import get_connection, put_connection
from services.api.repos import experiments_queries as q

_BETA_PRIOR_ALPHA: float = 1.0
_BETA_PRIOR_BETA: float = 1.0


class ExperimentAdminError(Exception):
    """Base class for experiment admin write errors."""


class ExperimentAlreadyExistsError(ExperimentAdminError):
    """Raised when creating an experiment whose ID already exists."""


class ExperimentNotFoundError(ExperimentAdminError):
    """Raised when the target experiment does not exist."""


class ExperimentArmAlreadyExistsError(ExperimentAdminError):
    """Raised when creating an arm whose ID already exists."""


class ExperimentArmNotFoundError(ExperimentAdminError):
    """Raised when the target arm does not exist."""


class ExperimentMutationValidationError(ExperimentAdminError):
    """Raised when a requested mutation is outside the supported contract."""


class ExperimentAdminService:
    """Handles experiment and arm writes for the API layer."""

    def __init__(
        self,
        *,
        get_conn: Callable[[], Any] = get_connection,
        put_conn: Callable[[Any], None] = put_connection,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._get_conn = get_conn
        self._put_conn = put_conn
        self._clock = clock

    def create_experiment(self, request: ExperimentCreateRequest) -> ExperimentAdminResponse:
        """Create a new experiment and seed each arm at Beta(1,1)."""
        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                q.ensure_experiments_admin_schema(cur)
                existing = q.fetch_experiment_identity(cur, request.experiment_id)
                if existing is not None:
                    raise ExperimentAlreadyExistsError(
                        f"experiment '{request.experiment_id}' already exists"
                    )

                for arm in request.arms:
                    q.insert_experiment_arm(
                        cur,
                        experiment_id=request.experiment_id,
                        label=request.label,
                        arm=arm.arm,
                        greeting_text=arm.greeting_text,
                        alpha_param=_BETA_PRIOR_ALPHA,
                        beta_param=_BETA_PRIOR_BETA,
                        enabled=True,
                        end_dated_at=None,
                    )

                rows = q.fetch_experiment_admin_rows(cur, request.experiment_id)
            conn.commit()
            return self._build_experiment_response(
                experiment_id=request.experiment_id,
                label=request.label,
                rows=rows,
            )
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                self._put_conn(conn)

    def add_arm(
        self,
        experiment_id: str,
        request: ExperimentArmCreateRequest,
    ) -> ExperimentArmAdminResponse:
        """Add one new arm to an existing experiment at Beta(1,1)."""
        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                q.ensure_experiments_admin_schema(cur)
                identity = q.fetch_experiment_identity(cur, experiment_id)
                if identity is None:
                    raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found")

                existing_arm = q.fetch_experiment_arm_row(
                    cur,
                    experiment_id=experiment_id,
                    arm=request.arm,
                )
                if existing_arm is not None:
                    raise ExperimentArmAlreadyExistsError(
                        f"arm '{request.arm}' already exists for experiment '{experiment_id}'"
                    )

                label = str(identity.get("label") or experiment_id)
                q.insert_experiment_arm(
                    cur,
                    experiment_id=experiment_id,
                    label=label,
                    arm=request.arm,
                    greeting_text=request.greeting_text,
                    alpha_param=_BETA_PRIOR_ALPHA,
                    beta_param=_BETA_PRIOR_BETA,
                    enabled=True,
                    end_dated_at=None,
                )
                row = q.fetch_experiment_arm_row(cur, experiment_id=experiment_id, arm=request.arm)
            conn.commit()
            if row is None:
                raise ExperimentArmNotFoundError(
                    f"arm '{request.arm}' not found for experiment '{experiment_id}' after insert"
                )
            return self._build_arm_response(row)
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                self._put_conn(conn)

    def patch_arm(
        self,
        experiment_id: str,
        arm_id: str,
        request: ExperimentArmPatchRequest,
    ) -> ExperimentArmAdminResponse:
        """Patch supported arm metadata without touching posterior state."""
        if request.enabled is True:
            raise ExperimentMutationValidationError(
                "enabled=true is not supported; use enabled=false to disable an arm"
            )

        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                q.ensure_experiments_admin_schema(cur)
                row = q.fetch_experiment_arm_row(cur, experiment_id=experiment_id, arm=arm_id)
                if row is None:
                    identity = q.fetch_experiment_identity(cur, experiment_id)
                    if identity is None:
                        raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found")
                    raise ExperimentArmNotFoundError(
                        f"arm '{arm_id}' not found for experiment '{experiment_id}'"
                    )

                greeting_text = request.greeting_text or str(row.get("greeting_text") or arm_id)
                enabled = bool(row.get("enabled")) if row.get("enabled") is not None else True
                end_dated_at = row.get("end_dated_at")

                if request.enabled is False:
                    enabled = False
                    if end_dated_at is None:
                        end_dated_at = self._clock()

                q.update_experiment_arm_metadata(
                    cur,
                    experiment_id=experiment_id,
                    arm=arm_id,
                    greeting_text=greeting_text,
                    enabled=enabled,
                    end_dated_at=end_dated_at,
                )
                updated_row = q.fetch_experiment_arm_row(
                    cur,
                    experiment_id=experiment_id,
                    arm=arm_id,
                )
            conn.commit()
            if updated_row is None:
                raise ExperimentArmNotFoundError(
                    f"arm '{arm_id}' not found for experiment '{experiment_id}' after update"
                )
            return self._build_arm_response(updated_row)
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                self._put_conn(conn)

    def delete_arm(self, experiment_id: str, arm_id: str) -> ExperimentArmDeleteResponse:
        """Delete an unused arm or soft-disable one with posterior history.

        The service layer owns the posterior-preservation guard: any arm
        with observed selections, or with posterior parameters that no
        longer equal the Beta(1,1) prior, is *not* hard-deleted. Instead
        it is disabled and end-dated via the metadata-only update path so
        `alpha_param`, `beta_param`, and historical rollups remain intact.
        """
        conn: Any = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                q.ensure_experiments_admin_schema(cur)
                row = q.fetch_experiment_arm_row(cur, experiment_id=experiment_id, arm=arm_id)
                if row is None:
                    identity = q.fetch_experiment_identity(cur, experiment_id)
                    if identity is None:
                        raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found")
                    raise ExperimentArmNotFoundError(
                        f"arm '{arm_id}' not found for experiment '{experiment_id}'"
                    )

                selection_count = q.fetch_experiment_arm_selection_count(
                    cur,
                    experiment_id=experiment_id,
                    arm=arm_id,
                )
                if self._requires_posterior_preservation(row, selection_count):
                    greeting_text = str(row.get("greeting_text") or arm_id)
                    end_dated_at = row.get("end_dated_at") or self._clock()
                    q.update_experiment_arm_metadata(
                        cur,
                        experiment_id=experiment_id,
                        arm=arm_id,
                        greeting_text=greeting_text,
                        enabled=False,
                        end_dated_at=end_dated_at,
                    )
                    updated_row = q.fetch_experiment_arm_row(
                        cur,
                        experiment_id=experiment_id,
                        arm=arm_id,
                    )
                    if updated_row is None:
                        raise ExperimentArmNotFoundError(
                            "arm "
                            f"'{arm_id}' not found for experiment '{experiment_id}' "
                            "after delete guard"
                        )
                    result = ExperimentArmDeleteResponse(
                        experiment_id=experiment_id,
                        arm=arm_id,
                        deleted=False,
                        posterior_preserved=True,
                        reason="arm has posterior history; disabled instead of hard-deleting",
                        arm_state=self._build_arm_response(updated_row),
                    )
                else:
                    q.delete_experiment_arm(cur, experiment_id=experiment_id, arm=arm_id)
                    result = ExperimentArmDeleteResponse(
                        experiment_id=experiment_id,
                        arm=arm_id,
                        deleted=True,
                        posterior_preserved=False,
                        reason="unused arm hard-deleted",
                        arm_state=None,
                    )
            conn.commit()
            return result
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                self._put_conn(conn)

    @staticmethod
    def _requires_posterior_preservation(row: dict[str, Any], selection_count: int) -> bool:
        if selection_count > 0:
            return True
        alpha_raw = row.get("alpha_param")
        beta_raw = row.get("beta_param")
        if alpha_raw is None or beta_raw is None:
            return True
        try:
            alpha = float(alpha_raw)
            beta = float(beta_raw)
        except (TypeError, ValueError):
            return True
        return alpha != _BETA_PRIOR_ALPHA or beta != _BETA_PRIOR_BETA

    def _build_experiment_response(
        self,
        *,
        experiment_id: str,
        label: str,
        rows: list[dict[str, Any]],
    ) -> ExperimentAdminResponse:
        return ExperimentAdminResponse(
            experiment_id=experiment_id,
            label=label,
            arms=[self._build_arm_response(row) for row in rows],
        )

    def _build_arm_response(self, row: dict[str, Any]) -> ExperimentArmAdminResponse:
        alpha_param = row.get("alpha_param")
        beta_param = row.get("beta_param")
        return ExperimentArmAdminResponse(
            experiment_id=str(row["experiment_id"]),
            label=str(row.get("label") or row["experiment_id"]),
            arm=str(row["arm"]),
            greeting_text=str(row.get("greeting_text") or row["arm"]),
            alpha_param=float(alpha_param) if alpha_param is not None else _BETA_PRIOR_ALPHA,
            beta_param=float(beta_param) if beta_param is not None else _BETA_PRIOR_BETA,
            enabled=bool(row.get("enabled")) if row.get("enabled") is not None else True,
            end_dated_at=row.get("end_dated_at"),
            updated_at=row.get("updated_at"),
        )
