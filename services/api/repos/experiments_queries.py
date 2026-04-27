"""Query helpers for experiment state read/write endpoints.

The public API exposes two surfaces over the same `experiments` table:
  * the legacy read shape under `/api/v1/experiments`
  * additive admin writes for creating experiments/arms and patching
    human-owned arm metadata.

This module keeps raw SQL centralized and parameterized.
"""

from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------------
# Legacy read shape (preserved)
# ----------------------------------------------------------------------

_LIST_EXPERIMENTS_SQL: str = """
    SELECT DISTINCT experiment_id
    FROM experiments
    ORDER BY experiment_id
"""

_GET_EXPERIMENT_SQL: str = """
    SELECT experiment_id, arm, alpha_param, beta_param, updated_at
    FROM experiments
    WHERE experiment_id = %(experiment_id)s
    ORDER BY arm
"""

# ----------------------------------------------------------------------
# Additive admin surface
# ----------------------------------------------------------------------

_SELECT_EXPERIMENT_IDENTITY_SQL: str = """
    SELECT experiment_id, COALESCE(label, experiment_id) AS label
    FROM experiments
    WHERE experiment_id = %(experiment_id)s
    LIMIT 1
"""

_SELECT_EXPERIMENT_ADMIN_ROWS_SQL: str = """
    SELECT
        experiment_id,
        COALESCE(label, experiment_id) AS label,
        arm,
        COALESCE(greeting_text, arm) AS greeting_text,
        alpha_param,
        beta_param,
        COALESCE(enabled, TRUE) AS enabled,
        end_dated_at,
        updated_at
    FROM experiments
    WHERE experiment_id = %(experiment_id)s
    ORDER BY arm
"""

_SELECT_ARM_ADMIN_ROW_SQL: str = """
    SELECT
        experiment_id,
        COALESCE(label, experiment_id) AS label,
        arm,
        COALESCE(greeting_text, arm) AS greeting_text,
        alpha_param,
        beta_param,
        COALESCE(enabled, TRUE) AS enabled,
        end_dated_at,
        updated_at
    FROM experiments
    WHERE experiment_id = %(experiment_id)s
      AND arm = %(arm)s
    LIMIT 1
"""

_INSERT_EXPERIMENT_ARM_SQL: str = """
    INSERT INTO experiments (
        experiment_id,
        label,
        arm,
        greeting_text,
        alpha_param,
        beta_param,
        enabled,
        end_dated_at,
        updated_at
    )
    VALUES (
        %(experiment_id)s,
        %(label)s,
        %(arm)s,
        %(greeting_text)s,
        %(alpha_param)s,
        %(beta_param)s,
        %(enabled)s,
        %(end_dated_at)s,
        NOW()
    )
"""

_UPDATE_EXPERIMENT_ARM_METADATA_SQL: str = """
    UPDATE experiments
    SET greeting_text = %(greeting_text)s,
        enabled = %(enabled)s,
        end_dated_at = %(end_dated_at)s,
        updated_at = NOW()
    WHERE experiment_id = %(experiment_id)s
      AND arm = %(arm)s
"""

_SELECT_ARM_SELECTION_COUNT_SQL: str = """
    SELECT COUNT(*)::int AS selection_count
    FROM encounter_log
    WHERE experiment_id = %(experiment_id)s
      AND arm = %(arm)s
"""

_DELETE_EXPERIMENT_ARM_SQL: str = """
    DELETE FROM experiments
    WHERE experiment_id = %(experiment_id)s
      AND arm = %(arm)s
"""

# ----------------------------------------------------------------------
# Rollout-safe additive schema support
# ----------------------------------------------------------------------

# These statements are safe to run repeatedly. They let the admin write
# surface land against a legacy deployment before a dedicated migration
# step has been executed.
_ENSURE_EXPERIMENTS_ADMIN_SCHEMA_SQL: tuple[str, ...] = (
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS label TEXT",
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS greeting_text TEXT",
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS enabled BOOLEAN",
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS end_dated_at TIMESTAMPTZ",
    "UPDATE experiments SET label = experiment_id WHERE label IS NULL",
    "UPDATE experiments SET greeting_text = arm WHERE greeting_text IS NULL",
    "UPDATE experiments SET enabled = TRUE WHERE enabled IS NULL",
    "ALTER TABLE experiments ALTER COLUMN enabled SET DEFAULT TRUE",
    "ALTER TABLE experiments ALTER COLUMN enabled SET NOT NULL",
    "ALTER TABLE experiments ALTER COLUMN end_dated_at SET DEFAULT NULL",
)


# ----------------------------------------------------------------------
# Row helpers
# ----------------------------------------------------------------------


def _row_to_dict(cursor: Any) -> dict[str, Any] | None:
    if cursor.description is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    row: Any = cursor.fetchone()
    if row is None:
        return None
    return dict(zip(columns, row, strict=True))


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [dict(zip(columns, row, strict=True)) for row in rows]


# ----------------------------------------------------------------------
# Public fetchers / mutators
# ----------------------------------------------------------------------


def ensure_experiments_admin_schema(cursor: Any) -> None:
    """Best-effort additive schema upgrade for admin write paths."""
    for statement in _ENSURE_EXPERIMENTS_ADMIN_SCHEMA_SQL:
        cursor.execute(statement)


def fetch_experiment_ids(cursor: Any) -> list[dict[str, Any]]:
    cursor.execute(_LIST_EXPERIMENTS_SQL)
    return _rows_to_dicts(cursor)


def fetch_experiment_rows(cursor: Any, experiment_id: str) -> list[dict[str, Any]]:
    cursor.execute(_GET_EXPERIMENT_SQL, {"experiment_id": experiment_id})
    return _rows_to_dicts(cursor)


def fetch_experiment_identity(cursor: Any, experiment_id: str) -> dict[str, Any] | None:
    cursor.execute(_SELECT_EXPERIMENT_IDENTITY_SQL, {"experiment_id": experiment_id})
    return _row_to_dict(cursor)


def fetch_experiment_admin_rows(cursor: Any, experiment_id: str) -> list[dict[str, Any]]:
    cursor.execute(_SELECT_EXPERIMENT_ADMIN_ROWS_SQL, {"experiment_id": experiment_id})
    return _rows_to_dicts(cursor)


def fetch_experiment_arm_row(
    cursor: Any,
    *,
    experiment_id: str,
    arm: str,
) -> dict[str, Any] | None:
    cursor.execute(
        _SELECT_ARM_ADMIN_ROW_SQL,
        {
            "experiment_id": experiment_id,
            "arm": arm,
        },
    )
    return _row_to_dict(cursor)


def insert_experiment_arm(
    cursor: Any,
    *,
    experiment_id: str,
    label: str,
    arm: str,
    greeting_text: str,
    alpha_param: float,
    beta_param: float,
    enabled: bool,
    end_dated_at: Any,
) -> None:
    cursor.execute(
        _INSERT_EXPERIMENT_ARM_SQL,
        {
            "experiment_id": experiment_id,
            "label": label,
            "arm": arm,
            "greeting_text": greeting_text,
            "alpha_param": alpha_param,
            "beta_param": beta_param,
            "enabled": enabled,
            "end_dated_at": end_dated_at,
        },
    )


def update_experiment_arm_metadata(
    cursor: Any,
    *,
    experiment_id: str,
    arm: str,
    greeting_text: str,
    enabled: bool,
    end_dated_at: Any,
) -> None:
    cursor.execute(
        _UPDATE_EXPERIMENT_ARM_METADATA_SQL,
        {
            "experiment_id": experiment_id,
            "arm": arm,
            "greeting_text": greeting_text,
            "enabled": enabled,
            "end_dated_at": end_dated_at,
        },
    )


def fetch_experiment_arm_selection_count(
    cursor: Any,
    *,
    experiment_id: str,
    arm: str,
) -> int:
    cursor.execute(
        _SELECT_ARM_SELECTION_COUNT_SQL,
        {
            "experiment_id": experiment_id,
            "arm": arm,
        },
    )
    row = _row_to_dict(cursor)
    if row is None:
        return 0
    value = row.get("selection_count")
    return int(value) if value is not None else 0


def delete_experiment_arm(
    cursor: Any,
    *,
    experiment_id: str,
    arm: str,
) -> None:
    cursor.execute(
        _DELETE_EXPERIMENT_ARM_SQL,
        {
            "experiment_id": experiment_id,
            "arm": arm,
        },
    )
