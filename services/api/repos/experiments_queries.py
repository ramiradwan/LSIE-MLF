"""Query helpers for experiment state read/write endpoints.

The public API exposes two surfaces over the same `experiments` table:
  * the legacy read shape under `/api/v1/experiments`
  * additive admin writes for creating experiments/arms and patching
    human-owned arm metadata.

This module keeps raw SQL centralized and parameterized.
"""

from __future__ import annotations

import json
from typing import Any

from packages.schemas.data_tiers import DataTier, mark_data_tier
from packages.schemas.evaluation import StimulusDefinition

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
        stimulus_definition,
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
        stimulus_definition,
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

_INSERT_EXPERIMENT_ARM_SQL: str = mark_data_tier(
    """
    INSERT INTO experiments (
        experiment_id,
        label,
        arm,
        stimulus_definition,
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
        %(stimulus_definition)s,
        %(alpha_param)s,
        %(beta_param)s,
        %(enabled)s,
        %(end_dated_at)s,
        NOW()
    )
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
    purpose="Experiment arm analytical configuration INSERT",
)  # §5.2.3 Permanent Analytical Storage

_UPDATE_EXPERIMENT_ARM_METADATA_SQL: str = """
    UPDATE experiments
    SET stimulus_definition = %(stimulus_definition)s,
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


def _encode_stimulus_definition(stimulus_definition: StimulusDefinition) -> str:
    return json.dumps(
        stimulus_definition.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_stimulus_definition(value: Any) -> StimulusDefinition:
    if isinstance(value, StimulusDefinition):
        return value
    if isinstance(value, str):
        return StimulusDefinition.model_validate_json(value)
    return StimulusDefinition.model_validate(value)


def _decode_admin_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    decoded["stimulus_definition"] = _decode_stimulus_definition(row.get("stimulus_definition"))
    return decoded


# ----------------------------------------------------------------------
# Public fetchers / mutators
# ----------------------------------------------------------------------


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
    return [_decode_admin_row(row) for row in _rows_to_dicts(cursor)]


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
    row = _row_to_dict(cursor)
    return _decode_admin_row(row) if row is not None else None


def insert_experiment_arm(
    cursor: Any,
    *,
    experiment_id: str,
    label: str,
    arm: str,
    stimulus_definition: StimulusDefinition,
    alpha_param: float,
    beta_param: float,
    enabled: bool,
    end_dated_at: Any,
) -> None:
    cursor.execute(
        _INSERT_EXPERIMENT_ARM_SQL,
        mark_data_tier(
            {
                "experiment_id": experiment_id,
                "label": label,
                "arm": arm,
                "stimulus_definition": _encode_stimulus_definition(stimulus_definition),
                "alpha_param": alpha_param,
                "beta_param": beta_param,
                "enabled": enabled,
                "end_dated_at": end_dated_at,
            },
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Normalized experiment arm configuration row parameters",
        ),
    )


def update_experiment_arm_metadata(
    cursor: Any,
    *,
    experiment_id: str,
    arm: str,
    stimulus_definition: StimulusDefinition,
    enabled: bool,
    end_dated_at: Any,
) -> None:
    cursor.execute(
        _UPDATE_EXPERIMENT_ARM_METADATA_SQL,
        {
            "experiment_id": experiment_id,
            "arm": arm,
            "stimulus_definition": _encode_stimulus_definition(stimulus_definition),
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
