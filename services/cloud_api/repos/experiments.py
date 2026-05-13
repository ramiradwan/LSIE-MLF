"""Repository reads for signed cloud experiment bundles."""

from __future__ import annotations

from typing import Any

_SELECT_EXPERIMENT_ARMS_SQL = """
    SELECT
        id,
        experiment_id,
        label,
        arm AS arm_id,
        stimulus_definition,
        alpha_param AS posterior_alpha,
        beta_param AS posterior_beta,
        enabled,
        end_dated_at,
        updated_at,
        0 AS selection_count
    FROM experiments
    WHERE enabled = TRUE
      AND end_dated_at IS NULL
    ORDER BY experiment_id, arm
"""


def fetch_active_experiment_rows(cur: Any) -> list[dict[str, object]]:
    cur.execute(_SELECT_EXPERIMENT_ARMS_SQL)
    if cur.description is None:
        return []
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return [dict(zip(columns, row, strict=True)) for row in rows]
