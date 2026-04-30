"""
Operator query layer — raw DB row fetchers for the Operator Console aggregate endpoints.

These functions are the lowest layer of the Operator Console API stack:
they take a psycopg2 cursor and return plain row dicts. The higher
`OperatorReadService` composes them into Pydantic DTOs.

Design constraints:
  - Parameterized SQL only (§2 step 7, §5 data governance).
  - No Pydantic construction here — that is the service layer's job.
  - No raw biometric media (§5): only derived analytics tables are touched
    (`sessions`, `metrics`, `encounter_log`, `experiments`, `physiology_log`,
    `comodulation_log`, and the attribution ledger summary tables).
  - Keep functions small and composable — each answers one question the
    service layer needs.

Spec references:
  §2 step 7 — Parameterized INSERT/SELECT
  §4.E.1    — Encounter Log, experiment state
  §4.E.2    — Physiology persistence (per-role latest snapshot)
  §4.E.3    — Attribution analytics persistence
  §7B       — Thompson Sampling arm posteriors (experiments table)
  §7C       — Co-Modulation Index rows (comodulation_log)
  §7E       — Event→outcome attribution diagnostics
  §12       — Error-handling matrix (staleness heuristics used for alert synthesis)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

# Operator encounter rows project only canonical §7D observational-acoustic columns.
_CANONICAL_OBSERVATIONAL_ACOUSTIC_COLUMNS: tuple[str, ...] = (
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

_ACOUSTIC_METRICS_COLUMNS: tuple[str, ...] = _CANONICAL_OBSERVATIONAL_ACOUSTIC_COLUMNS

_ACOUSTIC_METRICS_SELECT_SQL: str = ",\n        ".join(
    ["acoustic.metrics_row_id", *[f"acoustic.{column}" for column in _ACOUSTIC_METRICS_COLUMNS]]
)

_ACOUSTIC_METRICS_LATERAL_SQL: str = ",\n            ".join(
    [
        "m.id AS metrics_row_id",
        *[f"m.{column}" for column in _ACOUSTIC_METRICS_COLUMNS],
    ]
)

# §7E / §6.4 semantic-attribution projections are read additively from the
# attribution ledger and pivoted onto the existing encounter aggregate rows.
# Prefix the service-layer source keys to avoid colliding with encounter reward
# fields such as encounter_log.semantic_gate and encounter_log.au12_baseline_pre.
_SEMANTIC_ATTRIBUTION_SELECT_SQL: str = """
        attr.semantic_reason_code AS semantic_reasoning,
        CASE
            WHEN attr.event_id IS NULL THEN NULL::boolean
            ELSE (e.semantic_gate = 1)
        END AS semantic_is_match,
        attr.semantic_p_match AS semantic_confidence_score,
        attr.semantic_method AS semantic_method,
        attr.semantic_method_version AS semantic_method_version,
        attr.finality AS attribution_finality,
        attribution_scores.soft_reward_candidate,
        CASE
            WHEN attr.event_id IS NULL THEN NULL::double precision
            ELSE e.au12_baseline_pre
        END AS attribution_au12_baseline_pre,
        attribution_scores.au12_lift_p90,
        attribution_scores.au12_lift_peak,
        attribution_scores.au12_peak_latency_ms,
        attribution_scores.sync_peak_corr,
        attribution_scores.sync_peak_lag,
        outcome_link.lag_s AS outcome_link_lag_s
""".strip()

_ATTRIBUTION_EVENT_LATERAL_SQL: str = """
    LEFT JOIN LATERAL (
        SELECT
            ae.event_id,
            ae.semantic_reason_code,
            ae.semantic_p_match,
            ae.semantic_method,
            ae.semantic_method_version,
            ae.finality,
            ae.created_at
        FROM attribution_event ae
        WHERE ae.session_id = e.session_id
          AND ae.segment_id = e.segment_id
          AND ae.event_type = 'greeting_interaction'
        ORDER BY ae.created_at DESC, ae.event_id DESC
        LIMIT 1
    ) attr ON TRUE
"""

_ATTRIBUTION_SCORE_LATERAL_SQL: str = """
    LEFT JOIN LATERAL (
        SELECT
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'soft_reward_candidate'
            ) AS soft_reward_candidate,
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'au12_lift_p90'
            ) AS au12_lift_p90,
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'au12_lift_peak'
            ) AS au12_lift_peak,
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'au12_peak_latency_ms'
            ) AS au12_peak_latency_ms,
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'sync_peak_corr'
            ) AS sync_peak_corr,
            MAX(s.score_raw) FILTER (
                WHERE s.attribution_method = 'sync_peak_lag'
            ) AS sync_peak_lag
        FROM attribution_score s
        WHERE s.event_id = attr.event_id
    ) attribution_scores ON TRUE
"""

_OUTCOME_LINK_LATERAL_SQL: str = """
    LEFT JOIN LATERAL (
        SELECT link.lag_s
        FROM event_outcome_link link
        WHERE link.event_id = attr.event_id
        ORDER BY link.created_at DESC, link.lag_s ASC
        LIMIT 1
    ) outcome_link ON TRUE
"""

# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------

# §4.E.1 — Recent sessions with the last encounter reward + active arm.
# The LEFT JOIN picks the newest encounter row per session using a
# correlated subquery so a session with no encounters still appears.
_LIST_RECENT_SESSIONS_SQL: str = """
    SELECT
        s.session_id,
        s.started_at,
        s.ended_at,
        EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at))::double precision
            AS duration_s,
        latest_enc.experiment_id,
        latest_enc.arm AS active_arm,
        latest_enc.timestamp_utc AS last_segment_completed_at_utc,
        latest_enc.gated_reward AS latest_reward,
        latest_enc.semantic_gate AS latest_semantic_gate,
        NULL::boolean AS is_calibrating,
        NULL::integer AS calibration_frames_accumulated,
        NULL::integer AS calibration_frames_required
    FROM sessions s
    LEFT JOIN LATERAL (
        SELECT e.experiment_id, e.arm, e.timestamp_utc,
               e.gated_reward, e.semantic_gate
        FROM encounter_log e
        WHERE e.session_id = s.session_id
        ORDER BY e.timestamp_utc DESC
        LIMIT 1
    ) latest_enc ON TRUE
    ORDER BY s.started_at DESC
    LIMIT %(limit)s
"""

# Single-session lookup variant.
_GET_SESSION_SQL: str = """
    SELECT
        s.session_id,
        s.started_at,
        s.ended_at,
        EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at))::double precision
            AS duration_s,
        latest_enc.experiment_id,
        latest_enc.arm AS active_arm,
        latest_enc.timestamp_utc AS last_segment_completed_at_utc,
        latest_enc.gated_reward AS latest_reward,
        latest_enc.semantic_gate AS latest_semantic_gate,
        NULL::boolean AS is_calibrating,
        NULL::integer AS calibration_frames_accumulated,
        NULL::integer AS calibration_frames_required
    FROM sessions s
    LEFT JOIN LATERAL (
        SELECT e.experiment_id, e.arm, e.timestamp_utc,
               e.gated_reward, e.semantic_gate
        FROM encounter_log e
        WHERE e.session_id = s.session_id
        ORDER BY e.timestamp_utc DESC
        LIMIT 1
    ) latest_enc ON TRUE
    WHERE s.session_id = %(session_id)s
"""

# The most recently started session that has not ended — used on Overview.
_GET_ACTIVE_SESSION_SQL: str = """
    SELECT
        s.session_id,
        s.started_at,
        s.ended_at,
        EXTRACT(EPOCH FROM (NOW() - s.started_at))::double precision AS duration_s,
        latest_enc.experiment_id,
        latest_enc.arm AS active_arm,
        latest_enc.timestamp_utc AS last_segment_completed_at_utc,
        latest_enc.gated_reward AS latest_reward,
        latest_enc.semantic_gate AS latest_semantic_gate,
        NULL::boolean AS is_calibrating,
        NULL::integer AS calibration_frames_accumulated,
        NULL::integer AS calibration_frames_required
    FROM sessions s
    LEFT JOIN LATERAL (
        SELECT e.experiment_id, e.arm, e.timestamp_utc,
               e.gated_reward, e.semantic_gate
        FROM encounter_log e
        WHERE e.session_id = s.session_id
        ORDER BY e.timestamp_utc DESC
        LIMIT 1
    ) latest_enc ON TRUE
    WHERE s.ended_at IS NULL
    ORDER BY s.started_at DESC
    LIMIT 1
"""

# ----------------------------------------------------------------------
# Encounter log (§4.E.1)
# ----------------------------------------------------------------------

# Full reward-explanation columns per §7B — `p90_intensity`, `semantic_gate`,
# `gated_reward`, `n_frames_in_window`, `au12_baseline_pre`, `stimulus_time`.
_SESSION_ENCOUNTERS_SQL: str = f"""
    SELECT
        e.id, e.session_id, e.segment_id, e.experiment_id,
        e.arm, e.timestamp_utc, e.gated_reward, e.p90_intensity,
        e.semantic_gate, e.n_frames_in_window,
        e.au12_baseline_pre, e.stimulus_time, e.created_at,
        {_ACOUSTIC_METRICS_SELECT_SQL},
        {_SEMANTIC_ATTRIBUTION_SELECT_SQL}
    FROM encounter_log e
    LEFT JOIN LATERAL (
        SELECT
            {_ACOUSTIC_METRICS_LATERAL_SQL}
        FROM metrics m
        WHERE m.session_id = e.session_id
          AND m.segment_id = e.segment_id
        ORDER BY m.created_at DESC, m.id DESC
        LIMIT 1
    ) acoustic ON TRUE
    {_ATTRIBUTION_EVENT_LATERAL_SQL}
    {_ATTRIBUTION_SCORE_LATERAL_SQL}
    {_OUTCOME_LINK_LATERAL_SQL}
    WHERE e.session_id = %(session_id)s
      AND (%(before_utc)s::timestamptz IS NULL OR e.timestamp_utc < %(before_utc)s)
    ORDER BY e.timestamp_utc DESC
    LIMIT %(limit)s
"""

# Latest encounter for a session (used on Overview).
_LATEST_ENCOUNTER_SQL: str = f"""
    SELECT
        e.id, e.session_id, e.segment_id, e.experiment_id,
        e.arm, e.timestamp_utc, e.gated_reward, e.p90_intensity,
        e.semantic_gate, e.n_frames_in_window,
        e.au12_baseline_pre, e.stimulus_time, e.created_at,
        {_ACOUSTIC_METRICS_SELECT_SQL},
        {_SEMANTIC_ATTRIBUTION_SELECT_SQL}
    FROM encounter_log e
    LEFT JOIN LATERAL (
        SELECT
            {_ACOUSTIC_METRICS_LATERAL_SQL}
        FROM metrics m
        WHERE m.session_id = e.session_id
          AND m.segment_id = e.segment_id
        ORDER BY m.created_at DESC, m.id DESC
        LIMIT 1
    ) acoustic ON TRUE
    {_ATTRIBUTION_EVENT_LATERAL_SQL}
    {_ATTRIBUTION_SCORE_LATERAL_SQL}
    {_OUTCOME_LINK_LATERAL_SQL}
    WHERE e.session_id = %(session_id)s
    ORDER BY e.timestamp_utc DESC
    LIMIT 1
"""

# ----------------------------------------------------------------------
# Experiments (§7B)
# ----------------------------------------------------------------------

# All arms for an experiment, including rollup counts from encounter_log.
# The management columns are additive. Build the projection after checking
# information_schema so pre-migration deployments keep serving operator reads
# instead of failing parse-time on missing columns.
_EXPERIMENT_MANAGEMENT_COLUMNS: tuple[str, ...] = (
    "label",
    "greeting_text",
    "enabled",
    "end_dated_at",
)

_EXPERIMENT_MANAGEMENT_COLUMNS_SQL: str = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'experiments'
      AND column_name IN ('label', 'greeting_text', 'enabled', 'end_dated_at')
"""

# Identify the most-recently-selected arm for an experiment from encounter_log —
# this becomes the "active" arm shown on the Overview/Experiments surfaces.
_ACTIVE_ARM_FOR_EXPERIMENT_SQL: str = """
    SELECT e.arm, e.timestamp_utc
    FROM encounter_log e
    WHERE e.experiment_id = %(experiment_id)s
    ORDER BY e.timestamp_utc DESC
    LIMIT 1
"""

# ----------------------------------------------------------------------
# Physiology (§4.E.2, §7C)
# ----------------------------------------------------------------------

# Latest row per `subject_role` for a session.
_LATEST_PHYSIO_SQL: str = """
    SELECT DISTINCT ON (p.subject_role)
        p.session_id, p.segment_id, p.subject_role, p.rmssd_ms,
        p.heart_rate_bpm, p.freshness_s, p.is_stale, p.provider,
        p.source_timestamp_utc, p.created_at
    FROM physiology_log p
    WHERE p.session_id = %(session_id)s
    ORDER BY p.subject_role, p.created_at DESC
"""

# Latest co-modulation row for a session. §7C: `co_modulation_index` can be
# null when insufficient aligned non-stale pairs exist.
_LATEST_COMOD_SQL: str = """
    SELECT
        c.session_id, c.window_end_utc, c.window_minutes,
        c.co_modulation_index, c.n_paired_observations,
        c.coverage_ratio, c.streamer_rmssd_mean,
        c.operator_rmssd_mean, c.created_at
    FROM comodulation_log c
    WHERE c.session_id = %(session_id)s
    ORDER BY c.window_end_utc DESC
    LIMIT 1
"""

# ----------------------------------------------------------------------
# Health heuristics
# ----------------------------------------------------------------------

# Most-recent `created_at` across the tables that correspond to live
# subsystems. The service layer reads this, compares to the current clock,
# and synthesises HealthSubsystemStatus rows. §12 permits degraded/recovering
# states alongside error — staleness → degraded, silence → unknown.
_SUBSYSTEM_PULSE_SQL: str = """
    SELECT
        (SELECT MAX(m.created_at) FROM metrics m)          AS last_metric_at,
        (SELECT MAX(p.created_at) FROM physiology_log p)   AS last_physio_at,
        (SELECT MAX(c.created_at) FROM comodulation_log c) AS last_comod_at,
        (SELECT MAX(e.created_at) FROM encounter_log e)    AS last_encounter_at
"""

# Physiology-staleness signal used to synthesise PHYSIOLOGY_STALE alerts.
_RECENT_STALE_PHYSIO_SQL: str = """
    SELECT
        p.session_id,
        p.subject_role,
        p.created_at,
        p.freshness_s
    FROM physiology_log p
    WHERE p.is_stale = TRUE
      AND p.created_at >= NOW() - INTERVAL '1 hour'
      AND (%(since_utc)s::timestamptz IS NULL OR p.created_at >= %(since_utc)s)
    ORDER BY p.created_at DESC
    LIMIT %(limit)s
"""

# Recently-ended sessions fuel SESSION_ENDED alerts.
_RECENTLY_ENDED_SESSIONS_SQL: str = """
    SELECT s.session_id, s.ended_at
    FROM sessions s
    WHERE s.ended_at IS NOT NULL
      AND s.ended_at >= NOW() - INTERVAL '1 hour'
      AND (%(since_utc)s::timestamptz IS NULL OR s.ended_at >= %(since_utc)s)
    ORDER BY s.ended_at DESC
    LIMIT %(limit)s
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


def _available_experiment_management_columns(cursor: Any) -> set[str]:
    """Return additive experiment-management columns present on this schema."""
    cursor.execute(_EXPERIMENT_MANAGEMENT_COLUMNS_SQL)
    return {str(row[0]) for row in cursor.fetchall()}


def _experiment_arms_sql(available_columns: set[str]) -> str:
    """Build a rollout-safe experiment-arm projection.

    PostgreSQL parses every referenced column before evaluating CASE or
    COALESCE, so legacy compatibility has to remove missing column names
    from the statement entirely rather than hiding them in expressions.
    """
    label_expr = (
        "COALESCE(ex.label, ex.experiment_id)"
        if "label" in available_columns
        else "ex.experiment_id"
    )
    greeting_expr = (
        "COALESCE(ex.greeting_text, ex.arm)" if "greeting_text" in available_columns else "ex.arm"
    )
    enabled_expr = "COALESCE(ex.enabled, TRUE)" if "enabled" in available_columns else "TRUE"
    end_dated_expr = (
        "ex.end_dated_at" if "end_dated_at" in available_columns else "NULL::timestamptz"
    )
    return f"""
    SELECT
        ex.experiment_id,
        {label_expr} AS label,
        ex.arm,
        {greeting_expr} AS greeting_text,
        ex.alpha_param,
        ex.beta_param,
        {enabled_expr} AS enabled,
        {end_dated_expr} AS end_dated_at,
        ex.updated_at,
        rollup.selection_count,
        rollup.recent_reward_mean,
        rollup.recent_semantic_pass_rate
    FROM experiments ex
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS selection_count,
            AVG(e.gated_reward)::double precision AS recent_reward_mean,
            AVG(e.semantic_gate::numeric)::double precision AS recent_semantic_pass_rate
        FROM encounter_log e
        WHERE e.experiment_id = ex.experiment_id
          AND e.arm = ex.arm
    ) rollup ON TRUE
    WHERE ex.experiment_id = %(experiment_id)s
    ORDER BY ex.arm
"""


# ----------------------------------------------------------------------
# Public fetchers — each takes a cursor and returns plain row dicts.
# ----------------------------------------------------------------------


def fetch_recent_sessions(cursor: Any, *, limit: int) -> list[dict[str, Any]]:
    cursor.execute(_LIST_RECENT_SESSIONS_SQL, {"limit": limit})
    return _rows_to_dicts(cursor)


def fetch_session_by_id(cursor: Any, session_id: UUID) -> dict[str, Any] | None:
    cursor.execute(_GET_SESSION_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_active_session(cursor: Any) -> dict[str, Any] | None:
    cursor.execute(_GET_ACTIVE_SESSION_SQL)
    return _row_to_dict(cursor)


def fetch_session_encounters(
    cursor: Any,
    session_id: UUID,
    *,
    limit: int,
    before_utc: datetime | None,
) -> list[dict[str, Any]]:
    cursor.execute(
        _SESSION_ENCOUNTERS_SQL,
        {
            "session_id": str(session_id),
            "limit": limit,
            "before_utc": before_utc,
        },
    )
    return _rows_to_dicts(cursor)


def fetch_latest_encounter(cursor: Any, session_id: UUID) -> dict[str, Any] | None:
    cursor.execute(_LATEST_ENCOUNTER_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_experiment_arms(cursor: Any, experiment_id: str) -> list[dict[str, Any]]:
    available_columns = _available_experiment_management_columns(cursor)
    cursor.execute(_experiment_arms_sql(available_columns), {"experiment_id": experiment_id})
    return _rows_to_dicts(cursor)


def fetch_active_arm_for_experiment(cursor: Any, experiment_id: str) -> dict[str, Any] | None:
    cursor.execute(_ACTIVE_ARM_FOR_EXPERIMENT_SQL, {"experiment_id": experiment_id})
    return _row_to_dict(cursor)


def fetch_latest_physiology_rows(cursor: Any, session_id: UUID) -> list[dict[str, Any]]:
    cursor.execute(_LATEST_PHYSIO_SQL, {"session_id": str(session_id)})
    return _rows_to_dicts(cursor)


def fetch_latest_comodulation_row(cursor: Any, session_id: UUID) -> dict[str, Any] | None:
    cursor.execute(_LATEST_COMOD_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_subsystem_pulse(cursor: Any) -> dict[str, Any]:
    cursor.execute(_SUBSYSTEM_PULSE_SQL)
    row = _row_to_dict(cursor)
    return row if row is not None else {}


def fetch_recent_stale_physiology(
    cursor: Any, *, since_utc: datetime | None, limit: int
) -> list[dict[str, Any]]:
    cursor.execute(
        _RECENT_STALE_PHYSIO_SQL,
        {"since_utc": since_utc, "limit": limit},
    )
    return _rows_to_dicts(cursor)


def fetch_recently_ended_sessions(
    cursor: Any, *, since_utc: datetime | None, limit: int
) -> list[dict[str, Any]]:
    cursor.execute(
        _RECENTLY_ENDED_SESSIONS_SQL,
        {"since_utc": since_utc, "limit": limit},
    )
    return _rows_to_dicts(cursor)
