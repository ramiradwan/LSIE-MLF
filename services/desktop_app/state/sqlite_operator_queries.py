"""SQLite-flavored operator query layer.

This is a desktop loopback read-model backend, not an Operator Console
import surface. The console calls FastAPI through ``ApiClient``; dependency
overrides inject these query functions behind ``OperatorReadService``.

Mirrors :mod:`services.api.repos.operator_queries` but produces SQLite
SQL and accepts a :class:`sqlite3.Cursor`. Function names, parameter
shapes, and returned row-dict keys are identical so
:class:`services.api.services.operator_read_service.OperatorReadService`
can swap query backends without touching its DTO assembly path.

Key translation rules from the PostgreSQL repo:

* ``%(name)s`` → ``:name`` parameter style.
* ``LATERAL JOIN ... LIMIT 1`` → ``LEFT JOIN`` against a window-function
  CTE (``ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)``). SQLite
  3.25+ ships with window functions and Python 3.11+ embeds a recent
  enough SQLite, so the rewrite is straight-line.
* ``EXTRACT(EPOCH FROM (NOW() - x))`` → ``(julianday('now') - julianday(x)) * 86400.0``.
* ``NOW() - INTERVAL '1 hour'`` → ``datetime('now','-1 hour')`` and the
  comparison stays a string compare since ISO-8601 sorts lexically.
* ``::boolean``, ``::double precision``, ``::integer`` casts → drop;
  the row dict carries the raw column type and the
  ``OperatorReadService`` builder coerces.
* ``MAX(x) FILTER (WHERE ...)`` → ``MAX(CASE WHEN ... THEN x END)``.
* The ``information_schema.columns`` rollout-safety probe returns the
  full set unconditionally — the SQLite schema is owned by
  :mod:`services.desktop_app.state.sqlite_schema` and ships with all
  experiment-management columns from day one.

The SQLite ``encounter_log`` row stores ``stimulus_time`` as ``REAL``
(epoch seconds) and ``timestamp_utc`` as ``TEXT``; the ``metrics`` row
stores ``f0_valid_*`` / ``perturbation_valid_*`` as ``INTEGER`` (0/1).
Builder coercion in ``OperatorReadService`` already handles those.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any, TypeAlias
from uuid import UUID

OperatorMarkerValue: TypeAlias = str | int | float | None
OperatorChangeMarker: TypeAlias = dict[str, OperatorMarkerValue]

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

_EXPERIMENT_MANAGEMENT_COLUMNS: tuple[str, ...] = (
    "label",
    "greeting_text",
    "enabled",
    "end_dated_at",
)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


_LATEST_ENCOUNTER_PER_SESSION_CTE: str = """
WITH latest_enc AS (
    SELECT
        e.session_id,
        e.experiment_id,
        e.arm,
        e.timestamp_utc,
        e.gated_reward,
        e.semantic_gate,
        ROW_NUMBER() OVER (
            PARTITION BY e.session_id
            ORDER BY e.timestamp_utc DESC
        ) AS rn
    FROM encounter_log e
)
"""


_LIST_RECENT_SESSIONS_SQL: str = (
    _LATEST_ENCOUNTER_PER_SESSION_CTE
    + """
SELECT
    s.session_id                                       AS session_id,
    s.started_at                                       AS started_at,
    s.ended_at                                         AS ended_at,
    (julianday(COALESCE(s.ended_at, datetime('now'))) - julianday(s.started_at)) * 86400.0
                                                       AS duration_s,
    COALESCE(le.experiment_id, s.experiment_id)       AS experiment_id,
    COALESCE(live.active_arm, s.active_arm, le.arm)    AS active_arm,
    COALESCE(live.expected_greeting, s.expected_greeting)
                                                           AS expected_greeting,
    le.timestamp_utc                                   AS last_segment_completed_at_utc,
    le.gated_reward                                    AS latest_reward,
    le.semantic_gate                                   AS latest_semantic_gate,
    live.is_calibrating                                AS is_calibrating,
    live.calibration_frames_accumulated                AS calibration_frames_accumulated,
    live.calibration_frames_required                   AS calibration_frames_required,
    (
        SELECT COUNT(*)
        FROM sessions active
        WHERE active.ended_at IS NULL
    )                                                  AS active_session_count
FROM sessions s
LEFT JOIN latest_enc le ON le.session_id = s.session_id AND le.rn = 1
LEFT JOIN live_session_state live ON live.session_id = s.session_id
ORDER BY s.started_at DESC
LIMIT :limit
"""
)


_GET_SESSION_SQL: str = (
    _LATEST_ENCOUNTER_PER_SESSION_CTE
    + """
SELECT
    s.session_id                                       AS session_id,
    s.started_at                                       AS started_at,
    s.ended_at                                         AS ended_at,
    (julianday(COALESCE(s.ended_at, datetime('now'))) - julianday(s.started_at)) * 86400.0
                                                       AS duration_s,
    COALESCE(le.experiment_id, s.experiment_id)       AS experiment_id,
    COALESCE(live.active_arm, s.active_arm, le.arm)    AS active_arm,
    COALESCE(live.expected_greeting, s.expected_greeting)
                                                           AS expected_greeting,
    le.timestamp_utc                                   AS last_segment_completed_at_utc,
    le.gated_reward                                    AS latest_reward,
    le.semantic_gate                                   AS latest_semantic_gate,
    live.is_calibrating                                AS is_calibrating,
    live.calibration_frames_accumulated                AS calibration_frames_accumulated,
    live.calibration_frames_required                   AS calibration_frames_required,
    (
        SELECT COUNT(*)
        FROM sessions active
        WHERE active.ended_at IS NULL
    )                                                  AS active_session_count
FROM sessions s
LEFT JOIN latest_enc le ON le.session_id = s.session_id AND le.rn = 1
LEFT JOIN live_session_state live ON live.session_id = s.session_id
WHERE s.session_id = :session_id
"""
)


_GET_ACTIVE_SESSION_SQL: str = (
    _LATEST_ENCOUNTER_PER_SESSION_CTE
    + """
SELECT
    s.session_id                                       AS session_id,
    s.started_at                                       AS started_at,
    s.ended_at                                         AS ended_at,
    (julianday(datetime('now')) - julianday(s.started_at)) * 86400.0
                                                       AS duration_s,
    COALESCE(le.experiment_id, s.experiment_id)       AS experiment_id,
    COALESCE(live.active_arm, s.active_arm, le.arm)    AS active_arm,
    COALESCE(live.expected_greeting, s.expected_greeting)
                                                           AS expected_greeting,
    le.timestamp_utc                                   AS last_segment_completed_at_utc,
    le.gated_reward                                    AS latest_reward,
    le.semantic_gate                                   AS latest_semantic_gate,
    live.is_calibrating                                AS is_calibrating,
    live.calibration_frames_accumulated                AS calibration_frames_accumulated,
    live.calibration_frames_required                   AS calibration_frames_required,
    (
        SELECT COUNT(*)
        FROM sessions active
        WHERE active.ended_at IS NULL
    )                                                  AS active_session_count
FROM sessions s
LEFT JOIN latest_enc le ON le.session_id = s.session_id AND le.rn = 1
LEFT JOIN live_session_state live ON live.session_id = s.session_id
WHERE s.ended_at IS NULL
ORDER BY s.started_at DESC
LIMIT 1
"""
)


# ---------------------------------------------------------------------------
# Encounter log (§4.E.1) — with acoustic and attribution lateral joins
# ---------------------------------------------------------------------------


def _acoustic_select_columns(prefix: str) -> str:
    return ",\n        ".join(
        [f"{prefix}.metrics_row_id"]
        + [f"{prefix}.{column}" for column in _CANONICAL_OBSERVATIONAL_ACOUSTIC_COLUMNS]
    )


_ACOUSTIC_LATEST_PER_SEGMENT_CTE: str = (
    """
acoustic_latest AS (
    SELECT
        m.session_id,
        m.segment_id,
        m.id AS metrics_row_id,
        """
    + ",\n        ".join(f"m.{column}" for column in _CANONICAL_OBSERVATIONAL_ACOUSTIC_COLUMNS)
    + """,
        ROW_NUMBER() OVER (
            PARTITION BY m.session_id, m.segment_id
            ORDER BY m.created_at DESC, m.id DESC
        ) AS rn
    FROM metrics m
),
transcript_latest AS (
    SELECT
        t.session_id,
        t.segment_id,
        t.text AS transcription,
        ROW_NUMBER() OVER (
            PARTITION BY t.session_id, t.segment_id
            ORDER BY t.created_at DESC, t.id DESC
        ) AS rn
    FROM transcripts t
)
"""
)


_ATTRIBUTION_EVENT_LATEST_PER_SEGMENT_CTE: str = """
attribution_event_latest AS (
    SELECT
        ae.session_id,
        ae.segment_id,
        ae.event_id,
        ae.semantic_reason_code,
        ae.semantic_p_match,
        ae.semantic_method,
        ae.semantic_method_version,
        ae.finality,
        ae.created_at,
        ROW_NUMBER() OVER (
            PARTITION BY ae.session_id, ae.segment_id
            ORDER BY ae.created_at DESC, ae.event_id DESC
        ) AS rn
    FROM attribution_event ae
    WHERE ae.event_type = 'greeting_interaction'
)
"""


_ATTRIBUTION_SCORES_BY_EVENT_CTE: str = """
attribution_scores_by_event AS (
    SELECT
        s.event_id,
        MAX(CASE WHEN s.attribution_method = 'soft_reward_candidate' THEN s.score_raw END)
            AS soft_reward_candidate,
        MAX(CASE WHEN s.attribution_method = 'au12_lift_p90' THEN s.score_raw END)
            AS au12_lift_p90,
        MAX(CASE WHEN s.attribution_method = 'au12_lift_peak' THEN s.score_raw END)
            AS au12_lift_peak,
        MAX(CASE WHEN s.attribution_method = 'au12_peak_latency_ms' THEN s.score_raw END)
            AS au12_peak_latency_ms,
        MAX(CASE WHEN s.attribution_method = 'sync_peak_corr' THEN s.score_raw END)
            AS sync_peak_corr,
        MAX(CASE WHEN s.attribution_method = 'sync_peak_lag' THEN s.score_raw END)
            AS sync_peak_lag
    FROM attribution_score s
    GROUP BY s.event_id
)
"""


_OUTCOME_LINK_LATEST_BY_EVENT_CTE: str = """
outcome_link_latest AS (
    SELECT
        link.event_id,
        link.lag_s,
        ROW_NUMBER() OVER (
            PARTITION BY link.event_id
            ORDER BY link.created_at DESC, link.lag_s ASC
        ) AS rn
    FROM event_outcome_link link
)
"""


_ENCOUNTERS_CTE_BLOCK: str = (
    "WITH "
    + _ACOUSTIC_LATEST_PER_SEGMENT_CTE
    + ",\n"
    + _ATTRIBUTION_EVENT_LATEST_PER_SEGMENT_CTE
    + ",\n"
    + _ATTRIBUTION_SCORES_BY_EVENT_CTE
    + ",\n"
    + _OUTCOME_LINK_LATEST_BY_EVENT_CTE
)


_ENCOUNTERS_PROJECTION: str = (
    """
        e.id, e.session_id, e.segment_id, e.experiment_id,
        e.arm, e.timestamp_utc, e.gated_reward, e.p90_intensity,
        e.semantic_gate, e.n_frames_in_window,
        e.au12_baseline_pre, e.stimulus_time, e.created_at,
        """
    + _acoustic_select_columns("acoustic")
    + """,
        attr.semantic_reason_code AS semantic_reasoning,
        CASE WHEN attr.event_id IS NULL THEN NULL ELSE (e.semantic_gate = 1) END
            AS semantic_is_match,
        attr.semantic_p_match     AS semantic_confidence_score,
        transcript.transcription  AS transcription,
        attr.semantic_method      AS semantic_method,
        attr.semantic_method_version AS semantic_method_version,
        attr.finality             AS attribution_finality,
        scores.soft_reward_candidate                AS soft_reward_candidate,
        CASE WHEN attr.event_id IS NULL THEN NULL ELSE e.au12_baseline_pre END
            AS attribution_au12_baseline_pre,
        scores.au12_lift_p90                        AS au12_lift_p90,
        scores.au12_lift_peak                       AS au12_lift_peak,
        scores.au12_peak_latency_ms                 AS au12_peak_latency_ms,
        scores.sync_peak_corr                       AS sync_peak_corr,
        scores.sync_peak_lag                        AS sync_peak_lag,
        outcome_link.lag_s                          AS outcome_link_lag_s
"""
)


_ENCOUNTERS_FROM_BLOCK: str = """
    FROM encounter_log e
    LEFT JOIN acoustic_latest acoustic
        ON acoustic.session_id = e.session_id
        AND acoustic.segment_id = e.segment_id
        AND acoustic.rn = 1
    LEFT JOIN transcript_latest transcript
        ON transcript.session_id = e.session_id
        AND transcript.segment_id = e.segment_id
        AND transcript.rn = 1
    LEFT JOIN attribution_event_latest attr
        ON attr.session_id = e.session_id
        AND attr.segment_id = e.segment_id
        AND attr.rn = 1
    LEFT JOIN attribution_scores_by_event scores
        ON scores.event_id = attr.event_id
    LEFT JOIN outcome_link_latest outcome_link
        ON outcome_link.event_id = attr.event_id
        AND outcome_link.rn = 1
"""


_SESSION_ENCOUNTERS_SQL: str = (
    _ENCOUNTERS_CTE_BLOCK
    + "SELECT\n"
    + _ENCOUNTERS_PROJECTION
    + _ENCOUNTERS_FROM_BLOCK
    + """
    WHERE e.session_id = :session_id
      AND (:before_utc IS NULL OR e.timestamp_utc < :before_utc)
    ORDER BY e.timestamp_utc DESC
    LIMIT :limit
"""
)


_LATEST_ENCOUNTER_SQL: str = (
    _ENCOUNTERS_CTE_BLOCK
    + "SELECT\n"
    + _ENCOUNTERS_PROJECTION
    + _ENCOUNTERS_FROM_BLOCK
    + """
    WHERE e.session_id = :session_id
    ORDER BY e.timestamp_utc DESC
    LIMIT 1
"""
)


# ---------------------------------------------------------------------------
# Experiments (§7B)
# ---------------------------------------------------------------------------


_EXPERIMENT_ARMS_SQL: str = """
WITH rollup AS (
    SELECT
        e.experiment_id,
        e.arm,
        COUNT(*) AS selection_count,
        AVG(e.gated_reward) AS recent_reward_mean,
        AVG(CAST(e.semantic_gate AS REAL)) AS recent_semantic_pass_rate
    FROM encounter_log e
    GROUP BY e.experiment_id, e.arm
)
SELECT
    ex.experiment_id,
    COALESCE(ex.label, ex.experiment_id)            AS label,
    ex.arm,
    COALESCE(ex.greeting_text, ex.arm)              AS greeting_text,
    ex.alpha_param,
    ex.beta_param,
    COALESCE(ex.enabled, 1)                         AS enabled,
    ex.end_dated_at,
    ex.updated_at,
    rollup.selection_count                          AS selection_count,
    rollup.recent_reward_mean                       AS recent_reward_mean,
    rollup.recent_semantic_pass_rate                AS recent_semantic_pass_rate
FROM experiments ex
LEFT JOIN rollup
    ON rollup.experiment_id = ex.experiment_id
   AND rollup.arm = ex.arm
WHERE ex.experiment_id = :experiment_id
ORDER BY ex.arm
"""


_ACTIVE_ARM_FOR_EXPERIMENT_SQL: str = """
SELECT active_arm AS arm, started_at AS timestamp_utc, bandit_decision_snapshot
FROM sessions
WHERE experiment_id = :experiment_id
  AND ended_at IS NULL
  AND active_arm IS NOT NULL
UNION ALL
SELECT e.arm, e.timestamp_utc, ae.bandit_decision_snapshot
FROM encounter_log e
LEFT JOIN attribution_event ae
    ON ae.session_id = e.session_id
   AND ae.segment_id = e.segment_id
WHERE e.experiment_id = :experiment_id
ORDER BY timestamp_utc DESC
LIMIT 1
"""


_EXPERIMENT_SUMMARIES_SQL: str = """
SELECT
    ex.experiment_id,
    COALESCE(ex.label, ex.experiment_id) AS label,
    COUNT(*) AS arm_count,
    MAX(ex.updated_at) AS last_updated_utc
FROM experiments ex
GROUP BY ex.experiment_id, COALESCE(ex.label, ex.experiment_id)
ORDER BY ex.experiment_id
"""


# ---------------------------------------------------------------------------
# Physiology (§4.E.2, §7C)
# ---------------------------------------------------------------------------


_LATEST_PHYSIO_SQL: str = """
WITH ranked AS (
    SELECT
        p.session_id,
        p.segment_id,
        p.subject_role,
        p.rmssd_ms,
        p.heart_rate_bpm,
        p.freshness_s,
        p.is_stale,
        p.provider,
        p.source_timestamp_utc,
        p.created_at,
        ROW_NUMBER() OVER (
            PARTITION BY p.subject_role
            ORDER BY p.created_at DESC
        ) AS rn
    FROM physiology_log p
    WHERE p.session_id = :session_id
)
SELECT
    session_id, segment_id, subject_role, rmssd_ms, heart_rate_bpm,
    freshness_s, is_stale, provider, source_timestamp_utc, created_at
FROM ranked
WHERE rn = 1
ORDER BY subject_role
"""


_LATEST_COMOD_SQL: str = """
SELECT
    c.session_id,
    c.window_end_utc,
    c.window_minutes,
    c.co_modulation_index,
    c.n_paired_observations,
    c.coverage_ratio,
    c.streamer_rmssd_mean,
    c.operator_rmssd_mean,
    c.created_at
FROM comodulation_log c
WHERE c.session_id = :session_id
ORDER BY c.window_end_utc DESC
LIMIT 1
"""


# ---------------------------------------------------------------------------
# Health heuristics
# ---------------------------------------------------------------------------


_SUBSYSTEM_PULSE_SQL: str = """
SELECT
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'ui_api_shell'
    )                                                  AS last_ui_api_shell_at,
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'capture_supervisor'
    )                                                  AS last_capture_supervisor_at,
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'module_c_orchestrator'
    )                                                  AS last_module_c_orchestrator_at,
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'gpu_ml_worker'
    )                                                  AS last_gpu_ml_worker_at,
    (
        SELECT c.state
        FROM capture_status c
        WHERE c.status_key = 'gpu_ml_worker'
    )                                                  AS gpu_ml_worker_state,
    (
        SELECT c.detail
        FROM capture_status c
        WHERE c.status_key = 'gpu_ml_worker'
    )                                                  AS gpu_ml_worker_detail,
    (
        SELECT c.operator_action_hint
        FROM capture_status c
        WHERE c.status_key = 'gpu_ml_worker'
    )                                                  AS gpu_ml_worker_hint,
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'analytics_state_worker'
    )                                                  AS last_analytics_state_worker_at,
    (
        SELECT MAX(h.last_heartbeat_utc)
        FROM process_heartbeat h
        WHERE h.process_name = 'cloud_sync_worker'
    )                                                  AS last_cloud_sync_worker_at,
    (
        SELECT c.state
        FROM capture_status c
        WHERE c.status_key = 'adb'
    )                                                  AS adb_state,
    (
        SELECT c.label
        FROM capture_status c
        WHERE c.status_key = 'adb'
    )                                                  AS adb_label,
    (
        SELECT c.detail
        FROM capture_status c
        WHERE c.status_key = 'adb'
    )                                                  AS adb_detail,
    (
        SELECT c.operator_action_hint
        FROM capture_status c
        WHERE c.status_key = 'adb'
    )                                                  AS adb_hint,
    (
        SELECT c.updated_at_utc
        FROM capture_status c
        WHERE c.status_key = 'adb'
    )                                                  AS last_adb_at,
    (
        SELECT c.state
        FROM capture_status c
        WHERE c.status_key = 'audio_capture'
    )                                                  AS audio_capture_state,
    (
        SELECT c.detail
        FROM capture_status c
        WHERE c.status_key = 'audio_capture'
    )                                                  AS audio_capture_detail,
    (
        SELECT c.operator_action_hint
        FROM capture_status c
        WHERE c.status_key = 'audio_capture'
    )                                                  AS audio_capture_hint,
    (
        SELECT c.updated_at_utc
        FROM capture_status c
        WHERE c.status_key = 'audio_capture'
    )                                                  AS last_audio_capture_at,
    (
        SELECT c.state
        FROM capture_status c
        WHERE c.status_key = 'video_capture'
    )                                                  AS video_capture_state,
    (
        SELECT c.detail
        FROM capture_status c
        WHERE c.status_key = 'video_capture'
    )                                                  AS video_capture_detail,
    (
        SELECT c.operator_action_hint
        FROM capture_status c
        WHERE c.status_key = 'video_capture'
    )                                                  AS video_capture_hint,
    (
        SELECT c.updated_at_utc
        FROM capture_status c
        WHERE c.status_key = 'video_capture'
    )                                                  AS last_video_capture_at,
    (
        SELECT MAX(live.updated_at_utc)
        FROM live_session_state live
        JOIN sessions s ON s.session_id = live.session_id
        WHERE s.ended_at IS NULL
    )                                                  AS last_live_visual_state_at,
    (
        SELECT live.status
        FROM live_session_state live
        JOIN sessions s ON s.session_id = live.session_id
        WHERE s.ended_at IS NULL
        ORDER BY live.updated_at_utc DESC
        LIMIT 1
    )                                                  AS live_visual_state_status,
    (
        SELECT MAX(e.timestamp_utc)
        FROM encounter_log e
        JOIN sessions s ON s.session_id = e.session_id
        WHERE s.ended_at IS NULL
    )                                                  AS last_live_encounter_at,
    (
        SELECT COUNT(*)
        FROM sessions s
        WHERE s.ended_at IS NULL
    )                                                  AS active_session_count
"""


_RECENT_STALE_PHYSIO_SQL: str = """
SELECT
    p.session_id,
    p.subject_role,
    p.created_at,
    p.freshness_s
FROM physiology_log p
WHERE p.is_stale = 1
  AND p.created_at >= datetime('now', '-1 hour')
  AND (:since_utc IS NULL OR p.created_at >= :since_utc)
ORDER BY p.created_at DESC
LIMIT :limit
"""


_RECENTLY_ENDED_SESSIONS_SQL: str = """
SELECT s.session_id, s.ended_at
FROM sessions s
WHERE s.ended_at IS NOT NULL
  AND s.ended_at >= datetime('now', '-1 hour')
  AND (:since_utc IS NULL OR s.ended_at >= :since_utc)
ORDER BY s.ended_at DESC
LIMIT :limit
"""


# ---------------------------------------------------------------------------
# Change markers
# ---------------------------------------------------------------------------

_MARKER_SESSIONS_SQL: str = (
    _LATEST_ENCOUNTER_PER_SESSION_CTE
    + """
SELECT
    COUNT(*) AS row_count,
    MAX(s.started_at) AS max_started_at,
    MAX(s.ended_at) AS max_ended_at,
    MAX(live.updated_at_utc) AS max_live_updated_at,
    MAX(le.timestamp_utc) AS max_last_segment_completed_at_utc,
    MAX(le.gated_reward) AS max_latest_reward,
    MIN(le.gated_reward) AS min_latest_reward,
    MAX(le.semantic_gate) AS max_latest_semantic_gate,
    MIN(le.semantic_gate) AS min_latest_semantic_gate,
    MAX(COALESCE(live.active_arm, le.arm)) AS max_active_arm,
    MAX(live.expected_greeting) AS max_expected_greeting,
    MAX(live.is_calibrating) AS max_is_calibrating,
    MIN(live.is_calibrating) AS min_is_calibrating,
    MAX(live.calibration_frames_accumulated) AS max_calibration_frames_accumulated,
    MIN(live.calibration_frames_accumulated) AS min_calibration_frames_accumulated,
    MAX(live.calibration_frames_required) AS max_calibration_frames_required,
    MIN(live.calibration_frames_required) AS min_calibration_frames_required,
    (
        SELECT COUNT(*)
        FROM sessions active
        WHERE active.ended_at IS NULL
    ) AS active_session_count
FROM sessions s
LEFT JOIN latest_enc le ON le.session_id = s.session_id AND le.rn = 1
LEFT JOIN live_session_state live ON live.session_id = s.session_id
"""
)

_MARKER_LIVE_SESSION_SQL: str = """
SELECT
    COUNT(*) AS active_session_count,
    MAX(s.started_at) AS max_active_started_at,
    MAX(live.updated_at_utc) AS max_live_updated_at,
    MAX(e.timestamp_utc) AS max_encounter_at
FROM sessions s
LEFT JOIN live_session_state live ON live.session_id = s.session_id
LEFT JOIN encounter_log e ON e.session_id = s.session_id
WHERE s.ended_at IS NULL
"""

_MARKER_ENCOUNTERS_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM encounter_log e
        WHERE (:session_id IS NULL OR e.session_id = :session_id)
    ) AS row_count,
    (
        SELECT MAX(e.timestamp_utc)
        FROM encounter_log e
        WHERE (:session_id IS NULL OR e.session_id = :session_id)
    ) AS max_timestamp_utc,
    (
        SELECT MAX(e.created_at)
        FROM encounter_log e
        WHERE (:session_id IS NULL OR e.session_id = :session_id)
    ) AS max_created_at,
    (
        SELECT MAX(e.id)
        FROM encounter_log e
        WHERE (:session_id IS NULL OR e.session_id = :session_id)
    ) AS max_id,
    (
        SELECT COUNT(*)
        FROM attribution_event a
        WHERE (:session_id IS NULL OR a.session_id = :session_id)
    ) AS attribution_event_count,
    (
        SELECT COUNT(*)
        FROM attribution_event a
        WHERE (:session_id IS NULL OR a.session_id = :session_id)
          AND a.finality = 'online_provisional'
    ) AS online_provisional_attribution_count,
    (
        SELECT COUNT(*)
        FROM attribution_event a
        WHERE (:session_id IS NULL OR a.session_id = :session_id)
          AND a.finality = 'offline_final'
    ) AS offline_final_attribution_count,
    (
        SELECT MAX(a.created_at)
        FROM attribution_event a
        WHERE (:session_id IS NULL OR a.session_id = :session_id)
    ) AS max_attribution_created_at
"""

_MARKER_EXPERIMENTS_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM experiments
    ) AS row_count,
    (
        SELECT MAX(updated_at)
        FROM experiments
    ) AS max_updated_at,
    (
        SELECT MAX(id)
        FROM experiments
    ) AS max_id,
    (
        SELECT SUM(alpha_param)
        FROM experiments
    ) AS posterior_alpha_sum,
    (
        SELECT SUM(beta_param)
        FROM experiments
    ) AS posterior_beta_sum,
    (
        SELECT COUNT(*)
        FROM sessions
        WHERE bandit_decision_snapshot IS NOT NULL
    ) AS session_decision_snapshot_count,
    (
        SELECT MAX(started_at)
        FROM sessions
        WHERE bandit_decision_snapshot IS NOT NULL
    ) AS max_session_decision_at,
    (
        SELECT COUNT(*)
        FROM attribution_event
    ) AS attribution_event_count,
    (
        SELECT MAX(created_at)
        FROM attribution_event
    ) AS max_attribution_created_at
"""

_MARKER_EXPERIMENT_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM experiments
        WHERE experiment_id = :experiment_id
    ) AS row_count,
    (
        SELECT MAX(updated_at)
        FROM experiments
        WHERE experiment_id = :experiment_id
    ) AS max_updated_at,
    (
        SELECT MAX(id)
        FROM experiments
        WHERE experiment_id = :experiment_id
    ) AS max_id,
    (
        SELECT SUM(alpha_param)
        FROM experiments
        WHERE experiment_id = :experiment_id
    ) AS posterior_alpha_sum,
    (
        SELECT SUM(beta_param)
        FROM experiments
        WHERE experiment_id = :experiment_id
    ) AS posterior_beta_sum,
    (
        SELECT COUNT(*)
        FROM sessions
        WHERE experiment_id = :experiment_id
          AND bandit_decision_snapshot IS NOT NULL
    ) AS session_decision_snapshot_count,
    (
        SELECT MAX(started_at)
        FROM sessions
        WHERE experiment_id = :experiment_id
          AND bandit_decision_snapshot IS NOT NULL
    ) AS max_session_decision_at,
    (
        SELECT COUNT(*)
        FROM attribution_event ae
        JOIN sessions s ON s.session_id = ae.session_id
        WHERE s.experiment_id = :experiment_id
    ) AS attribution_event_count,
    (
        SELECT MAX(ae.created_at)
        FROM attribution_event ae
        JOIN sessions s ON s.session_id = ae.session_id
        WHERE s.experiment_id = :experiment_id
    ) AS max_attribution_created_at
"""

_MARKER_PHYSIOLOGY_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM physiology_log p
        WHERE (:session_id IS NULL OR p.session_id = :session_id)
    ) AS physiology_count,
    (
        SELECT MAX(p.created_at)
        FROM physiology_log p
        WHERE (:session_id IS NULL OR p.session_id = :session_id)
    ) AS max_physiology_created_at,
    (
        SELECT MAX(p.id)
        FROM physiology_log p
        WHERE (:session_id IS NULL OR p.session_id = :session_id)
    ) AS max_physiology_id,
    (
        SELECT COUNT(*)
        FROM comodulation_log c
        WHERE (:session_id IS NULL OR c.session_id = :session_id)
    ) AS comodulation_count,
    (
        SELECT MAX(c.created_at)
        FROM comodulation_log c
        WHERE (:session_id IS NULL OR c.session_id = :session_id)
    ) AS max_comodulation_created_at,
    (
        SELECT MAX(c.id)
        FROM comodulation_log c
        WHERE (:session_id IS NULL OR c.session_id = :session_id)
    ) AS max_comodulation_id
"""

_MARKER_HEALTH_SQL: str = """
SELECT
    (
        SELECT MAX(last_heartbeat_utc)
        FROM process_heartbeat
    ) AS max_process_heartbeat_utc,
    (
        SELECT COUNT(*)
        FROM process_heartbeat
    ) AS process_count,
    (
        SELECT MAX(updated_at_utc)
        FROM capture_status
    ) AS max_capture_status_updated_at,
    (
        SELECT COUNT(*)
        FROM capture_status
    ) AS capture_status_count,
    (
        SELECT MAX(updated_at_utc)
        FROM live_session_state
    ) AS max_live_updated_at,
    (
        SELECT COUNT(*)
        FROM sessions
        WHERE ended_at IS NULL
    ) AS active_session_count,
    (
        SELECT MAX(timestamp_utc)
        FROM encounter_log
    ) AS max_encounter_at
"""

_MARKER_ALERTS_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM physiology_log p
        WHERE p.is_stale = 1 AND p.created_at >= datetime('now', '-1 hour')
    ) AS stale_physiology_count,
    (
        SELECT MAX(p.created_at)
        FROM physiology_log p
        WHERE p.is_stale = 1 AND p.created_at >= datetime('now', '-1 hour')
    ) AS max_stale_physiology_created_at,
    (
        SELECT COUNT(*)
        FROM sessions s
        WHERE s.ended_at IS NOT NULL AND s.ended_at >= datetime('now', '-1 hour')
    ) AS ended_session_count,
    (
        SELECT MAX(s.ended_at)
        FROM sessions s
        WHERE s.ended_at IS NOT NULL AND s.ended_at >= datetime('now', '-1 hour')
    ) AS max_ended_at,
    (
        SELECT MAX(updated_at_utc)
        FROM capture_status
        WHERE state IN ('degraded', 'recovering', 'error')
    ) AS max_alerting_capture_updated_at
"""

_MARKER_OVERVIEW_SQL: str = """
SELECT
    (
        SELECT COUNT(*)
        FROM sessions
        WHERE ended_at IS NULL
    ) AS active_session_count,
    (
        SELECT MAX(started_at)
        FROM sessions
        WHERE ended_at IS NULL
    ) AS max_active_started_at,
    (
        SELECT MAX(ended_at)
        FROM sessions
    ) AS max_ended_at,
    (
        SELECT MAX(timestamp_utc)
        FROM encounter_log
    ) AS max_encounter_at,
    (
        SELECT MAX(created_at)
        FROM physiology_log
    ) AS max_physiology_created_at,
    (
        SELECT MAX(created_at)
        FROM comodulation_log
    ) AS max_comodulation_created_at,
    (
        SELECT MAX(updated_at)
        FROM experiments
    ) AS max_experiment_updated_at,
    (
        SELECT SUM(alpha_param)
        FROM experiments
    ) AS experiment_posterior_alpha_sum,
    (
        SELECT SUM(beta_param)
        FROM experiments
    ) AS experiment_posterior_beta_sum,
    (
        SELECT COUNT(*)
        FROM sessions
        WHERE bandit_decision_snapshot IS NOT NULL
    ) AS session_decision_snapshot_count,
    (
        SELECT MAX(started_at)
        FROM sessions
        WHERE bandit_decision_snapshot IS NOT NULL
    ) AS max_session_decision_at,
    (
        SELECT MAX(last_heartbeat_utc)
        FROM process_heartbeat
    ) AS max_process_heartbeat_utc,
    (
        SELECT MAX(updated_at_utc)
        FROM capture_status
    ) AS max_capture_status_updated_at,
    (
        SELECT MAX(updated_at_utc)
        FROM live_session_state
    ) AS max_live_updated_at,
    (
        SELECT COUNT(*)
        FROM attribution_event
    ) AS attribution_event_count,
    (
        SELECT COUNT(*)
        FROM attribution_event
        WHERE finality = 'online_provisional'
    ) AS online_provisional_attribution_count,
    (
        SELECT COUNT(*)
        FROM attribution_event
        WHERE finality = 'offline_final'
    ) AS offline_final_attribution_count,
    (
        SELECT MAX(created_at)
        FROM attribution_event
    ) AS max_attribution_created_at
"""


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _row_to_dict(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
    if cursor.description is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(zip(columns, row, strict=True))


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    row = _row_to_dict(cursor)
    return {} if row is None else {key: _marker_value(value) for key, value in row.items()}


def _marker_value(value: object) -> OperatorMarkerValue:
    if isinstance(value, str | int | float) or value is None:
        return value
    return str(value)


def _isoformat_utc(value: datetime | None) -> str | None:
    """Serialize a datetime for SQLite parameter binding, dropping tz suffix.

    SQLite stores ``CURRENT_TIMESTAMP`` as ``'YYYY-MM-DD HH:MM:SS'`` (no
    ``T`` separator, no offset), and our writer emits ISO-8601 with
    ``+00:00``. The ``datetime('now', '-1 hour')`` produced inside SQLite
    is space-separated. Comparing ``ts >= :since_utc`` therefore works
    correctly only when both sides are space-separated UTC strings.
    """
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Public fetchers — same names + return shapes as the Postgres repo.
# ---------------------------------------------------------------------------


def fetch_sessions_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_SESSIONS_SQL)
    return _marker(cursor)


def fetch_live_session_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_LIVE_SESSION_SQL)
    return _marker(cursor)


def fetch_encounters_marker(
    cursor: sqlite3.Cursor, *, session_id: UUID | None = None
) -> OperatorChangeMarker:
    cursor.execute(_MARKER_ENCOUNTERS_SQL, {"session_id": str(session_id) if session_id else None})
    return _marker(cursor)


def fetch_experiments_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_EXPERIMENTS_SQL)
    return _marker(cursor)


def fetch_experiment_marker(cursor: sqlite3.Cursor, *, experiment_id: str) -> OperatorChangeMarker:
    cursor.execute(_MARKER_EXPERIMENT_SQL, {"experiment_id": experiment_id})
    return _marker(cursor)


def fetch_physiology_marker(
    cursor: sqlite3.Cursor, *, session_id: UUID | None = None
) -> OperatorChangeMarker:
    cursor.execute(_MARKER_PHYSIOLOGY_SQL, {"session_id": str(session_id) if session_id else None})
    return _marker(cursor)


def fetch_health_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_HEALTH_SQL)
    return _marker(cursor)


def fetch_alerts_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_ALERTS_SQL)
    return _marker(cursor)


def fetch_overview_marker(cursor: sqlite3.Cursor) -> OperatorChangeMarker:
    cursor.execute(_MARKER_OVERVIEW_SQL)
    return _marker(cursor)


def fetch_recent_sessions(cursor: sqlite3.Cursor, *, limit: int) -> list[dict[str, Any]]:
    cursor.execute(_LIST_RECENT_SESSIONS_SQL, {"limit": limit})
    return _rows_to_dicts(cursor)


def fetch_session_by_id(cursor: sqlite3.Cursor, session_id: UUID) -> dict[str, Any] | None:
    cursor.execute(_GET_SESSION_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_active_session(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
    cursor.execute(_GET_ACTIVE_SESSION_SQL)
    return _row_to_dict(cursor)


def fetch_session_encounters(
    cursor: sqlite3.Cursor,
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
            "before_utc": _isoformat_utc(before_utc),
        },
    )
    return _rows_to_dicts(cursor)


def fetch_latest_encounter(cursor: sqlite3.Cursor, session_id: UUID) -> dict[str, Any] | None:
    cursor.execute(_LATEST_ENCOUNTER_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_experiment_summaries(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    cursor.execute(_EXPERIMENT_SUMMARIES_SQL)
    return _rows_to_dicts(cursor)


def fetch_experiment_arms(cursor: sqlite3.Cursor, experiment_id: str) -> list[dict[str, Any]]:
    cursor.execute(_EXPERIMENT_ARMS_SQL, {"experiment_id": experiment_id})
    return _rows_to_dicts(cursor)


def fetch_active_arm_for_experiment(
    cursor: sqlite3.Cursor, experiment_id: str
) -> dict[str, Any] | None:
    cursor.execute(_ACTIVE_ARM_FOR_EXPERIMENT_SQL, {"experiment_id": experiment_id})
    return _row_to_dict(cursor)


def fetch_latest_physiology_rows(cursor: sqlite3.Cursor, session_id: UUID) -> list[dict[str, Any]]:
    cursor.execute(_LATEST_PHYSIO_SQL, {"session_id": str(session_id)})
    return _rows_to_dicts(cursor)


def fetch_latest_comodulation_row(
    cursor: sqlite3.Cursor, session_id: UUID
) -> dict[str, Any] | None:
    cursor.execute(_LATEST_COMOD_SQL, {"session_id": str(session_id)})
    return _row_to_dict(cursor)


def fetch_subsystem_pulse(cursor: sqlite3.Cursor) -> dict[str, Any]:
    cursor.execute(_SUBSYSTEM_PULSE_SQL)
    row = _row_to_dict(cursor)
    return row if row is not None else {}


def fetch_recent_stale_physiology(
    cursor: sqlite3.Cursor, *, since_utc: datetime | None, limit: int
) -> list[dict[str, Any]]:
    cursor.execute(
        _RECENT_STALE_PHYSIO_SQL,
        {"since_utc": _isoformat_utc(since_utc), "limit": limit},
    )
    return _rows_to_dicts(cursor)


def fetch_recently_ended_sessions(
    cursor: sqlite3.Cursor, *, since_utc: datetime | None, limit: int
) -> list[dict[str, Any]]:
    cursor.execute(
        _RECENTLY_ENDED_SESSIONS_SQL,
        {"since_utc": _isoformat_utc(since_utc), "limit": limit},
    )
    return _rows_to_dicts(cursor)


__all__ = [
    "OperatorChangeMarker",
    "OperatorMarkerValue",
    "fetch_active_arm_for_experiment",
    "fetch_active_session",
    "fetch_alerts_marker",
    "fetch_encounters_marker",
    "fetch_experiment_arms",
    "fetch_experiment_marker",
    "fetch_experiments_marker",
    "fetch_health_marker",
    "fetch_latest_comodulation_row",
    "fetch_latest_encounter",
    "fetch_latest_physiology_rows",
    "fetch_live_session_marker",
    "fetch_overview_marker",
    "fetch_physiology_marker",
    "fetch_recent_sessions",
    "fetch_recent_stale_physiology",
    "fetch_recently_ended_sessions",
    "fetch_session_by_id",
    "fetch_session_encounters",
    "fetch_sessions_marker",
    "fetch_subsystem_pulse",
]
