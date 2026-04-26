"""
OperatorReadService — composes Phase-1 DTOs from DB row dicts.

This is the only place in the API Server that performs cross-table
assembly for the Operator Console surfaces. Route handlers call into
this class; the repo layer (`services.api.repos.operator_queries`)
stays pure SQL. Every public method returns a validated Pydantic DTO
from `packages.schemas.operator_console` (Phase 1).

Design constraints:
  - No raw SQL here; the repo layer owns statements (§2 step 7).
  - No HTTP concerns; FastAPI plumbing stays in the route module.
  - No formatting for display; operator-facing text lives in
    `services.operator_console.formatters` (Phase 3).
  - Null is a first-class outcome for §7C co-modulation — always surface
    `null_reason` when the index is None rather than flattening to 0.0.
  - §12 degraded-but-recovering states must carry `recovery_mode` and
    `operator_action_hint` so the operator sees in-progress self-heal
    paths distinctly from hard errors.

Spec references:
  §4.C     — Orchestrator `_active_arm`, `_expected_greeting`,
             authoritative `_stimulus_time`.
  §4.C.4   — Physiological State Buffer freshness semantics.
  §4.E.1   — Operator-facing execution details (replaces the retired
             Streamlit dashboard per SPEC-AMEND-008).
  §4.E.2   — Physiology persistence schema (rmssd_ms, hr, provider).
  §7B      — Thompson Sampling reward = p90_intensity × semantic_gate.
  §7C      — Rolling Co-Modulation Index; null-valid.
  §12      — Error-handling matrix (degraded/recovering vs error).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from packages.schemas.operator_console import (
    AcousticObservationalMetrics,
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    ObservationalAcousticSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
)
from services.api.db.connection import get_connection, put_connection
from services.api.repos import operator_queries as q

# Subsystem-freshness thresholds drive §12 degraded/recovering/error
# classification. "Degraded" = a recent signal but older than expected;
# "Recovering" is reserved for the narrow recent window right after a
# subsystem has resumed writing. Outside any threshold → UNKNOWN/ERROR.
_RECOVERING_WINDOW_S: float = 30.0
_DEGRADED_WINDOW_S: float = 120.0
_ERROR_WINDOW_S: float = 600.0
_ALERT_LOOKBACK_S: float = 3600.0  # 1h window for alert synthesis


class OperatorReadService:
    """Read-side aggregation for `/api/v1/operator/*`.

    The service takes injectable connection hooks and a clock so tests
    can drive both without monkey-patching the module-level pool.
    """

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

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def get_overview(self) -> OverviewSnapshot:
        """§4.E.1 — composed Overview card set."""
        now = self._clock()
        with self._cursor() as cur:
            active_row = q.fetch_active_session(cur)
            active_session = (
                self._build_session_summary(active_row) if active_row is not None else None
            )

            latest_encounter: LatestEncounterSummary | None = None
            experiment_summary: ExperimentSummary | None = None
            physiology: SessionPhysiologySnapshot | None = None

            if active_session is not None:
                enc_row = q.fetch_latest_encounter(cur, active_session.session_id)
                latest_encounter = self._build_latest_encounter_summary(enc_row)

                exp_id = (enc_row or {}).get("experiment_id")
                if isinstance(exp_id, str) and exp_id:
                    arm_rows = q.fetch_experiment_arms(cur, exp_id)
                    active_arm_row = q.fetch_active_arm_for_experiment(cur, exp_id)
                    experiment_summary = self._build_experiment_summary(
                        exp_id, arm_rows, active_arm_row, enc_row
                    )

                physiology = self._build_session_physiology(cur, active_session.session_id, now)

            health = self._build_health_snapshot(cur, now)
            alerts = self._list_alerts_internal(cur, limit=20, since_utc=None)

        return OverviewSnapshot(
            generated_at_utc=now,
            active_session=active_session,
            latest_encounter=latest_encounter,
            experiment_summary=experiment_summary,
            physiology=physiology,
            health=health,
            alerts=alerts,
        )

    def list_sessions(self, *, limit: int = 50) -> list[SessionSummary]:
        with self._cursor() as cur:
            rows = q.fetch_recent_sessions(cur, limit=limit)
        return [self._build_session_summary(row) for row in rows]

    def get_session(self, session_id: UUID) -> SessionSummary | None:
        with self._cursor() as cur:
            row = q.fetch_session_by_id(cur, session_id)
        if row is None:
            return None
        return self._build_session_summary(row)

    def list_encounters(
        self,
        session_id: UUID,
        *,
        limit: int = 100,
        before_utc: datetime | None = None,
    ) -> list[EncounterSummary]:
        with self._cursor() as cur:
            rows = q.fetch_session_encounters(cur, session_id, limit=limit, before_utc=before_utc)
        return [self._build_encounter_summary(row) for row in rows]

    def get_experiment_detail(self, experiment_id: str) -> ExperimentDetail | None:
        with self._cursor() as cur:
            arm_rows = q.fetch_experiment_arms(cur, experiment_id)
            if not arm_rows:
                return None
            active_arm_row = q.fetch_active_arm_for_experiment(cur, experiment_id)

        arms = [self._build_arm_summary(row) for row in arm_rows]
        active_arm_id = (active_arm_row or {}).get("arm") if active_arm_row is not None else None
        last_updated = _latest_datetime(row.get("updated_at") for row in arm_rows)
        return ExperimentDetail(
            experiment_id=experiment_id,
            label=None,
            active_arm_id=active_arm_id if isinstance(active_arm_id, str) else None,
            arms=arms,
            last_update_summary=None,
            last_updated_utc=_ensure_utc(last_updated),
        )

    def get_session_physiology(self, session_id: UUID) -> SessionPhysiologySnapshot | None:
        now = self._clock()
        with self._cursor() as cur:
            session = q.fetch_session_by_id(cur, session_id)
            if session is None:
                return None
            return self._build_session_physiology(cur, session_id, now)

    def get_health(self) -> HealthSnapshot:
        now = self._clock()
        with self._cursor() as cur:
            return self._build_health_snapshot(cur, now)

    def list_alerts(
        self, *, limit: int = 50, since_utc: datetime | None = None
    ) -> list[AlertEvent]:
        with self._cursor() as cur:
            return self._list_alerts_internal(cur, limit=limit, since_utc=since_utc)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    class _CursorContext:
        def __init__(
            self,
            get_conn: Callable[[], Any],
            put_conn: Callable[[Any], None],
        ) -> None:
            self._get_conn = get_conn
            self._put_conn = put_conn
            self._conn: Any = None
            self._cur: Any = None

        def __enter__(self) -> Any:
            self._conn = self._get_conn()
            self._cur = self._conn.cursor()
            # psycopg2 cursors are context managers themselves; use the
            # raw cursor so callers can stay cursor-only.
            return self._cur.__enter__()

        def __exit__(self, *exc: Any) -> None:
            try:
                self._cur.__exit__(*exc)
            finally:
                if self._conn is not None:
                    self._put_conn(self._conn)

    def _cursor(self) -> _CursorContext:
        return self._CursorContext(self._get_conn, self._put_conn)

    # ------------------------------------------------------------------
    # Builders — DB row dict → Phase-1 DTO
    # ------------------------------------------------------------------

    def _build_session_summary(self, row: dict[str, Any]) -> SessionSummary:
        """§4.E.1 — lightweight session card."""
        status = "active" if row.get("ended_at") is None else "ended"
        experiment_id = row.get("experiment_id")
        active_arm = row.get("active_arm")
        return SessionSummary(
            session_id=_as_uuid(row["session_id"]),
            status=status,
            started_at_utc=_ensure_utc_strict(row["started_at"]),
            ended_at_utc=_ensure_utc(row.get("ended_at")),
            duration_s=_as_float(row.get("duration_s")),
            experiment_id=experiment_id if isinstance(experiment_id, str) else None,
            active_arm=active_arm if isinstance(active_arm, str) else None,
            expected_greeting=None,  # orchestrator-owned; not persisted.
            last_segment_completed_at_utc=_ensure_utc(row.get("last_segment_completed_at_utc")),
            latest_reward=_as_float(row.get("latest_reward")),
            latest_semantic_gate=_as_int(row.get("latest_semantic_gate")),
        )

    def _build_encounter_summary(self, row: dict[str, Any]) -> EncounterSummary:
        """§7B — full reward-explanation row. Notes flag physiology staleness."""
        state = _encounter_state_for(row)
        notes: list[str] = []
        if row.get("semantic_gate") == 0:
            notes.append("gate-closed: reward zero per §7B")
        if (row.get("n_frames") or 0) == 0:
            notes.append("no valid frames in measurement window")
        acoustic = self._build_acoustic_observational_metrics(row)
        observational_acoustic = self._build_observational_acoustic_summary(row)
        return EncounterSummary(
            encounter_id=str(row["id"]),
            session_id=_as_uuid(row["session_id"]),
            segment_timestamp_utc=_ensure_utc_strict(row["timestamp_utc"]),
            state=state,
            active_arm=_as_str(row.get("arm")),
            expected_greeting=None,
            stimulus_time_utc=_stimulus_epoch_to_utc(row.get("stimulus_time")),
            semantic_gate=_as_int(row.get("semantic_gate")),
            semantic_confidence=None,
            p90_intensity=_as_float(row.get("p90_intensity")),
            gated_reward=_as_float(row.get("gated_reward")),
            n_frames_in_window=_as_int(row.get("n_frames")),
            baseline_b_neutral=_as_float(row.get("baseline_neutral")),
            acoustic=acoustic,
            observational_acoustic=observational_acoustic,
            physiology_attached=False,
            physiology_stale=None,
            notes=notes,
        )

    def _build_latest_encounter_summary(
        self, row: dict[str, Any] | None
    ) -> LatestEncounterSummary | None:
        if row is None:
            return None
        state = _encounter_state_for(row)
        acoustic = self._build_acoustic_observational_metrics(row)
        observational_acoustic = self._build_observational_acoustic_summary(row)
        return LatestEncounterSummary(
            encounter_id=str(row["id"]),
            session_id=_as_uuid(row["session_id"]),
            segment_timestamp_utc=_ensure_utc_strict(row["timestamp_utc"]),
            state=state,
            active_arm=_as_str(row.get("arm")),
            expected_greeting=None,
            stimulus_time_utc=_stimulus_epoch_to_utc(row.get("stimulus_time")),
            semantic_gate=_as_int(row.get("semantic_gate")),
            p90_intensity=_as_float(row.get("p90_intensity")),
            gated_reward=_as_float(row.get("gated_reward")),
            n_frames_in_window=_as_int(row.get("n_frames")),
            acoustic=acoustic,
            observational_acoustic=observational_acoustic,
        )

    def _build_acoustic_observational_metrics(
        self, row: dict[str, Any]
    ) -> AcousticObservationalMetrics | None:
        acoustic_fields = (
            "pitch_f0",
            "jitter",
            "shimmer",
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
        has_metrics_row = row.get("metrics_row_id") is not None
        has_inline_acoustic_values = any(row.get(field) is not None for field in acoustic_fields)
        if not has_metrics_row and not has_inline_acoustic_values:
            return None

        voiced_coverage_measure = _as_float(row.get("voiced_coverage_measure_s"))
        voiced_coverage_baseline = _as_float(row.get("voiced_coverage_baseline_s"))
        return AcousticObservationalMetrics(
            pitch_f0=_as_float(row.get("pitch_f0")),
            jitter=_as_float(row.get("jitter")),
            shimmer=_as_float(row.get("shimmer")),
            f0_valid_measure=_as_bool(row.get("f0_valid_measure")),
            f0_valid_baseline=_as_bool(row.get("f0_valid_baseline")),
            perturbation_valid_measure=_as_bool(row.get("perturbation_valid_measure")),
            perturbation_valid_baseline=_as_bool(row.get("perturbation_valid_baseline")),
            voiced_coverage_measure_s=(
                voiced_coverage_measure if voiced_coverage_measure is not None else 0.0
            ),
            voiced_coverage_baseline_s=(
                voiced_coverage_baseline if voiced_coverage_baseline is not None else 0.0
            ),
            f0_mean_measure_hz=_as_float(row.get("f0_mean_measure_hz")),
            f0_mean_baseline_hz=_as_float(row.get("f0_mean_baseline_hz")),
            f0_delta_semitones=_as_float(row.get("f0_delta_semitones")),
            jitter_mean_measure=_as_float(row.get("jitter_mean_measure")),
            jitter_mean_baseline=_as_float(row.get("jitter_mean_baseline")),
            jitter_delta=_as_float(row.get("jitter_delta")),
            shimmer_mean_measure=_as_float(row.get("shimmer_mean_measure")),
            shimmer_mean_baseline=_as_float(row.get("shimmer_mean_baseline")),
            shimmer_delta=_as_float(row.get("shimmer_delta")),
        )

    def _build_observational_acoustic_summary(
        self, row: dict[str, Any]
    ) -> ObservationalAcousticSummary | None:
        """Hydrate canonical §7D acoustic analytics, preserving SQL NULLs."""
        f0_valid_measure_raw = row.get("f0_valid_measure")
        f0_valid_baseline_raw = row.get("f0_valid_baseline")
        perturbation_valid_measure_raw = row.get("perturbation_valid_measure")
        perturbation_valid_baseline_raw = row.get("perturbation_valid_baseline")

        f0_valid_measure = None if f0_valid_measure_raw is None else bool(f0_valid_measure_raw)
        f0_valid_baseline = None if f0_valid_baseline_raw is None else bool(f0_valid_baseline_raw)
        perturbation_valid_measure = (
            None if perturbation_valid_measure_raw is None else bool(perturbation_valid_measure_raw)
        )
        perturbation_valid_baseline = (
            None
            if perturbation_valid_baseline_raw is None
            else bool(perturbation_valid_baseline_raw)
        )
        voiced_coverage_measure_s = _as_float(row.get("voiced_coverage_measure_s"))
        voiced_coverage_baseline_s = _as_float(row.get("voiced_coverage_baseline_s"))
        f0_mean_measure_hz = _as_float(row.get("f0_mean_measure_hz"))
        f0_mean_baseline_hz = _as_float(row.get("f0_mean_baseline_hz"))
        f0_delta_semitones = _as_float(row.get("f0_delta_semitones"))
        jitter_mean_measure = _as_float(row.get("jitter_mean_measure"))
        jitter_mean_baseline = _as_float(row.get("jitter_mean_baseline"))
        jitter_delta = _as_float(row.get("jitter_delta"))
        shimmer_mean_measure = _as_float(row.get("shimmer_mean_measure"))
        shimmer_mean_baseline = _as_float(row.get("shimmer_mean_baseline"))
        shimmer_delta = _as_float(row.get("shimmer_delta"))

        acoustic_values: tuple[bool | float | None, ...] = (
            f0_valid_measure,
            f0_valid_baseline,
            perturbation_valid_measure,
            perturbation_valid_baseline,
            voiced_coverage_measure_s,
            voiced_coverage_baseline_s,
            f0_mean_measure_hz,
            f0_mean_baseline_hz,
            f0_delta_semitones,
            jitter_mean_measure,
            jitter_mean_baseline,
            jitter_delta,
            shimmer_mean_measure,
            shimmer_mean_baseline,
            shimmer_delta,
        )
        if all(value is None for value in acoustic_values):
            return None

        return ObservationalAcousticSummary(
            f0_valid_measure=f0_valid_measure,
            f0_valid_baseline=f0_valid_baseline,
            perturbation_valid_measure=perturbation_valid_measure,
            perturbation_valid_baseline=perturbation_valid_baseline,
            voiced_coverage_measure_s=voiced_coverage_measure_s,
            voiced_coverage_baseline_s=voiced_coverage_baseline_s,
            f0_mean_measure_hz=f0_mean_measure_hz,
            f0_mean_baseline_hz=f0_mean_baseline_hz,
            f0_delta_semitones=f0_delta_semitones,
            jitter_mean_measure=jitter_mean_measure,
            jitter_mean_baseline=jitter_mean_baseline,
            jitter_delta=jitter_delta,
            shimmer_mean_measure=shimmer_mean_measure,
            shimmer_mean_baseline=shimmer_mean_baseline,
            shimmer_delta=shimmer_delta,
        )

    def _build_arm_summary(self, row: dict[str, Any]) -> ArmSummary:
        """§7B — arm posterior + historical rollup."""
        alpha = _as_float(row.get("alpha_param")) or 1.0
        beta = _as_float(row.get("beta_param")) or 1.0
        # Beta-distribution variance = αβ / ((α+β)² (α+β+1))
        total = alpha + beta
        variance = (alpha * beta) / (total * total * (total + 1.0)) if total > 0 else None
        return ArmSummary(
            arm_id=str(row["arm"]),
            greeting_text=str(row["arm"]),  # canonical label until greeting map exposed.
            posterior_alpha=alpha,
            posterior_beta=beta,
            evaluation_variance=variance,
            selection_count=_as_int(row.get("selection_count")) or 0,
            recent_reward_mean=_as_float(row.get("recent_reward_mean")),
            recent_semantic_pass_rate=_as_float(row.get("recent_semantic_pass_rate")),
        )

    def _build_experiment_summary(
        self,
        experiment_id: str,
        arm_rows: list[dict[str, Any]],
        active_arm_row: dict[str, Any] | None,
        latest_encounter_row: dict[str, Any] | None,
    ) -> ExperimentSummary:
        active_arm_id = (active_arm_row or {}).get("arm") if active_arm_row is not None else None
        last_updated = _latest_datetime(row.get("updated_at") for row in arm_rows)
        latest_reward: float | None = None
        if latest_encounter_row is not None:
            latest_reward = _as_float(latest_encounter_row.get("gated_reward"))
        return ExperimentSummary(
            experiment_id=experiment_id,
            label=None,
            active_arm_id=active_arm_id if isinstance(active_arm_id, str) else None,
            arm_count=len(arm_rows),
            last_updated_utc=_ensure_utc(last_updated),
            latest_reward=latest_reward,
        )

    def _build_physiology_snapshot(self, row: dict[str, Any]) -> PhysiologyCurrentSnapshot:
        """§4.E.2 — per-subject_role physiology row."""
        role = row.get("subject_role")
        if role not in ("streamer", "operator"):
            raise ValueError(f"unexpected subject_role: {role!r}")
        return PhysiologyCurrentSnapshot(
            subject_role=role,
            rmssd_ms=_as_float(row.get("rmssd_ms")),
            heart_rate_bpm=_as_int(row.get("heart_rate_bpm")),
            provider=_as_str(row.get("provider")),
            source_timestamp_utc=_ensure_utc(row.get("source_timestamp_utc")),
            freshness_s=_as_float(row.get("freshness_s")),
            is_stale=bool(row.get("is_stale")) if row.get("is_stale") is not None else None,
        )

    def _build_co_modulation_summary(
        self, session_id: UUID, row: dict[str, Any] | None
    ) -> CoModulationSummary:
        """§7C — null-valid co-modulation readback.

        Per §7C, `co_modulation_index is None` when insufficient aligned
        non-stale pairs are available for the rolling window. We must
        surface a human-readable `null_reason` so the UI renders this
        legitimately rather than as an error.
        """
        if row is None:
            return CoModulationSummary(
                session_id=session_id,
                co_modulation_index=None,
                n_paired_observations=0,
                coverage_ratio=0.0,
                null_reason="no co-modulation window computed yet",
            )
        index = _as_float(row.get("co_modulation_index"))
        n = _as_int(row.get("n_paired_observations")) or 0
        coverage = _as_float(row.get("coverage_ratio")) or 0.0
        null_reason: str | None = None
        if index is None:
            if n == 0:
                null_reason = "insufficient aligned non-stale pairs"
            else:
                null_reason = "rolling window produced an undefined correlation"
        window_end = _ensure_utc(row.get("window_end_utc"))
        window_minutes = _as_int(row.get("window_minutes"))
        window_start: datetime | None = None
        if window_end is not None and window_minutes is not None:
            window_start = window_end - timedelta(minutes=window_minutes)
        return CoModulationSummary(
            session_id=session_id,
            co_modulation_index=index,
            n_paired_observations=n,
            coverage_ratio=max(0.0, min(1.0, coverage)),
            streamer_rmssd_mean=_as_float(row.get("streamer_rmssd_mean")),
            operator_rmssd_mean=_as_float(row.get("operator_rmssd_mean")),
            null_reason=null_reason,
            window_start_utc=window_start,
            window_end_utc=window_end,
        )

    def _build_session_physiology(
        self, cur: Any, session_id: UUID, now: datetime
    ) -> SessionPhysiologySnapshot:
        rows = q.fetch_latest_physiology_rows(cur, session_id)
        streamer: PhysiologyCurrentSnapshot | None = None
        operator: PhysiologyCurrentSnapshot | None = None
        for row in rows:
            snap = self._build_physiology_snapshot(row)
            if snap.subject_role == "streamer":
                streamer = snap
            elif snap.subject_role == "operator":
                operator = snap
        comod_row = q.fetch_latest_comodulation_row(cur, session_id)
        comodulation = self._build_co_modulation_summary(session_id, comod_row)
        return SessionPhysiologySnapshot(
            session_id=session_id,
            streamer=streamer,
            operator=operator,
            comodulation=comodulation,
            generated_at_utc=now,
        )

    # ------------------------------------------------------------------
    # Health synthesis (§12)
    # ------------------------------------------------------------------

    def _build_health_rows(
        self, pulse: dict[str, Any], now: datetime
    ) -> list[HealthSubsystemStatus]:
        """§12 — map per-subsystem last-success timestamps to health rows.

        `recovery_mode` and `operator_action_hint` are populated for
        DEGRADED/RECOVERING states so the UI reads them distinctly from
        outright ERROR (the degraded-but-recovering vs error distinction
        the plan calls out explicitly).
        """
        subsystems: list[tuple[str, str, str, str, str]] = [
            # (key, label, pulse_field, recovery_mode, action_hint)
            (
                "metrics",
                "Metrics Writer (Module E)",
                "last_metric_at",
                "db_retry_buffer",
                "confirm Persistent Store is reachable; check worker logs",
            ),
            (
                "physiology",
                "Physiology Persistence (§4.E.2)",
                "last_physio_at",
                "physio_adapter_retry",
                "check Oura webhook delivery and API ingress logs",
            ),
            (
                "comodulation",
                "Co-Modulation Analytics (§7C)",
                "last_comod_at",
                "awaiting_alignment",
                "null is valid until aligned non-stale pairs accumulate",
            ),
            (
                "encounters",
                "Encounter Logger (§4.E.1)",
                "last_encounter_at",
                "reward_pipeline_retry",
                "verify stimulus injection path and Thompson update loop",
            ),
        ]
        rows: list[HealthSubsystemStatus] = []
        for key, label, field, recovery_mode, hint in subsystems:
            last = _ensure_utc(pulse.get(field))
            state, detail = _classify_subsystem(last, now)
            rows.append(
                HealthSubsystemStatus(
                    subsystem_key=key,
                    label=label,
                    state=state,
                    last_success_utc=last,
                    detail=detail,
                    recovery_mode=(
                        recovery_mode
                        if state in {HealthState.DEGRADED, HealthState.RECOVERING}
                        else None
                    ),
                    operator_action_hint=(
                        hint if state in {HealthState.DEGRADED, HealthState.RECOVERING} else None
                    ),
                )
            )
        return rows

    def _build_health_snapshot(self, cur: Any, now: datetime) -> HealthSnapshot:
        pulse = q.fetch_subsystem_pulse(cur)
        rows = self._build_health_rows(pulse, now)
        degraded = sum(1 for r in rows if r.state is HealthState.DEGRADED)
        recovering = sum(1 for r in rows if r.state is HealthState.RECOVERING)
        errors = sum(1 for r in rows if r.state is HealthState.ERROR)
        unknown = sum(1 for r in rows if r.state is HealthState.UNKNOWN)
        overall: HealthState
        if errors > 0:
            overall = HealthState.ERROR
        elif degraded > 0:
            overall = HealthState.DEGRADED
        elif recovering > 0:
            overall = HealthState.RECOVERING
        elif unknown == len(rows):
            overall = HealthState.UNKNOWN
        else:
            overall = HealthState.OK
        return HealthSnapshot(
            generated_at_utc=now,
            overall_state=overall,
            subsystems=rows,
            degraded_count=degraded,
            recovering_count=recovering,
            error_count=errors,
        )

    # ------------------------------------------------------------------
    # Alert synthesis
    # ------------------------------------------------------------------

    def _list_alerts_internal(
        self, cur: Any, *, limit: int, since_utc: datetime | None
    ) -> list[AlertEvent]:
        alerts: list[AlertEvent] = []
        stale_rows = q.fetch_recent_stale_physiology(cur, since_utc=since_utc, limit=limit)
        for row in stale_rows:
            alerts.append(
                AlertEvent(
                    alert_id=f"physio-stale-{row['subject_role']}-{row['created_at']}",
                    severity=AlertSeverity.WARNING,
                    kind=AlertKind.PHYSIOLOGY_STALE,
                    message=(
                        f"{row['subject_role']} physiology stale "
                        f"(freshness {row['freshness_s']:.1f}s)"
                    ),
                    session_id=_as_uuid(row["session_id"]),
                    subsystem_key="physiology",
                    emitted_at_utc=_ensure_utc_strict(row["created_at"]),
                )
            )
        ended_rows = q.fetch_recently_ended_sessions(cur, since_utc=since_utc, limit=limit)
        for row in ended_rows:
            alerts.append(
                AlertEvent(
                    alert_id=f"session-ended-{row['session_id']}",
                    severity=AlertSeverity.INFO,
                    kind=AlertKind.SESSION_ENDED,
                    message="session ended",
                    session_id=_as_uuid(row["session_id"]),
                    subsystem_key=None,
                    emitted_at_utc=_ensure_utc_strict(row["ended_at"]),
                )
            )
        alerts.sort(key=lambda a: a.emitted_at_utc, reverse=True)
        return alerts[:limit]


# ----------------------------------------------------------------------
# Classification / coercion helpers
# ----------------------------------------------------------------------


def _classify_subsystem(
    last_success: datetime | None, now: datetime
) -> tuple[HealthState, str | None]:
    """§12 — map last-success age to a health state bucket."""
    if last_success is None:
        return HealthState.UNKNOWN, "no writes observed since boot"
    age = (now - last_success).total_seconds()
    if age < 0:
        age = 0.0
    if age <= _RECOVERING_WINDOW_S:
        return HealthState.OK, f"last write {age:.1f}s ago"
    if age <= _DEGRADED_WINDOW_S:
        return (
            HealthState.RECOVERING,
            f"last write {age:.1f}s ago — within self-heal window",
        )
    if age <= _ERROR_WINDOW_S:
        return (
            HealthState.DEGRADED,
            f"last write {age:.1f}s ago — exceeds healthy threshold",
        )
    return HealthState.ERROR, f"last write {age:.1f}s ago — subsystem silent"


def _encounter_state_for(row: dict[str, Any]) -> EncounterState:
    """Derive §7B encounter lifecycle state from persisted row fields."""
    is_valid = bool(row.get("is_valid"))
    gate = row.get("semantic_gate")
    n_frames = row.get("n_frames") or 0
    if is_valid:
        return EncounterState.COMPLETED
    if gate == 0:
        return EncounterState.REJECTED_GATE_CLOSED
    if n_frames == 0:
        return EncounterState.REJECTED_NO_FRAMES
    return EncounterState.MEASURING


def _as_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _latest_datetime(values: Any) -> datetime | None:
    """Return the most recent UTC-normalized datetime in a sequence."""
    latest: datetime | None = None
    for value in values:
        normalized = _ensure_utc(value)
        if normalized is None:
            continue
        if latest is None or normalized > latest:
            latest = normalized
    return latest


def _ensure_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _ensure_utc_strict(value: Any) -> datetime:
    result = _ensure_utc(value)
    if result is None:
        raise ValueError("expected UTC-aware datetime, got None")
    return result


def _stimulus_epoch_to_utc(value: Any) -> datetime | None:
    """`encounter_log.stimulus_time` is a drift-corrected epoch float (§4.C)."""
    if value is None:
        return None
    epoch = float(value)
    if epoch <= 0.0:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC)


# Stable unique identifier for synthesized alerts that need it (currently
# unused; reserved for future alert-ack path so dedup keys stay external).
def _new_alert_id() -> str:  # pragma: no cover - helper reserved for later phase
    return str(uuid4())
