"""
OperatorReadService — composes Operator Console DTOs from DB row dicts.

This is the only place in the API Server that performs cross-table
assembly for the Operator Console surfaces. Route handlers call into
this class; the repo layer (`services.api.repos.operator_queries`)
stays pure SQL. Every public method returns a validated Pydantic DTO
from `packages.schemas.operator_console`.

Design constraints:
  - No raw SQL here; the repo layer owns statements (§2 step 7).
  - No HTTP concerns; FastAPI plumbing stays in the route module.
  - No formatting for display; operator-facing text lives in
    `services.operator_console.formatters`.
  - Null is a first-class outcome for §7C co-modulation — always surface
    `null_reason` when the index is None rather than flattening to 0.0.
  - §12 degraded-but-recovering states must carry `recovery_mode` and
    `operator_action_hint` so the operator sees in-progress self-heal
    paths distinctly from hard errors.

Spec references:
  §4.C     — Orchestrator `_active_arm`, `_expected_greeting`,
             authoritative `_stimulus_time`.
  §4.C.4   — Physiological State Buffer freshness semantics.
  §4.E.1   — Operator-facing execution details for the PySide6 console.
  §4.E.2   — Physiology persistence schema (rmssd_ms, hr, provider).
  §4.E.3   — Attribution analytics persistence.
  §7B      — Thompson Sampling reward = p90_intensity × semantic_gate.
  §7C      — Rolling Co-Modulation Index; null-valid.
  §7E      — Event→outcome attribution diagnostics.
  §12      — Error-handling matrix (degraded/recovering vs error).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from types import ModuleType
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4

from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    AttributionSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    ObservationalAcousticSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
)
from services.api.db.connection import get_connection, put_connection
from services.api.repos import operator_queries as q
from services.api.services.subsystem_probes import ProbeResult, collect_subsystem_probes

logger = logging.getLogger(__name__)

AttributionFinality = Literal["online_provisional", "offline_final"]

_LIVE_SESSION_STATE_KEY_PREFIX: str = "operator:live_session:"


class RedisLiveStateClientLike(Protocol):
    """Minimal Redis surface for operator-read live-session overlays."""

    def get(self, name: str) -> Any: ...

    def close(self) -> None: ...


def _default_redis_factory() -> RedisLiveStateClientLike:
    """Lazy import — read-side live-session state is optional in tests."""

    import redis as redis_lib

    url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis_lib.Redis.from_url(url, decode_responses=True)  # type: ignore[no-any-return,no-untyped-call]


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
        redis_factory: Callable[[], RedisLiveStateClientLike] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        subsystem_probe_runner: Callable[
            ..., Awaitable[list[ProbeResult]]
        ] = collect_subsystem_probes,
        queries: ModuleType | Any = q,
    ) -> None:
        self._get_conn = get_conn
        self._put_conn = put_conn
        self._redis_factory = redis_factory
        self._clock = clock
        self._subsystem_probe_runner = subsystem_probe_runner
        # Query backend — either ``services.api.repos.operator_queries``
        # (Postgres, the default) or ``services.desktop_app.state
        # .sqlite_operator_queries`` (SQLite). Both expose the same
        # ``fetch_*`` callable surface returning identically-shaped row
        # dicts so the DTO builders below stay backend-agnostic.
        self._queries = queries

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def get_overview(self) -> OverviewSnapshot:
        """§4.E.1 — composed Overview card set."""
        now = self._clock()
        with self._cursor() as cur, self._live_state_client() as live_state_client:
            active_row = self._queries.fetch_active_session(cur)
            active_session = (
                self._build_session_summary(active_row, live_state_client)
                if active_row is not None
                else None
            )

            latest_encounter: LatestEncounterSummary | None = None
            experiment_summary: ExperimentSummary | None = None
            physiology: SessionPhysiologySnapshot | None = None

            if active_session is not None:
                enc_row = self._queries.fetch_latest_encounter(cur, active_session.session_id)
                latest_encounter = self._build_latest_encounter_summary(enc_row)

                exp_id = (enc_row or {}).get("experiment_id")
                if isinstance(exp_id, str) and exp_id:
                    arm_rows = self._queries.fetch_experiment_arms(cur, exp_id)
                    active_arm_row = self._queries.fetch_active_arm_for_experiment(cur, exp_id)
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
        with self._cursor() as cur, self._live_state_client() as live_state_client:
            rows = self._queries.fetch_recent_sessions(cur, limit=limit)
            return [self._build_session_summary(row, live_state_client) for row in rows]

    def get_session(self, session_id: UUID) -> SessionSummary | None:
        with self._cursor() as cur, self._live_state_client() as live_state_client:
            row = self._queries.fetch_session_by_id(cur, session_id)
            if row is None:
                return None
            return self._build_session_summary(row, live_state_client)

    def list_encounters(
        self,
        session_id: UUID,
        *,
        limit: int = 100,
        before_utc: datetime | None = None,
    ) -> list[EncounterSummary]:
        with self._cursor() as cur:
            rows = self._queries.fetch_session_encounters(
                cur, session_id, limit=limit, before_utc=before_utc
            )
        return [self._build_encounter_summary(row) for row in rows]

    def get_experiment_detail(self, experiment_id: str) -> ExperimentDetail | None:
        with self._cursor() as cur:
            arm_rows = self._queries.fetch_experiment_arms(cur, experiment_id)
            if not arm_rows:
                return None
            active_arm_row = self._queries.fetch_active_arm_for_experiment(cur, experiment_id)

        arms = [self._build_arm_summary(row) for row in arm_rows]
        active_arm_id = (active_arm_row or {}).get("arm") if active_arm_row is not None else None
        last_updated = _latest_datetime(row.get("updated_at") for row in arm_rows)
        label = _experiment_label_from_rows(experiment_id, arm_rows)
        return ExperimentDetail(
            experiment_id=experiment_id,
            label=label,
            active_arm_id=active_arm_id if isinstance(active_arm_id, str) else None,
            arms=arms,
            last_update_summary=None,
            last_updated_utc=_ensure_utc(last_updated),
        )

    def get_session_physiology(self, session_id: UUID) -> SessionPhysiologySnapshot | None:
        now = self._clock()
        with self._cursor() as cur:
            session = self._queries.fetch_session_by_id(cur, session_id)
            if session is None:
                return None
            return self._build_session_physiology(cur, session_id, now)

    async def get_health(self) -> HealthSnapshot:
        now = self._clock()
        with self._cursor() as cur:
            snapshot = self._build_health_snapshot(cur, now)
        probes = await self._collect_probe_rows(now)
        return HealthSnapshot(
            generated_at_utc=snapshot.generated_at_utc,
            overall_state=snapshot.overall_state,
            subsystems=snapshot.subsystems,
            subsystem_probes=probes,
            degraded_count=snapshot.degraded_count,
            recovering_count=snapshot.recovering_count,
            error_count=snapshot.error_count,
        )

    def list_alerts(
        self, *, limit: int = 50, since_utc: datetime | None = None
    ) -> list[AlertEvent]:
        with self._cursor() as cur:
            return self._list_alerts_internal(cur, limit=limit, since_utc=since_utc)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Yield a backend cursor inside a get_conn / put_conn envelope.

        The Postgres path (psycopg2) cursors are themselves context
        managers, so the inner ``with`` is required for proper close.
        Backends that do not implement ``__enter__`` on their cursor
        (e.g. ``sqlite3.Cursor``) must override this method — the
        ``SqliteOperatorReadService`` subclass does exactly that.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            with cur as scoped:
                yield scoped
        finally:
            self._put_conn(conn)

    @contextmanager
    def _live_state_client(self) -> Iterator[RedisLiveStateClientLike | None]:
        if self._redis_factory is None:
            yield None
            return
        client = self._redis_factory()
        try:
            yield client
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                logger.debug("operator live-session Redis client close failed", exc_info=True)

    # ------------------------------------------------------------------
    # Builders — DB row dict → Operator Console DTO
    # ------------------------------------------------------------------

    def _build_session_summary(
        self,
        row: dict[str, Any],
        live_state_client: RedisLiveStateClientLike | None = None,
    ) -> SessionSummary:
        """§4.E.1 — lightweight session card."""
        session_id = _as_uuid(row["session_id"])
        status = "active" if row.get("ended_at") is None else "ended"
        experiment_id = row.get("experiment_id")
        live_state = self._fetch_live_session_state(live_state_client, session_id)
        active_arm = _as_str((live_state or {}).get("active_arm")) or _as_str(row.get("active_arm"))
        expected_greeting = _as_str((live_state or {}).get("expected_greeting"))
        return SessionSummary(
            session_id=session_id,
            status=status,
            started_at_utc=_ensure_utc_strict(row["started_at"]),
            ended_at_utc=_ensure_utc(row.get("ended_at")),
            duration_s=_as_float(row.get("duration_s")),
            experiment_id=experiment_id if isinstance(experiment_id, str) else None,
            active_arm=active_arm,
            expected_greeting=expected_greeting,
            is_calibrating=_as_optional_bool(
                (live_state or {}).get("is_calibrating", row.get("is_calibrating"))
            ),
            calibration_frames_accumulated=_as_int(
                (live_state or {}).get(
                    "calibration_frames_accumulated",
                    row.get("calibration_frames_accumulated"),
                )
            ),
            calibration_frames_required=_as_int(
                (live_state or {}).get(
                    "calibration_frames_required",
                    row.get("calibration_frames_required"),
                )
            ),
            last_segment_completed_at_utc=_ensure_utc(row.get("last_segment_completed_at_utc")),
            latest_reward=_as_float(row.get("latest_reward")),
            latest_semantic_gate=_as_int(row.get("latest_semantic_gate")),
        )

    def _fetch_live_session_state(
        self,
        client: RedisLiveStateClientLike | None,
        session_id: UUID,
    ) -> dict[str, Any] | None:
        if client is None:
            return None
        try:
            raw = client.get(_live_session_state_key(session_id))
        except Exception:  # noqa: BLE001
            logger.debug("operator live-session state read unavailable", exc_info=True)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed live-session state ignored for %s", session_id)
            return None
        if not isinstance(payload, dict):
            logger.warning("Unexpected live-session state payload ignored for %s", session_id)
            return None
        return payload

    def _build_encounter_summary(self, row: dict[str, Any]) -> EncounterSummary:
        """§7B — full reward-explanation row. Notes flag physiology staleness."""
        state = _encounter_state_for(row)
        notes: list[str] = []
        if row.get("semantic_gate") == 0:
            notes.append("gate-closed: reward zero per §7B")
        if (row.get("n_frames_in_window") or 0) == 0:
            notes.append("no valid frames in measurement window")
        observational_acoustic = self._build_observational_acoustic_summary(row)
        semantic_evaluation = self._build_semantic_evaluation_summary(row)
        attribution = self._build_attribution_summary(row)
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
            n_frames_in_window=_as_int(row.get("n_frames_in_window")),
            au12_baseline_pre=_as_float(row.get("au12_baseline_pre")),
            observational_acoustic=observational_acoustic,
            semantic_evaluation=semantic_evaluation,
            attribution=attribution,
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
        observational_acoustic = self._build_observational_acoustic_summary(row)
        semantic_evaluation = self._build_semantic_evaluation_summary(row)
        attribution = self._build_attribution_summary(row)
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
            n_frames_in_window=_as_int(row.get("n_frames_in_window")),
            observational_acoustic=observational_acoustic,
            semantic_evaluation=semantic_evaluation,
            attribution=attribution,
        )

    def _build_semantic_evaluation_summary(
        self, row: dict[str, Any]
    ) -> SemanticEvaluationSummary | None:
        """Hydrate §7E semantic attribution readback with all-null → None."""
        semantic_reasoning = _as_str(row.get("semantic_reasoning"))
        semantic_is_match = _as_optional_bool(row.get("semantic_is_match"))
        semantic_confidence_score = _as_float(row.get("semantic_confidence_score"))
        semantic_method = _as_str(row.get("semantic_method"))
        semantic_method_version = _as_str(row.get("semantic_method_version"))

        semantic_values: tuple[str | bool | float | None, ...] = (
            semantic_reasoning,
            semantic_is_match,
            semantic_confidence_score,
            semantic_method,
            semantic_method_version,
        )
        if all(value is None for value in semantic_values):
            return None

        return SemanticEvaluationSummary(
            reasoning=semantic_reasoning,
            is_match=semantic_is_match,
            confidence_score=semantic_confidence_score,
            semantic_method=semantic_method,
            semantic_method_version=semantic_method_version,
        )

    def _build_attribution_summary(self, row: dict[str, Any]) -> AttributionSummary | None:
        """Hydrate §7E attribution diagnostics with all-null → None.

        Use identity-based null checks so legitimate ``False``/``0`` source
        values still produce a populated DTO.
        """
        finality = _as_str(row.get("attribution_finality"))
        soft_reward_candidate = _as_float(row.get("soft_reward_candidate"))
        au12_baseline_pre = _as_float(row.get("attribution_au12_baseline_pre"))
        au12_lift_p90 = _as_float(row.get("au12_lift_p90"))
        au12_lift_peak = _as_float(row.get("au12_lift_peak"))
        au12_peak_latency_ms = _as_float(row.get("au12_peak_latency_ms"))
        sync_peak_corr = _as_float(row.get("sync_peak_corr"))
        sync_peak_lag = _as_integral_int(row.get("sync_peak_lag"))
        outcome_link_lag_s = _as_float(row.get("outcome_link_lag_s"))

        attribution_values: tuple[str | float | int | None, ...] = (
            finality,
            soft_reward_candidate,
            au12_baseline_pre,
            au12_lift_p90,
            au12_lift_peak,
            au12_peak_latency_ms,
            sync_peak_corr,
            sync_peak_lag,
            outcome_link_lag_s,
        )
        if all(value is None for value in attribution_values):
            return None

        return AttributionSummary(
            finality=cast(AttributionFinality | None, finality),
            soft_reward_candidate=soft_reward_candidate,
            au12_baseline_pre=au12_baseline_pre,
            au12_lift_p90=au12_lift_p90,
            au12_lift_peak=au12_lift_peak,
            au12_peak_latency_ms=au12_peak_latency_ms,
            sync_peak_corr=sync_peak_corr,
            sync_peak_lag=sync_peak_lag,
            outcome_link_lag_s=outcome_link_lag_s,
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
        if row.get("metrics_row_id") is None and all(value is None for value in acoustic_values):
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
            greeting_text=str(row.get("greeting_text") or row["arm"]),
            posterior_alpha=alpha,
            posterior_beta=beta,
            evaluation_variance=variance,
            selection_count=_as_int(row.get("selection_count")) or 0,
            recent_reward_mean=_as_float(row.get("recent_reward_mean")),
            recent_semantic_pass_rate=_as_float(row.get("recent_semantic_pass_rate")),
            enabled=_as_bool(row.get("enabled"), default=True),
            end_dated_at=_ensure_utc(row.get("end_dated_at")),
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
            label=_experiment_label_from_rows(experiment_id, arm_rows),
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
        rows = self._queries.fetch_latest_physiology_rows(cur, session_id)
        streamer: PhysiologyCurrentSnapshot | None = None
        operator: PhysiologyCurrentSnapshot | None = None
        for row in rows:
            snap = self._build_physiology_snapshot(row)
            if snap.subject_role == "streamer":
                streamer = snap
            elif snap.subject_role == "operator":
                operator = snap
        comod_row = self._queries.fetch_latest_comodulation_row(cur, session_id)
        comodulation = self._build_co_modulation_summary(session_id, comod_row)
        return SessionPhysiologySnapshot(
            session_id=session_id,
            streamer=streamer,
            operator=operator,
            comodulation=comodulation,
            generated_at_utc=now,
        )

    # ------------------------------------------------------------------
    # Health probes + synthesis (§12)
    # ------------------------------------------------------------------

    async def _collect_probe_rows(self, checked_at: datetime) -> dict[str, HealthSubsystemProbe]:
        results = await self._subsystem_probe_runner(
            get_conn=self._get_conn,
            put_conn=self._put_conn,
            redis_factory=self._redis_factory,
            clock=self._clock,
        )
        rows: dict[str, HealthSubsystemProbe] = {}
        for result in results:
            rows[result.subsystem_key] = HealthSubsystemProbe(
                subsystem_key=result.subsystem_key,
                label=result.label,
                state=result.state,
                latency_ms=result.latency_ms,
                detail=result.detail,
                checked_at_utc=checked_at,
            )
        return rows

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
        pulse = self._queries.fetch_subsystem_pulse(cur)
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
        stale_rows = self._queries.fetch_recent_stale_physiology(
            cur, since_utc=since_utc, limit=limit
        )
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
        ended_rows = self._queries.fetch_recently_ended_sessions(
            cur, since_utc=since_utc, limit=limit
        )
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
    gate = row.get("semantic_gate")
    frames_in_window = row.get("n_frames_in_window") or 0
    if gate == 0:
        return EncounterState.REJECTED_GATE_CLOSED
    if frames_in_window == 0:
        return EncounterState.REJECTED_NO_FRAMES
    return EncounterState.COMPLETED


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


def _as_integral_int(value: Any) -> int | None:
    """Return an int only when the source value is exactly integral.

    Attribution sync_peak_lag is specified as an integer lag. Values may arrive
    from PostgreSQL numeric/double projections or string fixtures; fractional
    values must not be silently truncated before DTO validation.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer value must not be boolean")
    if isinstance(value, int):
        return value

    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"expected integral integer value, got {value!r}") from exc

    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise ValueError(f"expected integral integer value, got {value!r}")
    return int(decimal_value)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _experiment_label_from_rows(experiment_id: str, rows: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty label from experiment arm rows."""
    for row in rows:
        label = row.get("label")
        if isinstance(label, str) and label:
            return label
    return experiment_id if experiment_id else None


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
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
    if isinstance(value, str):
        parsed = _parse_iso_utc(value)
        if parsed is None:
            return None
        value = parsed
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _ensure_utc_strict(value: Any) -> datetime:
    result = _ensure_utc(value)
    if result is None:
        raise ValueError(f"expected UTC-aware datetime, got {value!r}")
    return result


def _parse_iso_utc(text: str) -> datetime | None:
    """Parse a UTC datetime from common ISO-8601 / SQLite encodings.

    Accepts ``'2026-05-01T12:34:56+00:00'``, ``'2026-05-01T12:34:56Z'``,
    and the SQLite-default ``'2026-05-01 12:34:56'`` (space separator,
    naive). Naive results are stamped UTC by ``_ensure_utc``.
    """
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    if "T" not in candidate and " " in candidate:
        candidate = candidate.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _stimulus_epoch_to_utc(value: Any) -> datetime | None:
    """`encounter_log.stimulus_time` is a drift-corrected epoch float (§4.C)."""
    if value is None:
        return None
    epoch = float(value)
    if epoch <= 0.0:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC)


def _live_session_state_key(session_id: UUID | str) -> str:
    return f"{_LIVE_SESSION_STATE_KEY_PREFIX}{session_id}"


# Stable unique identifier for synthesized alerts that need it (currently
# unused; reserved for future alert-ack path so dedup keys stay external).
def _new_alert_id() -> str:  # pragma: no cover - helper reserved for later phase
    return str(uuid4())
