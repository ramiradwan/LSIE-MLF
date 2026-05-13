"""Pure attribution ledger builders for §4.E.3 / §7E analytics.

The module derives deterministic UUIDv5 identifiers, baseline-aware AU12
diagnostics, semantic metadata, event/outcome links, and score records for the
§6.4 attribution schemas. It performs no direct Persistent Store I/O and has no
reward, posterior-update, or arm-selection side effects; attribution remains
observational over the §7B reward path (§13.15).
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from packages.schemas.attribution import (
    AttributionEvent,
    AttributionScore,
    BanditDecisionSnapshot,
    EventOutcomeLink,
    OutcomeEvent,
)
from packages.schemas.evaluation import SEMANTIC_METHODS, SEMANTIC_REASON_CODES

ATTRIBUTION_SCHEMA_VERSION: str = "v3.4"
ATTRIBUTION_FINALITY_ONLINE: str = "online_provisional"
ATTRIBUTION_FINALITY_OFFLINE: str = "offline_final"
ATTRIBUTION_EVENT_TYPE_STIMULUS: Literal["stimulus_interaction"] = "stimulus_interaction"
OUTCOME_TYPE_CREATOR_FOLLOW: Literal["creator_follow"] = "creator_follow"
DEFAULT_REWARD_PATH_VERSION: str = "7B.v3.4"
DEFAULT_LINK_RULE_VERSION: str = "event_outcome_lag_horizon_v1"
DEFAULT_ATTRIBUTION_HORIZON_S: float = 7 * 24 * 60 * 60.0
DEFAULT_ATTRIBUTION_METHOD_VERSION: str = "7E.v1"

# Deterministic project namespace for all attribution UUIDv5 identifiers.
ATTRIBUTION_UUID_NAMESPACE: uuid.UUID = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "urn:lsie-mlf:attribution-ledger:v3.4",
)

STIMULUS_WINDOW_START: float = 0.5
STIMULUS_WINDOW_END: float = 5.0
BASELINE_WINDOW_START: float = -5.0
BASELINE_WINDOW_END: float = -2.0


@dataclass(frozen=True)
class AttributionLedgerRecords:
    """
    Bundle caller-ready attribution records for persistence.

    Accepts the materialized §6.4 event, outcome, link, and score schema
    instances assembled by helper functions and produces an immutable grouping
    for callers to write. It does not generate identifiers, query the Persistent
    Store, or mutate reward and bandit state.
    """

    event: AttributionEvent
    outcomes: tuple[OutcomeEvent, ...]
    links: tuple[EventOutcomeLink, ...]
    scores: tuple[AttributionScore, ...]


def _canonical_part(value: Any) -> str:
    """Serialize UUIDv5 name components without locale/process variance."""

    if value is None:
        return "<null>"
    if isinstance(value, datetime):
        return _to_utc(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "<nonfinite>"
        return repr(value)
    return str(value)


def deterministic_uuid5(kind: str, *parts: Any) -> uuid.UUID:
    """Return a deterministic UUIDv5 for one attribution ledger identity tuple."""

    name = "|".join((kind, *(_canonical_part(part) for part in parts)))
    return uuid.uuid5(ATTRIBUTION_UUID_NAMESPACE, name)


def attribution_event_id(
    *,
    session_id: str | uuid.UUID,
    segment_id: str,
    event_type: str,
    reward_path_version: str,
) -> uuid.UUID:
    """UUIDv5(session_id, segment_id, event_type, reward_path_version)."""

    return deterministic_uuid5(
        "attribution_event",
        session_id,
        segment_id,
        event_type,
        reward_path_version,
    )


def outcome_event_id(
    *,
    session_id: str | uuid.UUID,
    outcome_type: str,
    outcome_time_utc: datetime,
    source_system: str,
    source_event_ref: str | None,
) -> uuid.UUID:
    """UUIDv5(session_id, outcome_type, outcome_time_utc, source_system, ref)."""

    return deterministic_uuid5(
        "outcome_event",
        session_id,
        outcome_type,
        outcome_time_utc,
        source_system,
        source_event_ref,
    )


def event_outcome_link_id(
    *,
    event_id: str | uuid.UUID,
    outcome_id: str | uuid.UUID,
    link_rule_version: str,
) -> uuid.UUID:
    """UUIDv5(event_id, outcome_id, link_rule_version)."""

    return deterministic_uuid5(
        "event_outcome_link",
        event_id,
        outcome_id,
        link_rule_version,
    )


def attribution_score_id(
    *,
    event_id: str | uuid.UUID,
    outcome_id: str | uuid.UUID | None,
    attribution_method: str,
    method_version: str,
) -> uuid.UUID:
    """UUIDv5(event_id, outcome_id, method, method_version).

    Finality is intentionally excluded from the deterministic identity tuple so
    an online_provisional score can be replayed/upserted as offline_final
    without creating a second AttributionScore record. The lifecycle state
    remains a mutable persisted AttributionScore.finality field.
    """

    return deterministic_uuid5(
        "attribution_score",
        event_id,
        outcome_id,
        attribution_method,
        method_version,
    )


def expected_rule_text_hash(rule_text: str) -> str:
    """Return the canonical SHA-256 hash for persisted rule-text identity."""

    normalized = " ".join(rule_text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse an RFC3339/datetime value to aware UTC, returning None on invalid input."""

    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, str) and value:
        try:
            return _to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def epoch_s_to_utc_datetime(value: Any) -> datetime | None:
    """Convert finite UTC epoch seconds to an aware UTC datetime."""

    number = finite_float_or_none(value)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def finite_float_or_none(value: Any) -> float | None:
    """Return a finite float or None without raising for malformed inputs."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probability_or_none(value: Any) -> float | None:
    number = finite_float_or_none(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def _unique_sorted_flags(flags: Iterable[str]) -> list[str]:
    return sorted({flag for flag in flags if isinstance(flag, str) and flag})


def compute_soft_reward_candidate(
    p_match: float | None, proxy_reward: float | None
) -> float | None:
    """Compute the observational §7E soft semantic reward candidate."""

    p = _probability_or_none(p_match)
    reward = finite_float_or_none(proxy_reward)
    if p is None or reward is None:
        return None
    return p * max(0.0, min(1.0, reward))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_au12_attribution_metrics(
    au12_series: Iterable[dict[str, Any]],
    stimulus_time_s: float | None,
) -> dict[str, float | None]:
    """Compute baseline-aware AU12 attribution diagnostics for §7E scores."""

    stimulus = finite_float_or_none(stimulus_time_s)
    if stimulus is None:
        return {
            "au12_baseline_pre": None,
            "au12_lift_p90": None,
            "au12_lift_peak": None,
            "au12_peak_latency_ms": None,
        }

    baseline_values: list[float] = []
    measure_obs: list[tuple[float, float]] = []
    for obs in au12_series:
        timestamp = finite_float_or_none(obs.get("timestamp_s"))
        intensity = finite_float_or_none(obs.get("intensity"))
        if timestamp is None or intensity is None:
            continue
        bounded_intensity = max(0.0, min(1.0, intensity))
        if stimulus + BASELINE_WINDOW_START <= timestamp <= stimulus + BASELINE_WINDOW_END:
            baseline_values.append(bounded_intensity)
        if stimulus + STIMULUS_WINDOW_START <= timestamp <= stimulus + STIMULUS_WINDOW_END:
            measure_obs.append((timestamp, bounded_intensity))

    baseline_pre = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
    if not measure_obs:
        return {
            "au12_baseline_pre": baseline_pre if baseline_values else None,
            "au12_lift_p90": None,
            "au12_lift_peak": None,
            "au12_peak_latency_ms": None,
        }

    values = [value for _, value in measure_obs]
    p90_post = _percentile(values, 90.0)
    peak_timestamp, peak_value = max(measure_obs, key=lambda item: item[1])
    return {
        "au12_baseline_pre": baseline_pre,
        "au12_lift_p90": None if p90_post is None else max(0.0, p90_post - baseline_pre),
        "au12_lift_peak": max(0.0, peak_value - baseline_pre),
        "au12_peak_latency_ms": (peak_timestamp - stimulus) * 1000.0,
    }


def _semantic_metadata(metrics: dict[str, Any]) -> dict[str, Any]:
    raw_semantic = metrics.get("semantic")
    semantic = cast(dict[str, Any], raw_semantic) if isinstance(raw_semantic, dict) else {}
    method = semantic.get("semantic_method") or metrics.get("semantic_method") or "cross_encoder"
    if method not in SEMANTIC_METHODS:
        method = "cross_encoder"
    version = (
        semantic.get("semantic_method_version")
        or metrics.get("semantic_method_version")
        or "unknown"
    )
    p_match = _probability_or_none(
        semantic.get("confidence_score", metrics.get("semantic_p_match"))
    )
    reason_code = semantic.get("reasoning") or semantic.get("semantic_reason_code")
    if reason_code not in SEMANTIC_REASON_CODES:
        reason_code = None
    return {
        "semantic_method": method,
        "semantic_method_version": str(version),
        "semantic_p_match": p_match,
        "semantic_reason_code": reason_code,
    }


def _bandit_snapshot(metrics: dict[str, Any]) -> BanditDecisionSnapshot | None:
    raw = metrics.get("_bandit_decision_snapshot") or metrics.get("bandit_decision_snapshot")
    if raw is None:
        return None
    try:
        return BanditDecisionSnapshot.model_validate(raw)
    except Exception:
        return None


def _outcome_inputs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in (
        "_outcome_event",
        "outcome_event",
        "_creator_follow_outcome",
        "_attribution_outcome",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for key in ("_outcome_events", "outcome_events"):
        value = metrics.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    creator_follow = metrics.get("_creator_follow")
    if creator_follow is True:
        candidates.append(
            {
                "outcome_type": OUTCOME_TYPE_CREATOR_FOLLOW,
                "outcome_value": 1.0,
                "outcome_time_utc": metrics.get("timestamp_utc"),
                "source_system": "module_e_metrics",
                "source_event_ref": metrics.get("segment_id"),
                "confidence": 1.0,
            }
        )
    elif isinstance(creator_follow, dict):
        candidates.append(creator_follow)
    return candidates


def _build_outcome_event(
    raw: dict[str, Any],
    *,
    session_id: str,
    finality: str,
    schema_version: str,
    created_at: datetime,
) -> OutcomeEvent | None:
    outcome_type = str(raw.get("outcome_type") or raw.get("event_type") or "").lower()
    if outcome_type in {"follow", "user_follow", "followevent", "creator_follow"}:
        outcome_type = OUTCOME_TYPE_CREATOR_FOLLOW
    if outcome_type != OUTCOME_TYPE_CREATOR_FOLLOW:
        return None

    outcome_time = parse_utc_datetime(
        raw.get("outcome_time_utc") or raw.get("timestamp_utc") or raw.get("event_time_utc")
    )
    if outcome_time is None:
        return None

    source_system = str(raw.get("source_system") or "tiktok_webcast")
    raw_payload = raw.get("payload")
    payload = cast(dict[str, Any], raw_payload) if isinstance(raw_payload, dict) else {}
    source_event_ref = raw.get("source_event_ref")
    if source_event_ref is None:
        source_event_ref = raw.get("event_id") or raw.get("message_id") or payload.get("event_id")
    if source_event_ref is not None:
        source_event_ref = str(source_event_ref)

    outcome_value = finite_float_or_none(raw.get("outcome_value"))
    if outcome_value is None:
        outcome_value = 1.0
    confidence = _probability_or_none(raw.get("confidence"))
    if confidence is None:
        confidence = 1.0

    try:
        return OutcomeEvent(
            outcome_id=outcome_event_id(
                session_id=session_id,
                outcome_type=outcome_type,
                outcome_time_utc=outcome_time,
                source_system=source_system,
                source_event_ref=source_event_ref,
            ),
            session_id=uuid.UUID(session_id),
            outcome_type=outcome_type,
            outcome_value=outcome_value,
            outcome_time_utc=outcome_time,
            source_system=source_system,
            source_event_ref=source_event_ref,
            confidence=confidence,
            finality=finality,  # type: ignore[arg-type]
            schema_version=schema_version,
            created_at=created_at,
        )
    except Exception:
        return None


def _reward_proxy(reward_result: Any, metrics: dict[str, Any]) -> float | None:
    if reward_result is not None:
        p90 = finite_float_or_none(getattr(reward_result, "p90_intensity", None))
        if p90 is not None:
            return p90
        reward = finite_float_or_none(getattr(reward_result, "gated_reward", None))
        if reward is not None:
            return reward
    return finite_float_or_none(metrics.get("au12_intensity"))


def _score(
    *,
    event_id: uuid.UUID,
    outcome_id_value: uuid.UUID | None,
    attribution_method: str,
    method_version: str,
    score_raw: float | None,
    score_normalized: float | None = None,
    confidence: float | None = None,
    evidence_flags: Iterable[str] = (),
    finality: str,
    schema_version: str,
    created_at: datetime,
) -> AttributionScore:
    return AttributionScore(
        score_id=attribution_score_id(
            event_id=event_id,
            outcome_id=outcome_id_value,
            attribution_method=attribution_method,
            method_version=method_version,
        ),
        event_id=event_id,
        outcome_id=outcome_id_value,
        attribution_method=attribution_method,
        method_version=method_version,
        score_raw=score_raw,
        score_normalized=score_normalized,
        confidence=confidence,
        evidence_flags=_unique_sorted_flags(evidence_flags),
        finality=finality,  # type: ignore[arg-type]
        schema_version=schema_version,
        created_at=created_at,
    )


def _build_scores(
    *,
    event: AttributionEvent,
    outcome_id_value: uuid.UUID | None,
    metrics: dict[str, Any],
    reward_result: Any,
    comodulation_result: dict[str, Any] | None,
    finality: str,
    schema_version: str,
    created_at: datetime,
) -> tuple[AttributionScore, ...]:
    au12_metrics = compute_au12_attribution_metrics(
        metrics.get("_au12_series") or [],
        finite_float_or_none(metrics.get("_stimulus_time")),
    )
    p_match = event.semantic_p_match
    proxy_reward = _reward_proxy(reward_result, metrics)
    soft_reward = compute_soft_reward_candidate(p_match, proxy_reward)

    # §7E.5 / §11.5.13-14 are lag-aware: these fields are valid only when an
    # upstream lag scan has already supplied its peak correlation and associated
    # lag. The legacy zero-lag co_modulation_index is deliberately not a
    # substitute. When no lag-aware result is present, persist NULL scores so the
    # ledger records analytic unavailability without synthesizing proxy evidence.
    sync_peak_corr = finite_float_or_none(metrics.get("sync_peak_corr"))
    sync_peak_lag = finite_float_or_none(metrics.get("sync_peak_lag"))
    if comodulation_result is not None:
        if sync_peak_corr is None:
            sync_peak_corr = finite_float_or_none(comodulation_result.get("sync_peak_corr"))
        if sync_peak_lag is None:
            sync_peak_lag = finite_float_or_none(comodulation_result.get("sync_peak_lag"))
    sync_flags: tuple[str, ...] = (
        ("lag_scan_result",) if sync_peak_corr is not None or sync_peak_lag is not None else ()
    )

    confidence = _probability_or_none(p_match)
    event_id_value = event.event_id
    return (
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="soft_reward_candidate",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=soft_reward,
            score_normalized=soft_reward,
            confidence=confidence,
            evidence_flags=("semantic_p_match", "au12_reward_proxy"),
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="au12_lift_p90",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=au12_metrics["au12_lift_p90"],
            score_normalized=au12_metrics["au12_lift_p90"],
            evidence_flags=("au12_series", "baseline_pre"),
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="au12_lift_peak",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=au12_metrics["au12_lift_peak"],
            score_normalized=au12_metrics["au12_lift_peak"],
            evidence_flags=("au12_series", "baseline_pre"),
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="au12_peak_latency_ms",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=au12_metrics["au12_peak_latency_ms"],
            evidence_flags=("au12_series", "stimulus_time"),
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="sync_peak_corr",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=sync_peak_corr,
            score_normalized=sync_peak_corr,
            evidence_flags=sync_flags,
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
        _score(
            event_id=event_id_value,
            outcome_id_value=outcome_id_value,
            attribution_method="sync_peak_lag",
            method_version=DEFAULT_ATTRIBUTION_METHOD_VERSION,
            score_raw=sync_peak_lag,
            evidence_flags=sync_flags,
            finality=finality,
            schema_version=schema_version,
            created_at=created_at,
        ),
    )


def build_attribution_ledger_records(
    metrics: dict[str, Any],
    *,
    reward_result: Any = None,
    comodulation_result: dict[str, Any] | None = None,
    finality: str = ATTRIBUTION_FINALITY_ONLINE,
    schema_version: str = ATTRIBUTION_SCHEMA_VERSION,
    created_at: datetime | None = None,
    reward_path_version: str = DEFAULT_REWARD_PATH_VERSION,
    horizon_s: float = DEFAULT_ATTRIBUTION_HORIZON_S,
    link_rule_version: str = DEFAULT_LINK_RULE_VERSION,
) -> AttributionLedgerRecords | None:
    """Build replay-safe §7E ledger records from a Module E metrics payload.

    Returns None when the required v3.4 event identity inputs are absent or
    invalid. Missing outcomes, outcome timestamps outside the attribution
    horizon, and null linkage are represented by empty outcome/link tuples and
    score records with ``outcome_id=None`` rather than exceptions.
    """

    created = _to_utc(created_at or datetime.now(UTC))
    session_id = metrics.get("session_id")
    segment_id = metrics.get("segment_id")
    if not isinstance(session_id, str) or not isinstance(segment_id, str):
        return None
    if len(segment_id) != 64 or any(ch not in "0123456789abcdef" for ch in segment_id):
        return None

    snapshot = _bandit_snapshot(metrics)
    if snapshot is None:
        return None

    event_time = parse_utc_datetime(metrics.get("event_time_utc") or metrics.get("timestamp_utc"))
    if event_time is None:
        return None
    stimulus_time_utc = parse_utc_datetime(metrics.get("stimulus_time_utc"))
    if stimulus_time_utc is None:
        stimulus_time_utc = epoch_s_to_utc_datetime(metrics.get("_stimulus_time"))

    expected_stimulus_rule = str(
        metrics.get("_expected_stimulus_rule")
        or metrics.get("expected_stimulus_rule")
        or snapshot.expected_stimulus_rule
    )
    expected_response_rule = str(
        metrics.get("_expected_response_rule")
        or metrics.get("expected_response_rule")
        or snapshot.expected_response_rule
    )
    active_arm = str(
        metrics.get("_active_arm") or metrics.get("active_arm") or snapshot.selected_arm_id
    )
    if not active_arm:
        return None
    semantic = _semantic_metadata(metrics)
    response_inference = metrics.get("response_inference")
    if not isinstance(response_inference, dict):
        response_inference = {}
    stimulus_id_raw = metrics.get("_stimulus_id") or metrics.get("stimulus_id")
    try:
        stimulus_id = None if stimulus_id_raw is None else uuid.UUID(str(stimulus_id_raw))
    except (TypeError, ValueError):
        stimulus_id = None
    matched_response_time_utc = parse_utc_datetime(metrics.get("matched_response_time_utc"))
    if matched_response_time_utc is None:
        matched_response_time_utc = epoch_s_to_utc_datetime(
            response_inference.get("matched_response_time")
        )
    response_registration_status = response_inference.get("registration_status") or metrics.get(
        "response_registration_status"
    )
    response_reason_code = response_inference.get("response_reason_code") or metrics.get(
        "response_reason_code"
    )

    try:
        event = AttributionEvent(
            event_id=attribution_event_id(
                session_id=session_id,
                segment_id=segment_id,
                event_type=ATTRIBUTION_EVENT_TYPE_STIMULUS,
                reward_path_version=reward_path_version,
            ),
            session_id=uuid.UUID(session_id),
            segment_id=segment_id,
            event_type=ATTRIBUTION_EVENT_TYPE_STIMULUS,
            event_time_utc=event_time,
            stimulus_time_utc=stimulus_time_utc,
            selected_arm_id=active_arm,
            expected_rule_text_hash=expected_rule_text_hash(expected_stimulus_rule),
            semantic_method=semantic["semantic_method"],
            semantic_method_version=semantic["semantic_method_version"],
            semantic_p_match=semantic["semantic_p_match"],
            semantic_reason_code=semantic["semantic_reason_code"],
            reward_path_version=reward_path_version,
            bandit_decision_snapshot=snapshot,
            evidence_flags=_unique_sorted_flags(("module_d_metrics", "bandit_decision_snapshot")),
            finality=finality,  # type: ignore[arg-type]
            schema_version=schema_version,
            created_at=created,
            stimulus_id=stimulus_id,
            stimulus_modality=(
                metrics.get("_stimulus_modality")
                or metrics.get("stimulus_modality")
                or snapshot.stimulus_modality
            ),
            matched_response_time_utc=matched_response_time_utc,
            response_registration_status=response_registration_status,
            response_reason_code=response_reason_code,
            expected_response_rule_text_hash=expected_rule_text_hash(expected_response_rule),
        )
    except Exception:
        return None

    outcomes = tuple(
        outcome
        for outcome in (
            _build_outcome_event(
                raw,
                session_id=session_id,
                finality=finality,
                schema_version=schema_version,
                created_at=created,
            )
            for raw in _outcome_inputs(metrics)
        )
        if outcome is not None
    )

    links: list[EventOutcomeLink] = []
    eligible_outcome_id: uuid.UUID | None = None
    for outcome in outcomes:
        lag_s = (outcome.outcome_time_utc - event.event_time_utc).total_seconds()
        if lag_s < 0.0 or lag_s > horizon_s:
            continue
        try:
            link = EventOutcomeLink(
                link_id=event_outcome_link_id(
                    event_id=event.event_id,
                    outcome_id=outcome.outcome_id,
                    link_rule_version=link_rule_version,
                ),
                event_id=event.event_id,
                outcome_id=outcome.outcome_id,
                lag_s=lag_s,
                horizon_s=horizon_s,
                link_rule_version=link_rule_version,
                eligibility_flags=_unique_sorted_flags(("same_session", "within_horizon")),
                finality=finality,  # type: ignore[arg-type]
                schema_version=schema_version,
                created_at=created,
            )
        except Exception:
            continue
        links.append(link)
        if eligible_outcome_id is None:
            eligible_outcome_id = outcome.outcome_id

    scores = _build_scores(
        event=event,
        outcome_id_value=eligible_outcome_id,
        metrics=metrics,
        reward_result=reward_result,
        comodulation_result=comodulation_result,
        finality=finality,
        schema_version=schema_version,
        created_at=created,
    )
    return AttributionLedgerRecords(
        event=event,
        outcomes=outcomes,
        links=tuple(links),
        scores=scores,
    )
