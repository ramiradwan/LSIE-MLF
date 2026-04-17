"""
Operator-language formatters — Phase 3 of the Operator Console cycle.

Pure string helpers that translate DTO fields into the short, precise
phrases the operator UI shows. Nothing here creates widgets, talks to
the DB, or makes a network call; every function is a pure function of
its arguments so views/viewmodels stay simple and tests stay trivial.

Separating this from the widgets serves three purposes:

  1. One home for the language. When the spec wording shifts (e.g., §7B
     renames "semantic gate" to something else), we grep one file.
  2. View code never builds f-strings inline, so widgets remain layout-
     only and the viewmodel layer remains data-only.
  3. Reward-explanation text is the heart of operator trust; centralising
     it means the same phrasing is used on the Overview card and on the
     Live Session detail pane.

Spec references:
  §4.C.4     — physiology freshness / staleness vocabulary
  §7B        — reward = p90_intensity × semantic_gate; the explainer
               names the exact fields the pipeline used
  §7C        — co-modulation null-valid; we render the `null_reason`
               verbatim so "not enough aligned pairs yet" reads as a
               legitimate outcome, not a bug
  §12        — degraded-but-recovering vocabulary for health rows
"""

from __future__ import annotations

from datetime import datetime, timezone

from packages.schemas.operator_console import (
    CoModulationSummary,
    EncounterSummary,
    HealthState,
    HealthSubsystemStatus,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    UiStatusKind,
)

_EM_DASH = "—"
_DEGREE_OF_FRESHNESS_STALE_S = 60.0  # §4.C.4 default — UI-side fallback


# ----------------------------------------------------------------------
# Primitive formatters
# ----------------------------------------------------------------------


def format_timestamp(value: datetime | None) -> str:
    """UTC ISO-like timestamp with `Z` suffix. `—` when absent."""
    if value is None:
        return _EM_DASH
    if value.tzinfo is None or value.utcoffset() is None:
        # Defensive: our DTOs reject naive timestamps at the boundary,
        # but a defensively-coded viewmodel may still hand one in.
        return value.isoformat()
    as_utc = value.astimezone(timezone.utc)
    return as_utc.strftime("%Y-%m-%d %H:%M:%SZ")


def format_duration(seconds: float | None) -> str:
    """`h m s` for a non-negative duration; `—` when absent."""
    if seconds is None or seconds < 0:
        return _EM_DASH
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_percentage(ratio: float | None, *, digits: int = 1) -> str:
    """`12.3%` from a 0..1 ratio; `—` when absent."""
    if ratio is None:
        return _EM_DASH
    return f"{ratio * 100:.{digits}f}%"


def format_reward(value: float | None, *, digits: int = 3) -> str:
    """Reward values are unitless; keep three decimals for readability."""
    if value is None:
        return _EM_DASH
    return f"{value:.{digits}f}"


def format_semantic_gate(gate: int | None) -> str:
    """Map the 0/1 gate to operator language.

    §7B: `semantic_gate=0` zeros the reward. Communicating this in plain
    words prevents an operator from misreading a numeric zero as missing
    data.
    """
    if gate is None:
        return _EM_DASH
    if gate == 1:
        return "open (reward admitted)"
    return "closed (reward suppressed)"


def format_semantic_confidence(confidence: float | None) -> str:
    """Confidence ∈ [0,1], rendered as a percentage. `—` when absent."""
    return format_percentage(confidence, digits=0)


def format_health_state(state: HealthState) -> str:
    """Operator-facing state label. `DEGRADED` and `RECOVERING` are kept
    distinct from `ERROR` per §12."""
    mapping = {
        HealthState.OK: "ok",
        HealthState.DEGRADED: "degraded",
        HealthState.RECOVERING: "recovering",
        HealthState.ERROR: "error",
        HealthState.UNKNOWN: "unknown",
    }
    return mapping[state]


# ----------------------------------------------------------------------
# Freshness / physiology
# ----------------------------------------------------------------------


def format_freshness(
    freshness_s: float | None,
    *,
    is_stale: bool | None = None,
) -> str:
    """`12s (fresh)` / `78s (stale)` / `—`.

    §4.C.4 treats stale snapshots as first-class: the UI must say
    "stale", not hide the number.
    """
    if freshness_s is None:
        return _EM_DASH
    stale = (
        bool(is_stale)
        if is_stale is not None
        else freshness_s >= _DEGREE_OF_FRESHNESS_STALE_S
    )
    label = "stale" if stale else "fresh"
    return f"{freshness_s:.0f}s ({label})"


def truncate_expected_greeting(greeting: str | None, *, limit: int = 60) -> str:
    """Short-form expected greeting for compact table cells.

    A greeting often runs several sentences; the action bar shows the
    full text, but cards/tables truncate with an ellipsis so layout
    does not break.
    """
    if greeting is None:
        return _EM_DASH
    stripped = greeting.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 1)] + "…"


# ----------------------------------------------------------------------
# Co-Modulation Index (§7C) — null-valid rendering
# ----------------------------------------------------------------------


def format_comodulation_index(summary: CoModulationSummary | None) -> str:
    """`0.342` or, when null-valid, the `null_reason` text verbatim.

    §7C defines a null result as a valid outcome when insufficient
    aligned non-stale pairs exist. Rendering "insufficient aligned pairs"
    (the reason) instead of a zero or an error preserves the meaning.
    """
    if summary is None:
        return _EM_DASH
    if summary.co_modulation_index is None:
        return summary.null_reason or "insufficient data"
    return f"{summary.co_modulation_index:+.3f}"


# ----------------------------------------------------------------------
# Compound explainers — used on Overview cards and detail panes
# ----------------------------------------------------------------------


def build_reward_explanation(encounter: EncounterSummary) -> str:
    """Human-readable breakdown of the §7B gated reward.

    The text surfaces every input the pipeline used:
      `r_t = p90_intensity × semantic_gate`
    Calling out `semantic_gate=0` explicitly — "gate closed, reward
    suppressed" — prevents an operator from misreading a zero reward
    as a frame-count issue, and vice-versa for `n_frames_in_window=0`.
    """
    if encounter.n_frames_in_window == 0:
        return (
            "No valid AU12 frames in the measurement window — "
            "reward not computed."
        )
    if encounter.semantic_gate == 0:
        return (
            f"Semantic gate closed (confidence "
            f"{format_semantic_confidence(encounter.semantic_confidence)}) — "
            f"reward suppressed. P90 intensity "
            f"{format_reward(encounter.p90_intensity)} was observed but not "
            f"admitted."
        )
    parts: list[str] = []
    parts.append(
        f"P90 intensity {format_reward(encounter.p90_intensity)} × "
        f"semantic gate {encounter.semantic_gate if encounter.semantic_gate is not None else _EM_DASH} "
        f"= gated reward {format_reward(encounter.gated_reward)}."
    )
    if encounter.baseline_b_neutral is not None:
        parts.append(
            f"Baseline B_neutral {format_reward(encounter.baseline_b_neutral)}."
        )
    if encounter.n_frames_in_window is not None:
        parts.append(f"{encounter.n_frames_in_window} AU12 frame(s) in window.")
    if encounter.physiology_attached:
        stale = encounter.physiology_stale is True
        parts.append(
            "Physiology attached "
            + ("(stale snapshot)." if stale else "(fresh snapshot).")
        )
    return " ".join(parts)


def build_physiology_explanation(snapshot: SessionPhysiologySnapshot | None) -> str:
    """One-line physiology summary used on Overview / session header.

    Distinguishes fresh / stale / absent so an operator never has to
    guess whether "no value" means "not wearing the strap" or "snapshot
    is old".
    """
    if snapshot is None:
        return "No physiology data for this session."
    parts: list[str] = []
    parts.append(_role_line("streamer", snapshot.streamer))
    parts.append(_role_line("operator", snapshot.operator))
    parts.append(
        "Co-Modulation Index: " + format_comodulation_index(snapshot.comodulation)
    )
    return " • ".join(parts)


def _role_line(label: str, snap: PhysiologyCurrentSnapshot | None) -> str:
    if snap is None:
        return f"{label}: absent"
    if snap.rmssd_ms is None:
        return f"{label}: no RMSSD"
    freshness = format_freshness(snap.freshness_s, is_stale=snap.is_stale)
    hr = f"{snap.heart_rate_bpm} bpm" if snap.heart_rate_bpm is not None else _EM_DASH
    return f"{label}: RMSSD {snap.rmssd_ms:.0f}ms, HR {hr}, {freshness}"


def build_health_detail(row: HealthSubsystemStatus) -> str:
    """Health-row detail line combining state, recovery hint, and last-success
    timestamp. §12 recovery-mode states read distinct from errors.
    """
    parts: list[str] = [format_health_state(row.state)]
    if row.recovery_mode:
        parts.append(f"recovery: {row.recovery_mode}")
    if row.detail:
        parts.append(row.detail)
    if row.last_success_utc is not None:
        parts.append(f"last ok {format_timestamp(row.last_success_utc)}")
    if row.operator_action_hint:
        parts.append(f"hint: {row.operator_action_hint}")
    return " — ".join(parts)


def ui_status_for_health(row: HealthSubsystemStatus) -> UiStatusKind:
    """Map a subsystem's `HealthState` to a `UiStatusKind` palette bucket.

    Recovering is visually distinct from degraded so the operator sees
    that the subsystem is self-healing rather than "just broken".
    """
    mapping: dict[HealthState, UiStatusKind] = {
        HealthState.OK: UiStatusKind.OK,
        HealthState.DEGRADED: UiStatusKind.WARN,
        HealthState.RECOVERING: UiStatusKind.PROGRESS,
        HealthState.ERROR: UiStatusKind.ERROR,
        HealthState.UNKNOWN: UiStatusKind.NEUTRAL,
    }
    return mapping[row.state]
