"""
Reward Computation — Continuous Facial Affect Reward Pipeline

Computes the continuous [0, 1] reward for the fractional Beta-Bernoulli
Thompson Sampling engine from raw AU12 telemetry arrays.

Mathematical recipe (Formalizing TS for LSIE-MLF v2.0):
  1. Stimulus-locked windowing: extract AU12 values in [t_stim + 0.5s, t_stim + 5.0s]
  2. 90th percentile aggregation: robust to single-frame noise, captures sustained peak
  3. Per-subject range normalization: (score - B_neutral) / (x_max - B_neutral)
  4. Semantic validity gate: r_t = I_AU12_90th × G_t (G_t ∈ {0, 1})

The final gated reward r_t ∈ [0, 1] feeds directly into the fractional
Beta-Bernoulli update: α += r_t, β += (1 - r_t).

Spec references:
  §7.4 — AU12 baseline calibration and scoring
  §4.E.1 — Thompson Sampling reward update
  §8.2 — SemanticEvaluationResult (is_match boolean gate)
  §11 — Variable Extraction Matrix (AU12 Intensity Score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Stimulus-locked measurement window bounds (seconds relative to stimulus onset).
# Physiological basis (Ekman & Friesen 1982, Dimberg & Thunberg 1998):
#   - AU12 onset latency: 300-500ms post-stimulus (EMG), 780ms median (social)
#   - Genuine smile duration: 0.5-4.0 seconds
#   - Window [+0.5s, +5.0s] captures onset through offset with margin
WINDOW_START_OFFSET_S: float = 0.5
WINDOW_END_OFFSET_S: float = 5.0

# Pre-stimulus baseline window for per-segment B_neutral calibration.
# Uses frames from [t_stim - 5.0s, t_stim - 2.0s] to avoid onset contamination.
BASELINE_START_OFFSET_S: float = -5.0
BASELINE_END_OFFSET_S: float = -2.0

# Percentile for robust temporal aggregation.
# 90th percentile: rejects top 10% (noise artifacts) while capturing
# sustained near-peak engagement. Variance ≈ 0.09 / [n · f(Q_90)²],
# yielding SE ≈ 0.005 at 30fps × 4.5s window = 135 frames.
REWARD_PERCENTILE: float = 90.0

# Minimum frames required in the measurement window for a valid reward.
# Below this threshold, the P90 estimate is unreliable.
MIN_FRAMES_FOR_REWARD: int = 10

# Epsilon guard for range normalization denominator
RANGE_EPSILON: float = 1e-6


@dataclass
class TimestampedAU12:
    """A single AU12 observation with its timestamp.

    §11 — AU12 Intensity Score from Variable Extraction Matrix.
    Timestamps are drift-corrected UTC epoch seconds (§4.C.1).
    """

    timestamp_s: float
    intensity: float  # [0.0, 1.0] from AU12Normalizer.compute_bounded_intensity()


@dataclass
class RewardResult:
    """Complete reward computation result for encounter logging.

    Stores both the final gated reward (for the Thompson Sampling update)
    and the intermediate values (for auditability and retroactive analysis).

    Fields map to the Phase3_EncounterLog JSON schema.
    """

    # Final reward fed to Thompson Sampling: r_t = P90 × G_t
    gated_reward: float

    # Intermediate values for traceability
    p90_intensity: float  # 90th percentile of stimulus-window AU12
    semantic_gate: int  # G_t ∈ {0, 1}
    is_valid: bool  # True if enough frames existed for computation
    n_frames_in_window: int  # Number of AU12 frames in measurement window
    baseline_b_neutral: float | None  # Per-segment pre-stimulus baseline

    # Raw AU12 time series within the measurement window (for logging)
    au12_window_series: list[float] = field(default_factory=list)


def extract_stimulus_window(
    au12_series: list[TimestampedAU12],
    stimulus_time_s: float,
) -> list[TimestampedAU12]:
    """
    Extract AU12 observations within the stimulus-locked measurement window.

    Window: [t_stimulus + 0.5s, t_stimulus + 5.0s]

    Physiological basis: smile onset latency is 300-500ms (EMG),
    genuine smiles last 0.5-4.0s. The +0.5s start discards pre-onset
    noise; the +5.0s end captures full offset with margin for
    delayed cognitive responses (humor stimuli: 1-3s onset).

    Args:
        au12_series: Timestamped AU12 observations for the full segment.
        stimulus_time_s: Drift-corrected UTC epoch of stimulus injection.

    Returns:
        Filtered list of AU12 observations within the measurement window.
    """
    window_start = stimulus_time_s + WINDOW_START_OFFSET_S
    window_end = stimulus_time_s + WINDOW_END_OFFSET_S

    return [obs for obs in au12_series if window_start <= obs.timestamp_s <= window_end]


def extract_baseline_window(
    au12_series: list[TimestampedAU12],
    stimulus_time_s: float,
) -> list[TimestampedAU12]:
    """
    Extract AU12 observations within the pre-stimulus baseline window.

    Window: [t_stimulus - 5.0s, t_stimulus - 2.0s]

    Used for per-segment neutral baseline calibration. The -2.0s end
    provides a buffer before stimulus onset to avoid anticipatory
    expression contamination.

    Args:
        au12_series: Timestamped AU12 observations for the full segment.
        stimulus_time_s: Drift-corrected UTC epoch of stimulus injection.

    Returns:
        Filtered list of AU12 observations within the baseline window.
    """
    baseline_start = stimulus_time_s + BASELINE_START_OFFSET_S
    baseline_end = stimulus_time_s + BASELINE_END_OFFSET_S

    return [obs for obs in au12_series if baseline_start <= obs.timestamp_s <= baseline_end]


def compute_p90(values: list[float]) -> float:
    """
    Compute the 90th percentile of a list of AU12 intensities.

    Order statistic properties (Mosteller 1946):
        Var(Q̂_p) = p(1-p) / [n · f(Q_p)²]
    At p=0.90, n=135 (30fps × 4.5s), f(Q_90) ≈ 2:
        SE ≈ sqrt(0.09 / (135 × 4)) ≈ 0.004 — negligible.

    Uses numpy's linear interpolation method for consistency with
    standard statistical packages.

    Args:
        values: List of AU12 intensity values in [0.0, 1.0].

    Returns:
        90th percentile value.
    """
    return float(np.percentile(values, REWARD_PERCENTILE))


def compute_reward(
    au12_series: list[TimestampedAU12],
    stimulus_time_s: float,
    is_match: bool,
    confidence_score: float = 0.0,
    x_max: float | None = None,
) -> RewardResult:
    """
    Compute the continuous gated reward for Thompson Sampling update.

    Full pipeline:
      1. Extract pre-stimulus baseline window → per-segment B_neutral
      2. Extract post-stimulus measurement window
      3. Apply per-subject range normalization if x_max is available
      4. Compute 90th percentile of normalized intensities
      5. Apply semantic validity gate: r_t = P90 × G_t

    If the semantic gate is closed (is_match=False), the reward is
    deterministically crushed to 0.0, preventing false positive
    posterior contamination from contextually unrelated AU12 spikes.

    If insufficient frames exist in the measurement window (< MIN_FRAMES),
    the result is marked invalid and returns 0.0 — the Thompson Sampling
    engine should skip the update entirely for this encounter.

    Args:
        au12_series: Full segment of timestamped AU12 observations.
        stimulus_time_s: Drift-corrected UTC epoch of stimulus injection.
        is_match: Semantic validity gate output from Module D (§8.2).
        confidence_score: LLM confidence in the semantic match (0-1).
        x_max: Estimated maximum AU12 response capability for this subject.
                If provided, range normalization is applied. If None,
                raw bounded intensities are used directly.

    Returns:
        RewardResult with the gated reward and all intermediate values.
    """
    # --- Step 1: Per-segment baseline from pre-stimulus window ---
    baseline_obs = extract_baseline_window(au12_series, stimulus_time_s)
    segment_baseline: float | None = None
    if baseline_obs:
        segment_baseline = float(np.mean([obs.intensity for obs in baseline_obs]))

    # --- Step 2: Extract stimulus-locked measurement window ---
    window_obs = extract_stimulus_window(au12_series, stimulus_time_s)
    n_frames = len(window_obs)

    if n_frames < MIN_FRAMES_FOR_REWARD:
        logger.warning(
            "Insufficient frames in measurement window: %d < %d (stimulus_t=%.3f)",
            n_frames,
            MIN_FRAMES_FOR_REWARD,
            stimulus_time_s,
        )
        return RewardResult(
            gated_reward=0.0,
            p90_intensity=0.0,
            semantic_gate=1 if is_match else 0,
            is_valid=False,
            n_frames_in_window=n_frames,
            baseline_b_neutral=segment_baseline,
        )

    # --- Step 3: Per-subject range normalization (if x_max available) ---
    raw_intensities = [obs.intensity for obs in window_obs]

    if x_max is not None and segment_baseline is not None:
        # Range normalization: maps [B_neutral, x_max] → [0, 1]
        # Handles multiplicative scaling bias across subjects
        denom = x_max - segment_baseline
        if denom > RANGE_EPSILON:
            normalized = [
                max(0.0, min(1.0, (v - segment_baseline) / denom)) for v in raw_intensities
            ]
        else:
            # Subject's expressiveness range is negligible — use raw values
            normalized = raw_intensities
    else:
        # No x_max calibration available — use raw bounded intensities
        # (already in [0, 1] from compute_bounded_intensity via tanh)
        normalized = raw_intensities

    # --- Step 4: 90th percentile robust aggregation ---
    p90: float = compute_p90(normalized)

    # --- Step 5: Semantic validity gate ---
    gate: int = 1 if is_match else 0
    gated_reward: float = p90 * gate

    # Clamp to [0, 1] as a safety net (should already be bounded)
    gated_reward = max(0.0, min(1.0, gated_reward))

    return RewardResult(
        gated_reward=gated_reward,
        p90_intensity=p90,
        semantic_gate=gate,
        is_valid=True,
        n_frames_in_window=n_frames,
        baseline_b_neutral=segment_baseline,
        au12_window_series=[obs.intensity for obs in window_obs],
    )
