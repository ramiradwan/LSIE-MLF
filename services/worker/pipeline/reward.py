"""
Current §7B reward computation for the Thompson Sampling update.

The module converts bounded AU12 telemetry and a binary semantic gate into
``RewardResult`` values: pre-stimulus AU12 baseline diagnostic, stimulus-window
P90 intensity, semantic gate, and gated reward. Only bounded AU12 observations
and ``is_match`` affect the reward; semantic confidence, physiology, acoustics,
and attribution analytics remain observational under the reward-path invariance
controls (§8.6, §13.21).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stimulus-locked measurement window bounds (seconds relative to stimulus onset).
# Physiological basis (Ekman & Friesen 1982, Dimberg & Thunberg 1998):
#   - AU12 onset latency: 300-500ms post-stimulus (EMG), 780ms median (social)
#   - Genuine smile duration: 0.5-4.0 seconds
#   - Window [+0.5s, +5.0s] captures onset through offset with margin
WINDOW_START_OFFSET_S: float = 0.5
WINDOW_END_OFFSET_S: float = 5.0

# Pre-stimulus baseline window for per-segment diagnostic AU12 baseline.
# Uses frames from [t_stim - 5.0s, t_stim - 2.0s] to avoid onset contamination.
BASELINE_START_OFFSET_S: float = -5.0
BASELINE_END_OFFSET_S: float = -2.0

# Percentile for robust temporal aggregation.
# 90th percentile: rejects top 10% (noise artifacts) while capturing
# sustained near-peak engagement. Variance ≈ 0.09 / [n · f(Q_90)²],
# yielding SE ≈ 0.005 at 30fps × 4.5s window = 135 frames.
REWARD_PERCENTILE: float = 90.0

# §7B defines no sparse-window invalidation threshold: compute P90 for any
# non-empty measurement window; an empty window yields P90=0.0.


@dataclass
class TimestampedAU12:
    """
    Represent one bounded AU12 observation for reward computation.

    Accepts a drift-corrected UTC epoch timestamp and a [0, 1] intensity value
    and produces the typed item consumed by window extraction. It does not
    validate face landmarks, compute AU12, or apply semantic gating.
    """

    timestamp_s: float
    intensity: float  # [0.0, 1.0] from AU12Normalizer.compute_bounded_intensity()


@dataclass
class RewardResult:
    """
    Represent the complete §7B reward computation output.

    Contains the gated reward, P90 AU12 intensity, binary semantic gate,
    measurement frame count, and optional pre-stimulus baseline diagnostic
    produced by ``compute_reward``. It does not update posteriors or carry
    observational analytics that are outside the reward path.
    """

    gated_reward: float
    p90_intensity: float
    semantic_gate: int
    n_frames_in_window: int
    au12_baseline_pre: float | None


def _bounded_au12(value: float) -> float:
    """Clamp an AU12 observation to the bounded §7B reward interval."""
    return max(0.0, min(1.0, float(value)))


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

    Used for per-segment diagnostic baseline. The -2.0s end provides a
    buffer before stimulus onset to avoid anticipatory expression contamination.
    This diagnostic is persisted as ``au12_baseline_pre`` and is not used to
    scale ``p90_intensity`` or ``gated_reward``.

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
) -> RewardResult:
    """
    Compute the continuous gated reward for Thompson Sampling update.

    §7B pipeline:
      1. Extract pre-stimulus baseline window → ``au12_baseline_pre`` diagnostic
      2. Extract post-stimulus measurement window
      3. Compute 90th percentile of bounded AU12 intensities in that window
      4. Apply semantic validity gate: r_t = P90 × G_t

    Only bounded post-stimulus AU12 data and the binary ``is_match`` gate can
    affect ``p90_intensity`` and ``gated_reward``.

    If the semantic gate is closed (is_match=False), the reward is
    deterministically crushed to 0.0, preventing false positive posterior
    contamination from contextually unrelated AU12 spikes.

    If the measurement window is empty, P90 is defined as 0.0. Otherwise,
    P90 is computed over whatever bounded AU12 observations are present;
    §7B defines no sparse-window invalidation threshold.

    Args:
        au12_series: Full segment of timestamped bounded AU12 observations.
        stimulus_time_s: Drift-corrected UTC epoch of stimulus injection.
        is_match: Semantic validity gate output from Module D (§8.5).

    Returns:
        Canonical RewardResult with the gated reward and §7B intermediate values.
    """
    baseline_obs = extract_baseline_window(au12_series, stimulus_time_s)
    au12_baseline_pre: float | None = None
    if baseline_obs:
        au12_baseline_pre = float(np.mean([_bounded_au12(obs.intensity) for obs in baseline_obs]))

    window_obs = extract_stimulus_window(au12_series, stimulus_time_s)
    n_frames_in_window = len(window_obs)

    bounded_intensities = [_bounded_au12(obs.intensity) for obs in window_obs]
    p90 = 0.0 if n_frames_in_window == 0 else compute_p90(bounded_intensities)

    gate: int = 1 if is_match else 0
    gated_reward: float = p90 * gate
    gated_reward = max(0.0, min(1.0, gated_reward))

    return RewardResult(
        gated_reward=gated_reward,
        p90_intensity=p90,
        semantic_gate=gate,
        n_frames_in_window=n_frames_in_window,
        au12_baseline_pre=au12_baseline_pre,
    )
