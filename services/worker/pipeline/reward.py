"""
Reward Computation — Continuous Facial Affect Reward Pipeline

Computes the continuous [0, 1] reward for the fractional Beta-Bernoulli
Thompson Sampling engine from bounded AU12 telemetry arrays.

Mathematical recipe (LSIE-MLF v3.4 §7B):
  1. Stimulus-locked windowing: extract bounded AU12 values in
     [t_stim + 0.5s, t_stim + 5.0s]
  2. 90th percentile aggregation: robust to single-frame noise, captures sustained peak
  3. Baseline diagnostic: record mean bounded AU12 in the pre-stimulus window
     as au12_baseline_pre; do not use it to scale the live reward
  4. Semantic validity gate: r_t = P90 × G_t (G_t ∈ {0, 1})

The final gated reward r_t ∈ [0, 1] feeds directly into the fractional
Beta-Bernoulli update: α += r_t, β += (1 - r_t).

Spec references:
  §7B — Reward pipeline and Thompson Sampling reference implementation
  §4.E.1 — Thompson Sampling reward update
  §8.5 — is_match boolean gate; confidence_score analytical only
  §11.5 — Module E reward variables
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """A single AU12 observation with its timestamp.

    §11 — AU12 Intensity Score from Variable Extraction Matrix.
    Timestamps are drift-corrected UTC epoch seconds (§4.C.1).
    """

    timestamp_s: float
    intensity: float  # [0.0, 1.0] from AU12Normalizer.compute_bounded_intensity()


@dataclass(init=False)
class RewardResult:
    """Complete reward computation result for encounter logging.

    Stores both the final gated reward (for the Thompson Sampling update)
    and the intermediate values (for auditability and retroactive analysis).

    The canonical §7B baseline field is ``au12_baseline_pre``. A legacy
    ``baseline_b_neutral`` property is retained as a read/write alias for
    downstream DTOs that have not yet changed their public API names.
    """

    # Final reward fed to Thompson Sampling: r_t = P90 × G_t
    gated_reward: float

    # Intermediate values for traceability
    p90_intensity: float  # 90th percentile of bounded stimulus-window AU12
    semantic_gate: int  # G_t ∈ {0, 1}
    is_valid: bool  # Legacy telemetry; True when §7B computation completed
    n_frames_in_window: int  # Number of AU12 frames in measurement window
    au12_baseline_pre: float | None  # Mean bounded pre-stimulus AU12 diagnostic

    # Bounded AU12 time series within the measurement window (for logging)
    au12_window_series: list[float] = field(default_factory=list)

    def __init__(
        self,
        gated_reward: float,
        p90_intensity: float,
        semantic_gate: int,
        is_valid: bool,
        n_frames_in_window: int,
        au12_baseline_pre: float | None = None,
        au12_window_series: list[float] | None = None,
        *,
        baseline_b_neutral: float | None = None,
    ) -> None:
        """Initialize a reward result, accepting the legacy baseline keyword.

        ``baseline_b_neutral`` is only a compatibility alias. New code should
        pass and read ``au12_baseline_pre`` to match the §7B/§11.5 name.
        """
        if (
            au12_baseline_pre is not None
            and baseline_b_neutral is not None
            and au12_baseline_pre != baseline_b_neutral
        ):
            raise ValueError("au12_baseline_pre and baseline_b_neutral aliases disagree")

        self.gated_reward = gated_reward
        self.p90_intensity = p90_intensity
        self.semantic_gate = semantic_gate
        self.is_valid = is_valid
        self.n_frames_in_window = n_frames_in_window
        self.au12_baseline_pre = (
            au12_baseline_pre if au12_baseline_pre is not None else baseline_b_neutral
        )
        self.au12_window_series = [] if au12_window_series is None else list(au12_window_series)

    @property
    def baseline_b_neutral(self) -> float | None:
        """Legacy alias for ``au12_baseline_pre``."""
        return self.au12_baseline_pre

    @baseline_b_neutral.setter
    def baseline_b_neutral(self, value: float | None) -> None:
        self.au12_baseline_pre = value


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
    **_ignored_diagnostics: object,
) -> RewardResult:
    """
    Compute the continuous gated reward for Thompson Sampling update.

    §7B pipeline:
      1. Extract pre-stimulus baseline window → ``au12_baseline_pre`` diagnostic
      2. Extract post-stimulus measurement window
      3. Compute 90th percentile of bounded AU12 intensities in that window
      4. Apply semantic validity gate: r_t = P90 × G_t

    The live reward intentionally ignores legacy/raw calibration telemetry
    (for example ``x_max``) and semantic probability diagnostics (for example
    ``confidence_score``). Only bounded post-stimulus AU12 data and the binary
    ``is_match`` gate can affect ``p90_intensity`` and ``gated_reward``.

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
        **_ignored_diagnostics: Deprecated side-channel inputs accepted only
            for backward compatibility; they cannot alter the reward.

    Returns:
        RewardResult with the gated reward and all intermediate values.
    """
    # --- Step 1: Diagnostic baseline from pre-stimulus window ---
    baseline_obs = extract_baseline_window(au12_series, stimulus_time_s)
    au12_baseline_pre: float | None = None
    if baseline_obs:
        au12_baseline_pre = float(np.mean([_bounded_au12(obs.intensity) for obs in baseline_obs]))

    # --- Step 2: Extract stimulus-locked measurement window ---
    window_obs = extract_stimulus_window(au12_series, stimulus_time_s)
    n_frames = len(window_obs)

    # --- Step 3: P90 over bounded post-stimulus AU12 only ---
    bounded_intensities = [_bounded_au12(obs.intensity) for obs in window_obs]
    p90 = 0.0 if n_frames == 0 else compute_p90(bounded_intensities)

    # --- Step 4: Semantic validity gate ---
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
        au12_baseline_pre=au12_baseline_pre,
        au12_window_series=bounded_intensities,
    )
