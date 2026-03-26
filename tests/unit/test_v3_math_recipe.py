"""
Tests for v3.0 Mathematical Recipe Alignment.

Validates:
  - AU12Normalizer.compute_bounded_intensity() — tanh [0,1] output
  - AU12Normalizer.compute_raw_ratio() — exposed for range normalization
  - Backward compatibility of compute_intensity() [0,5] clamp
  - Default alpha_scale = 6.0
  - Reward pipeline: stimulus windowing, P90 aggregation, semantic gate
  - ThompsonSamplingEngine fractional update

Spec references:
  §7.4 — AU12 scoring and calibration
  §7.5 — epsilon guard and bounds
  §4.E.1 — Thompson Sampling fractional Beta-Bernoulli update
  §8.2 — Semantic validity gate
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

# ─────────────────────────────────────────────────────────────────────
# AU12 Normalizer v3.0 Tests
# ─────────────────────────────────────────────────────────────────────
from packages.ml_core.au12 import DEFAULT_ALPHA_SCALE, AU12Normalizer


def _make_landmarks(
    mouth_width: float = 0.3,
    eye_separation: float = 0.4,
) -> npt.NDArray[np.float64]:
    """Build a synthetic (478, 3) landmark array with controlled geometry."""
    lm = np.zeros((478, 3), dtype=np.float64)
    # Eye landmarks (horizontally centered)
    center = 0.5
    half_eye = eye_separation / 2.0
    lm[33] = [center - half_eye - 0.05, 0.3, 0.0]  # left eye outer
    lm[133] = [center - half_eye + 0.05, 0.3, 0.0]  # left eye inner
    lm[362] = [center + half_eye - 0.05, 0.3, 0.0]  # right eye inner
    lm[263] = [center + half_eye + 0.05, 0.3, 0.0]  # right eye outer
    # Mouth corners (horizontally centered)
    half_mouth = mouth_width / 2.0
    lm[61] = [center - half_mouth, 0.6, 0.0]  # left lip
    lm[291] = [center + half_mouth, 0.6, 0.0]  # right lip
    return lm


class TestAU12NormalizerV3:
    """v3.0 — Bounded intensity and updated defaults."""

    def test_default_alpha_is_six(self) -> None:
        """v3.0 — Default alpha_scale changed from 5.0 to 6.0."""
        norm = AU12Normalizer()
        assert norm.alpha == 6.0
        assert DEFAULT_ALPHA_SCALE == 6.0

    def test_bounded_intensity_returns_unit_interval(self) -> None:
        """v3.0 — compute_bounded_intensity returns [0.0, 1.0] via tanh."""
        norm = AU12Normalizer()
        neutral = _make_landmarks(mouth_width=0.30)
        # Calibrate
        for _ in range(10):
            norm.compute_bounded_intensity(neutral, is_calibrating=True)

        # Neutral face → near 0
        score_neutral = norm.compute_bounded_intensity(neutral)
        assert 0.0 <= score_neutral <= 0.05  # Should be very close to 0

        # Smiling face → positive, bounded by 1.0
        smile = _make_landmarks(mouth_width=0.40)
        score_smile = norm.compute_bounded_intensity(smile)
        assert 0.0 < score_smile < 1.0
        assert score_smile > score_neutral

    def test_bounded_intensity_tanh_saturation(self) -> None:
        """v3.0 — Extreme mouth width saturates near 1.0 via tanh, never exceeds."""
        norm = AU12Normalizer()
        neutral = _make_landmarks(mouth_width=0.30)
        for _ in range(10):
            norm.compute_bounded_intensity(neutral, is_calibrating=True)

        extreme = _make_landmarks(mouth_width=1.0)
        score = norm.compute_bounded_intensity(extreme)
        assert 0.9 < score <= 1.0  # tanh saturates near 1.0

    def test_bounded_intensity_never_negative(self) -> None:
        """v3.0 — Mouth narrower than baseline returns 0.0 (max(0, deviation))."""
        norm = AU12Normalizer()
        neutral = _make_landmarks(mouth_width=0.30)
        for _ in range(10):
            norm.compute_bounded_intensity(neutral, is_calibrating=True)

        narrower = _make_landmarks(mouth_width=0.20)
        score = norm.compute_bounded_intensity(narrower)
        assert score == 0.0

    def test_compute_raw_ratio_returns_positive(self) -> None:
        """v3.0 — compute_raw_ratio exposes D_mouth/IOD without baseline."""
        norm = AU12Normalizer()
        lm = _make_landmarks(mouth_width=0.30, eye_separation=0.40)
        ratio = norm.compute_raw_ratio(lm)
        assert ratio is not None
        assert ratio > 0.0

    def test_compute_raw_ratio_degenerate_returns_none(self) -> None:
        """v3.0 — compute_raw_ratio returns None for degenerate face."""
        norm = AU12Normalizer()
        lm = np.zeros((478, 3), dtype=np.float64)
        ratio = norm.compute_raw_ratio(lm)
        assert ratio is None

    def test_backward_compat_compute_intensity_clamps_five(self) -> None:
        """Backward compat — compute_intensity still clamps to [0, 5]."""
        norm = AU12Normalizer(alpha=6.0)
        norm.b_neutral = 0.0
        norm.calibration_buffer = [0.0]
        extreme = _make_landmarks(mouth_width=1.0)
        score = norm.compute_intensity(extreme)
        assert score <= 5.0

    def test_calibration_returns_zero_bounded(self) -> None:
        """§7.4 — Calibration returns 0.0 for both output methods."""
        norm = AU12Normalizer()
        lm = _make_landmarks()
        assert norm.compute_bounded_intensity(lm, is_calibrating=True) == 0.0

    def test_inference_without_calibration_raises_bounded(self) -> None:
        """§7.5 — ValueError if baseline not calibrated."""
        norm = AU12Normalizer()
        lm = _make_landmarks()
        with pytest.raises(ValueError, match="Baseline not calibrated"):
            norm.compute_bounded_intensity(lm)


# ─────────────────────────────────────────────────────────────────────
# Reward Pipeline Tests
# ─────────────────────────────────────────────────────────────────────

from services.worker.pipeline.reward import (  # noqa: E402
    TimestampedAU12,
    compute_p90,
    compute_reward,
    extract_baseline_window,
    extract_stimulus_window,
)


class TestStimulusWindowing:
    """Stimulus-locked measurement window extraction."""

    def _make_series(
        self, start: float, end: float, fps: float = 30.0, intensity: float = 0.5
    ) -> list[TimestampedAU12]:
        """Generate evenly-spaced AU12 observations."""
        n = int((end - start) * fps)
        return [
            TimestampedAU12(
                timestamp_s=start + i / fps,
                intensity=intensity,
            )
            for i in range(n)
        ]

    def test_extracts_correct_window(self) -> None:
        """Window [+0.5s, +5.0s] relative to stimulus onset."""
        # 30 seconds of data, stimulus at t=15.0
        series = self._make_series(0.0, 30.0)
        window = extract_stimulus_window(series, stimulus_time_s=15.0)
        for obs in window:
            assert 15.5 <= obs.timestamp_s <= 20.0

    def test_baseline_window_precedes_stimulus(self) -> None:
        """Baseline window [-5.0s, -2.0s] relative to stimulus onset."""
        series = self._make_series(0.0, 30.0)
        baseline = extract_baseline_window(series, stimulus_time_s=15.0)
        for obs in baseline:
            assert 10.0 <= obs.timestamp_s <= 13.0

    def test_late_stimulus_yields_fewer_frames(self) -> None:
        """Stimulus at t=28.0 in a 30s segment yields only 2s of window data."""
        series = self._make_series(0.0, 30.0)
        window = extract_stimulus_window(series, stimulus_time_s=28.0)
        # Window is [28.5, 33.0] but data only goes to 30.0
        assert len(window) < 135  # Less than full 4.5s × 30fps


class TestP90Aggregation:
    """90th percentile robust aggregation."""

    def test_p90_known_distribution(self) -> None:
        """P90 of uniform [0, 1] data with 1000 samples ≈ 0.90."""
        np.random.seed(42)
        values = np.random.uniform(0.0, 1.0, 1000).tolist()
        p90 = compute_p90(values)
        assert 0.85 < p90 < 0.95

    def test_p90_ignores_single_spike(self) -> None:
        """P90 robust to single-frame noise spike at 1.0."""
        values = [0.3] * 99 + [1.0]
        p90 = compute_p90(values)
        assert p90 < 0.5

    def test_p90_captures_sustained_peak(self) -> None:
        """P90 captures sustained high values (>10% of frames)."""
        values = [0.1] * 70 + [0.8] * 30
        p90 = compute_p90(values)
        assert p90 >= 0.8


class TestComputeReward:
    """Full reward pipeline integration."""

    def _make_encounter(
        self,
        stimulus_time: float = 15.0,
        pre_stim_intensity: float = 0.1,
        post_stim_intensity: float = 0.7,
        fps: float = 30.0,
    ) -> list[TimestampedAU12]:
        """Build a realistic encounter with baseline + response."""
        series: list[TimestampedAU12] = []
        # Pre-stimulus baseline (0 to stimulus_time)
        for i in range(int(stimulus_time * fps)):
            series.append(
                TimestampedAU12(
                    timestamp_s=i / fps,
                    intensity=pre_stim_intensity,
                )
            )
        # Post-stimulus response (stimulus_time to stimulus_time + 10s)
        for i in range(int(10.0 * fps)):
            t = stimulus_time + i / fps
            series.append(TimestampedAU12(timestamp_s=t, intensity=post_stim_intensity))
        return series

    def test_semantic_gate_crushes_reward(self) -> None:
        """is_match=False → gated_reward = 0.0 regardless of AU12."""
        series = self._make_encounter(post_stim_intensity=0.9)
        result = compute_reward(series, stimulus_time_s=15.0, is_match=False)
        assert result.gated_reward == 0.0
        assert result.semantic_gate == 0
        assert result.p90_intensity > 0.0

    def test_valid_encounter_produces_positive_reward(self) -> None:
        """is_match=True with strong smile → positive gated reward."""
        series = self._make_encounter(post_stim_intensity=0.7)
        result = compute_reward(series, stimulus_time_s=15.0, is_match=True)
        assert result.gated_reward > 0.0
        assert result.is_valid is True
        assert result.semantic_gate == 1

    def test_reward_bounded_zero_one(self) -> None:
        """Gated reward is always in [0.0, 1.0]."""
        series = self._make_encounter(post_stim_intensity=0.95)
        result = compute_reward(series, stimulus_time_s=15.0, is_match=True)
        assert 0.0 <= result.gated_reward <= 1.0

    def test_insufficient_frames_returns_invalid(self) -> None:
        """Too few frames in window → is_valid=False, reward=0.0."""
        series = [
            TimestampedAU12(timestamp_s=15.6, intensity=0.8),
            TimestampedAU12(timestamp_s=15.7, intensity=0.9),
        ]
        result = compute_reward(series, stimulus_time_s=15.0, is_match=True)
        assert result.is_valid is False
        assert result.gated_reward == 0.0

    def test_range_normalization_with_x_max(self) -> None:
        """Per-subject range normalization maps to subject's capacity."""
        series = self._make_encounter(
            pre_stim_intensity=0.1,
            post_stim_intensity=0.5,
        )
        result_raw = compute_reward(series, stimulus_time_s=15.0, is_match=True, x_max=None)
        result_norm = compute_reward(series, stimulus_time_s=15.0, is_match=True, x_max=0.6)
        assert result_norm.gated_reward > result_raw.gated_reward

    def test_reward_result_stores_series(self) -> None:
        """RewardResult includes AU12 window series for traceability."""
        series = self._make_encounter(post_stim_intensity=0.6)
        result = compute_reward(series, stimulus_time_s=15.0, is_match=True)
        assert len(result.au12_window_series) > 0
        assert result.n_frames_in_window == len(result.au12_window_series)


# ─────────────────────────────────────────────────────────────────────
# Thompson Sampling Fractional Update Tests
# ─────────────────────────────────────────────────────────────────────

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

_mock_psycopg2 = MagicMock()
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE = 6


class TestFractionalUpdate:
    """§4.E.1 — Fractional Beta-Bernoulli update validation."""

    @pytest.fixture(autouse=True)
    def _patch_psycopg2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "psycopg2", _mock_psycopg2)
        monkeypatch.setitem(sys.modules, "psycopg2.pool", _mock_psycopg2.pool)
        monkeypatch.setitem(sys.modules, "psycopg2.extensions", _mock_psycopg2.extensions)

    def test_fractional_update_high_reward(self) -> None:
        """r_t = 0.85 → α += 0.85, β += 0.15."""
        from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.85)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 5.85, 3.15)

    def test_fractional_update_low_reward(self) -> None:
        """r_t = 0.2 → α += 0.2, β += 0.8."""
        from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 5.0, "beta_param": 3.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.2)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 5.2, 3.8)

    def test_fractional_update_zero_reward(self) -> None:
        """r_t = 0.0 (gated to zero) → α += 0, β += 1."""
        from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 1.0, "beta_param": 1.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=0.0)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 1.0, 2.0)

    def test_fractional_update_max_reward(self) -> None:
        """r_t = 1.0 → α += 1, β += 0."""
        from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 1.0, "beta_param": 1.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        engine.update("exp-1", "arm_a", reward=1.0)
        mock_store.update_experiment_arm.assert_called_once_with("exp-1", "arm_a", 2.0, 1.0)

    def test_posterior_mean_convergence(self) -> None:
        """After many fractional updates, E[α/(α+β)] → true mean."""
        alpha = 1.0
        beta = 1.0
        true_mean = 0.7
        np.random.seed(42)
        for _ in range(100):
            r = np.clip(np.random.normal(true_mean, 0.1), 0.0, 1.0)
            alpha += r
            beta += 1.0 - r

        posterior_mean = alpha / (alpha + beta)
        assert abs(posterior_mean - true_mean) < 0.05

    def test_rejects_out_of_bounds_reward(self) -> None:
        """Reward outside [0, 1] raises ValueError."""
        from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

        mock_store = MagicMock(spec=MetricsStore)
        mock_store.get_experiment_arms.return_value = [
            {"arm": "arm_a", "alpha_param": 1.0, "beta_param": 1.0},
        ]
        engine = ThompsonSamplingEngine(mock_store)
        with pytest.raises(ValueError, match="must be in"):
            engine.update("exp-1", "arm_a", reward=1.5)
        with pytest.raises(ValueError, match="must be in"):
            engine.update("exp-1", "arm_a", reward=-0.1)
