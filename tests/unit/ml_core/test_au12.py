"""
Tests for packages/ml_core/au12.py — canonical AU12 validation.

Verifies AU12Normalizer against §7A mathematical specifications: landmark
extraction, IOD derivation, baseline calibration, bounded scoring, and epsilon
guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.ml_core.au12 import AU12Normalizer
from tests.conftest import LandmarkArray


class TestAU12Normalizer:
    """§7A — AU12 bounded intensity computation."""

    def test_calibration_returns_zero(self, neutral_landmarks: LandmarkArray) -> None:
        """§7A.4 — During calibration, bounded intensity returns 0.0."""
        norm = AU12Normalizer()
        result = norm.compute_bounded_intensity(neutral_landmarks, is_calibrating=True)
        assert result == 0.0

    def test_calibration_sets_baseline(self, neutral_landmarks: LandmarkArray) -> None:
        """§7A.4 — Calibration accumulates buffer and sets b_neutral."""
        norm = AU12Normalizer()
        for _ in range(10):
            norm.compute_bounded_intensity(neutral_landmarks, is_calibrating=True)
        assert norm.b_neutral is not None
        assert norm.b_neutral > 0.0

    def test_inference_without_calibration_raises(self, neutral_landmarks: LandmarkArray) -> None:
        """§7A.5 — ValueError if baseline not calibrated."""
        norm = AU12Normalizer()
        with pytest.raises(ValueError, match="Baseline not calibrated"):
            norm.compute_bounded_intensity(neutral_landmarks, is_calibrating=False)

    def test_smiling_scores_higher(
        self, neutral_landmarks: LandmarkArray, smiling_landmarks: LandmarkArray
    ) -> None:
        """§7A.4 — Wider mouth yields higher bounded AU12 score."""
        norm = AU12Normalizer()
        for _ in range(10):
            norm.compute_bounded_intensity(neutral_landmarks, is_calibrating=True)
        neutral_score = norm.compute_bounded_intensity(neutral_landmarks)
        smile_score = norm.compute_bounded_intensity(smiling_landmarks)
        assert smile_score > neutral_score

    def test_output_bounded_to_unit_interval(self) -> None:
        """§7A.5 — AU12 output is bounded to [0.0, 1.0]."""
        norm = AU12Normalizer(alpha=5.0)
        norm.b_neutral = 0.0
        norm.calibration_buffer = [0.0]
        lm = np.zeros((478, 3), dtype=np.float64)
        lm[33] = [0.3, 0.3, 0.0]
        lm[133] = [0.4, 0.3, 0.0]
        lm[362] = [0.6, 0.3, 0.0]
        lm[263] = [0.7, 0.3, 0.0]
        lm[61] = [0.0, 0.6, 0.0]
        lm[291] = [1.0, 0.6, 0.0]
        score = norm.compute_bounded_intensity(lm)
        assert 0.0 <= score <= 1.0

    def test_epsilon_guard_zero_iod(self) -> None:
        """§7A.5 — Returns 0.0 when IOD < epsilon (degenerate face)."""
        norm = AU12Normalizer()
        lm = np.zeros((478, 3), dtype=np.float64)
        result = norm.compute_bounded_intensity(lm, is_calibrating=True)
        assert result == 0.0
