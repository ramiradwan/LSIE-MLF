"""
Tests for packages/ml_core/au12.py — Phase 0 validation.

Verifies AU12Normalizer against §7 mathematical specifications:
landmark extraction, IOD derivation, baseline calibration, scoring,
epsilon guard, and 5.0 hard-clamp.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.ml_core.au12 import AU12Normalizer
from tests.conftest import LandmarkArray


class TestAU12Normalizer:
    """§7 — AU12 intensity computation."""

    def test_calibration_returns_zero(self, neutral_landmarks: LandmarkArray) -> None:
        """§7.4 — During calibration, compute_intensity returns 0.0."""
        norm = AU12Normalizer()
        result = norm.compute_intensity(neutral_landmarks, is_calibrating=True)
        assert result == 0.0

    def test_calibration_sets_baseline(self, neutral_landmarks: LandmarkArray) -> None:
        """§7.4 — Calibration accumulates buffer and sets b_neutral."""
        norm = AU12Normalizer()
        for _ in range(10):
            norm.compute_intensity(neutral_landmarks, is_calibrating=True)
        assert norm.b_neutral is not None
        assert norm.b_neutral > 0.0

    def test_inference_without_calibration_raises(self, neutral_landmarks: LandmarkArray) -> None:
        """§7.5 — ValueError if baseline not calibrated."""
        norm = AU12Normalizer()
        with pytest.raises(ValueError, match="Baseline not calibrated"):
            norm.compute_intensity(neutral_landmarks, is_calibrating=False)

    def test_smiling_scores_higher(
        self, neutral_landmarks: LandmarkArray, smiling_landmarks: LandmarkArray
    ) -> None:
        """§7.4 — Wider mouth yields higher AU12 score."""
        norm = AU12Normalizer()
        for _ in range(10):
            norm.compute_intensity(neutral_landmarks, is_calibrating=True)
        neutral_score = norm.compute_intensity(neutral_landmarks)
        smile_score = norm.compute_intensity(smiling_landmarks)
        assert smile_score > neutral_score

    def test_output_clamped_to_five(self) -> None:
        """§7.5 — AU12 output hard-clamped to 5.0."""
        norm = AU12Normalizer(alpha=5.0)
        norm.b_neutral = 0.0
        norm.calibration_buffer = [0.0]
        # Construct landmarks with extreme mouth width
        lm = np.zeros((478, 3), dtype=np.float64)
        lm[33] = [0.3, 0.3, 0.0]
        lm[133] = [0.4, 0.3, 0.0]
        lm[362] = [0.6, 0.3, 0.0]
        lm[263] = [0.7, 0.3, 0.0]
        lm[61] = [0.0, 0.6, 0.0]
        lm[291] = [1.0, 0.6, 0.0]
        score = norm.compute_intensity(lm)
        assert score <= 5.0

    def test_epsilon_guard_zero_iod(self) -> None:
        """§7.5 — Returns 0.0 when IOD < epsilon (degenerate face)."""
        norm = AU12Normalizer()
        lm = np.zeros((478, 3), dtype=np.float64)
        # All eye landmarks at same point → IOD ≈ 0
        result = norm.compute_intensity(lm, is_calibrating=True)
        assert result == 0.0
