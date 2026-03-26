"""
AU12 Normalizer — §7 Mathematical Specifications (AU12)

Computes Action Unit 12 (Lip Corner Puller) intensity from MediaPipe
478-vertex 3D facial landmarks. Geometric normalization uses interocular
distance (IOD) as a scale-invariant reference in full 3D Euclidean space.

v2.0 Corrections: landmark indexing via landmarks[i], epsilon guard for
IOD→0, output hard-clamped to 5.0, full type annotations.

v3.0 — Mathematical Recipe Alignment:
  - Default alpha_scale changed from 5.0 → 6.0 (§7.4, FACS-anchored
    derivation: 95th percentile Duchenne deviation ≈0.15 maps to 0.90).
  - Added compute_bounded_intensity() returning [0.0, 1.0] via tanh
    soft-saturation for fractional Beta-Bernoulli Thompson Sampling.
  - Added compute_raw_ratio() exposing the raw D_mouth/IOD ratio for
    per-subject range normalization in the reward pipeline.
  - Original compute_intensity() retained with [0.0, 5.0] clamp for
    backward compatibility with existing dashboard and metrics persistence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

# §7.5 — Epsilon guard to avoid division by zero when IOD approaches zero.
EPSILON: float = 1e-6

# v3.0 — Default alpha_scale derived from FACS-anchored calibration:
#   α_scale = 0.90 / 0.15 = 6.0
#   (95th percentile Duchenne D_mouth/IOD deviation maps to 0.90)
#   Convergence of percentile-based and FACS-intensity-based derivations
#   validates 6.0 as the central estimate (acceptable range: 5.0–8.0).
DEFAULT_ALPHA_SCALE: float = 6.0


class AU12Normalizer:
    """
    §7.4–7.5 — AU12 intensity scorer.

    Calibration phase computes a neutral baseline B_neutral from rolling
    average of D_mouth / IOD. Inference phase returns a FACS score via
    one of two output methods:

      - compute_intensity(): returns [0.0, 5.0] (hard-clamp, backward compat)
      - compute_bounded_intensity(): returns [0.0, 1.0] (tanh soft-saturation,
        for fractional Beta-Bernoulli Thompson Sampling reward pipeline)

    Args:
        alpha: Empirical linear projection multiplier (default 6.0).
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA_SCALE) -> None:
        self.alpha: float = alpha
        self.b_neutral: float | None = None
        self.calibration_buffer: list[float] = []

    def _extract_geometry(
        self, landmarks: npt.NDArray[np.floating[Any]]
    ) -> tuple[float, float] | None:
        """
        Extract IOD and D_mouth from a (478, 3) landmark array.

        §7.2 — Landmark extraction
        §7.3 — IOD derivation
        §7.4 — D_mouth computation

        Returns:
            (iod, d_mouth) tuple, or None if IOD < epsilon (degenerate face).
        """
        # §7.2 — Landmark extraction: eye corners and lip corners
        left_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[33]
        left_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[133]
        right_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[362]
        right_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[263]
        left_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[61]
        right_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[291]

        # §7.3 — IOD derivation: 3D Euclidean distance between eye centers
        left_eye_center: npt.NDArray[np.floating[Any]] = (left_eye_outer + left_eye_inner) / 2.0
        right_eye_center: npt.NDArray[np.floating[Any]] = (right_eye_inner + right_eye_outer) / 2.0
        iod: float = float(np.linalg.norm(right_eye_center - left_eye_center))

        # §7.5 — Epsilon guard: degenerate face where IOD → 0
        if iod < EPSILON:
            return None

        # §7.4 — D_mouth: 3D Euclidean distance between lip corners
        d_mouth: float = float(np.linalg.norm(right_lip_corner - left_lip_corner))
        return (iod, d_mouth)

    def _update_calibration(self, ratio: float) -> None:
        """§7.4 — Accumulate baseline buffer during calibration."""
        self.calibration_buffer.append(ratio)
        self.b_neutral = float(np.mean(self.calibration_buffer))

    def compute_raw_ratio(self, landmarks: npt.NDArray[np.floating[Any]]) -> float | None:
        """
        Compute the raw D_mouth/IOD ratio without baseline subtraction.

        Exposed for per-subject range normalization in the reward pipeline.
        The reward module uses this to estimate x_max during calibration.

        §7.3 — IOD derivation
        §7.4 — D_mouth / IOD ratio

        Returns:
            Raw ratio, or None if face is degenerate (IOD < epsilon).
        """
        geom = self._extract_geometry(landmarks)
        if geom is None:
            return None
        iod, d_mouth = geom
        return d_mouth / iod

    def compute_intensity(
        self,
        landmarks: npt.NDArray[np.floating[Any]],
        is_calibrating: bool = False,
    ) -> float:
        """
        Compute AU12 intensity from a (478, 3) landmark array.

        §7.2 — Landmark extraction
        §7.3 — IOD derivation
        §7.4 — Distance, baseline calibration, and scoring

        Args:
            landmarks: MediaPipe Face Mesh output, shape (478, 3).
            is_calibrating: If True, accumulate baseline and return 0.0.

        Returns:
            AU12 FACS score clamped to [0.0, 5.0].

        Raises:
            ValueError: If baseline has not been calibrated before inference.
        """
        geom = self._extract_geometry(landmarks)
        if geom is None:
            return 0.0

        iod, d_mouth = geom
        ratio: float = d_mouth / iod

        if is_calibrating:
            self._update_calibration(ratio)
            return 0.0

        # §7.5 — Inference requires a calibrated baseline
        if self.b_neutral is None:
            raise ValueError("Baseline not calibrated")

        # §7.4 — FACS score: linear projection from baseline deviation
        score: float = self.alpha * (ratio - self.b_neutral)

        # §7.5 — Hard-clamp to [0.0, 5.0] (backward compatible)
        return float(min(max(score, 0.0), 5.0))

    def compute_bounded_intensity(
        self,
        landmarks: npt.NDArray[np.floating[Any]],
        is_calibrating: bool = False,
    ) -> float:
        """
        Compute AU12 intensity bounded to [0.0, 1.0] via tanh soft-saturation.

        v3.0 — This is the output method for the fractional Beta-Bernoulli
        Thompson Sampling reward pipeline. The tanh function provides a
        continuous gradient at the boundaries, avoiding the hard discontinuity
        of min/max clamping that would corrupt posterior updates near 0 or 1.

        Mathematical derivation (Formalizing TS for LSIE-MLF v2.0, §Geometric
        Calibration):
            raw_deviation = D_mouth/IOD - B_neutral
            bounded_score = tanh(α_scale × max(0, raw_deviation))

        The max(0, ·) ensures negative deviations (mouth narrower than
        baseline) map to 0.0. The tanh maps [0, +∞) → [0, 1), with
        α_scale = 6.0 placing a Duchenne smile (deviation ≈ 0.15) at
        tanh(6.0 × 0.15) = tanh(0.9) ≈ 0.72, and laughter (deviation ≈ 0.20)
        at tanh(1.2) ≈ 0.83.

        Args:
            landmarks: MediaPipe Face Mesh output, shape (478, 3).
            is_calibrating: If True, accumulate baseline and return 0.0.

        Returns:
            AU12 intensity in [0.0, 1.0].

        Raises:
            ValueError: If baseline has not been calibrated before inference.
        """
        geom = self._extract_geometry(landmarks)
        if geom is None:
            return 0.0

        iod, d_mouth = geom
        ratio: float = d_mouth / iod

        if is_calibrating:
            self._update_calibration(ratio)
            return 0.0

        if self.b_neutral is None:
            raise ValueError("Baseline not calibrated")

        # v3.0 — Baseline-subtracted deviation, floored at zero
        deviation: float = max(0.0, ratio - self.b_neutral)

        # v3.0 — tanh soft-saturation: continuous gradient, maps to [0, 1)
        return float(math.tanh(self.alpha * deviation))
