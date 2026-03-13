"""
AU12 Normalizer — §7 Mathematical Specifications (AU12)

Computes Action Unit 12 (Lip Corner Puller) intensity from MediaPipe
478-vertex 3D facial landmarks. Geometric normalization uses interocular
distance (IOD) as a scale-invariant reference in full 3D Euclidean space.

v2.0 Corrections: landmark indexing via landmarks[i], epsilon guard for
IOD→0, output hard-clamped to 5.0, full type annotations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# §7.5 — Epsilon guard to avoid division by zero when IOD approaches zero.
EPSILON: float = 1e-6


class AU12Normalizer:
    """
    §7.4–7.5 — AU12 intensity scorer.

    Calibration phase computes a neutral baseline B_neutral from rolling
    average of D_mouth / IOD. Inference phase returns a FACS score in [0, 5].

    Args:
        alpha: Empirical linear projection multiplier (default 5.0).
    """

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha: float = alpha
        self.b_neutral: float | None = None
        self.calibration_buffer: list[float] = []

    def compute_intensity(
        self, landmarks: npt.NDArray[np.floating[Any]], is_calibrating: bool = False
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
        # §7.2 — Landmark extraction: eye corners and lip corners
        left_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[33]
        left_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[133]
        right_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[362]
        right_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[263]
        left_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[61]
        right_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[291]

        # §7.3 — IOD derivation: 3D Euclidean distance between eye centers
        left_eye_center: npt.NDArray[np.floating[Any]] = (
            left_eye_outer + left_eye_inner
        ) / 2.0
        right_eye_center: npt.NDArray[np.floating[Any]] = (
            right_eye_inner + right_eye_outer
        ) / 2.0
        iod: float = float(np.linalg.norm(right_eye_center - left_eye_center))

        # §7.5 — Epsilon guard: degenerate face where IOD → 0
        if iod < EPSILON:
            return 0.0

        # §7.4 — D_mouth: 3D Euclidean distance between lip corners
        d_mouth: float = float(np.linalg.norm(right_lip_corner - left_lip_corner))
        ratio: float = d_mouth / iod

        if is_calibrating:
            # §7.4 — Accumulate baseline buffer during calibration
            self.calibration_buffer.append(ratio)
            self.b_neutral = float(np.mean(self.calibration_buffer))
            return 0.0

        # §7.5 — Inference requires a calibrated baseline
        if self.b_neutral is None:
            raise ValueError("Baseline not calibrated")

        # §7.4 — FACS score: linear projection from baseline deviation
        score: float = self.alpha * (ratio - self.b_neutral)

        # §7.5 — Hard-clamp to [0.0, 5.0]
        return float(min(max(score, 0.0), 5.0))
