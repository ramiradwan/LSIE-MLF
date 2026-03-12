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
        # TODO: Implement per §7.5 Python reference implementation
        raise NotImplementedError
