"""
AU12 bounded intensity utilities for the §7A reward signal.

The module converts MediaPipe 478-landmark frames into scale-normalized
lip-corner motion using interocular distance, maintains a neutral baseline,
and emits tanh-bounded [0, 1] AU12 values for the §7B fractional reward
pipeline. It follows the canonical AU12 names and alpha-scale convention in
§7A.1/§7A.4 and §13.15; it does not perform face detection or posterior
updates.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# §7A.5 — Epsilon guard to avoid division by zero when IOD approaches zero.
EPSILON: float = 1e-6

# §7A.1 — 6.0 maps a Duchenne-scale ratio deviation to the bounded reward range.
DEFAULT_ALPHA_SCALE: float = 6.0


class AU12Normalizer:
    """
    Compute baseline-normalized AU12 intensity from one face landmark stream.

    Accepts MediaPipe Face Mesh landmark arrays shaped (478, 3), with optional
    calibration calls used to build ``B_neutral`` from ``D_mouth / IOD``. Produces
    raw calibration ratios and tanh-bounded intensities in [0.0, 1.0] using the
    §7A.4 alpha-scale mapping (default 6.0; ~0.15 deviation maps to ~0.90 before
    bounding). It does not detect faces, smooth frames, persist calibration state,
    or update Thompson Sampling posteriors.

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

        §7A.2 — Landmark extraction
        §7A.3 — IOD derivation
        §7A.4 — D_mouth computation

        Returns:
            (iod, d_mouth) tuple, or None if IOD < epsilon (degenerate face).
        """
        # §7A.2 — Landmark extraction: eye corners and lip corners
        left_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[33]
        left_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[133]
        right_eye_inner: npt.NDArray[np.floating[Any]] = landmarks[362]
        right_eye_outer: npt.NDArray[np.floating[Any]] = landmarks[263]
        left_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[61]
        right_lip_corner: npt.NDArray[np.floating[Any]] = landmarks[291]

        # §7A.3 — IOD derivation: 3D Euclidean distance between eye centers
        left_eye_center: npt.NDArray[np.floating[Any]] = (left_eye_outer + left_eye_inner) / 2.0
        right_eye_center: npt.NDArray[np.floating[Any]] = (right_eye_inner + right_eye_outer) / 2.0
        iod: float = float(np.linalg.norm(right_eye_center - left_eye_center))

        # §7A.5 — Epsilon guard: degenerate face where IOD → 0
        if iod < EPSILON:
            return None

        # §7A.4 — D_mouth: 3D Euclidean distance between lip corners
        d_mouth: float = float(np.linalg.norm(right_lip_corner - left_lip_corner))
        return (iod, d_mouth)

    def _update_calibration(self, ratio: float) -> None:
        """§7A.4 — Accumulate baseline buffer during calibration."""
        self.calibration_buffer.append(ratio)
        self.b_neutral = float(np.mean(self.calibration_buffer))

    def compute_raw_ratio(self, landmarks: npt.NDArray[np.floating[Any]]) -> float | None:
        """
        Compute the raw D_mouth/IOD ratio without baseline subtraction.

        §7A.3 — IOD derivation
        §7A.4 — D_mouth / IOD ratio

        Returns:
            Raw ratio, or None if face is degenerate (IOD < epsilon).
        """
        geom = self._extract_geometry(landmarks)
        if geom is None:
            return None
        iod, d_mouth = geom
        return d_mouth / iod

    def compute_bounded_intensity(
        self,
        landmarks: npt.NDArray[np.floating[Any]],
        is_calibrating: bool = False,
    ) -> float:
        """
        Compute AU12 intensity bounded to [0.0, 1.0] via tanh soft-saturation.

        Mathematical derivation:
            raw_deviation = D_mouth/IOD - B_neutral
            bounded_score = tanh(α_scale × max(0, raw_deviation))

        Args:
            landmarks: MediaPipe Face Mesh output, shape (478, 3).
            is_calibrating: If True, accumulate baseline and return 0.0.

        Returns:
            AU12 intensity in [0.0, 1.0].

        Raises:
            ValueError: If baseline has not been calibrated before inference.
        """
        started_at = time.perf_counter()
        try:
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

            deviation: float = max(0.0, ratio - self.b_neutral)
            return float(math.tanh(self.alpha * deviation))
        finally:
            logger.debug(
                "BENCHMARK au12_bounded_ms=%.3f calibrating=%s",
                (time.perf_counter() - started_at) * 1000.0,
                is_calibrating,
            )
