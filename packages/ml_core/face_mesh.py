"""
Face Mesh — §4.D.2 Computer Vision

MediaPipe Face Mesh generating a 478-vertex 3D landmark mesh.
Uses Procrustes Analysis to construct a normalized metric face space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class FaceMeshProcessor:
    """
    §4.D.2 — MediaPipe Face Mesh inference.

    Processes raw video frames and returns 478-vertex 3D landmark arrays.
    Missing face returns null facial metrics (§4.D contract failure mode).
    """

    def __init__(self) -> None:
        self._face_mesh: Any = None  # Lazy-loaded MediaPipe FaceMesh instance

    def load_model(self) -> None:
        """Initialize MediaPipe Face Mesh solution (§4.D.2)."""
        import mediapipe as mp

        # §4.D.2 — 478-vertex 3D landmark mesh, static image mode off for video
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # 478 landmarks (includes iris)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_landmarks(
        self, frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.floating[Any]] | None:
        """
        Extract 478 3D landmarks from a single video frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            np.ndarray of shape (478, 3) with normalized coordinates,
            or None if no face detected (§4.D contract failure mode).
        """
        import cv2

        if self._face_mesh is None:
            self.load_model()

        # §4.D.2 — MediaPipe expects RGB input
        rgb_frame: npt.NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results: Any = self._face_mesh.process(rgb_frame)

        # §4.D contract — missing face returns None (null facial metrics)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        # §4.D.2 — Extract (478, 3) normalized coordinate array
        landmarks: npt.NDArray[np.floating[Any]] = np.array(
            [[lm.x, lm.y, lm.z] for lm in face.landmark],
            dtype=np.float64,
        )
        return landmarks
