"""
Tests for packages/ml_core/face_mesh.py — Phase 1 validation.

Verifies FaceMeshProcessor against §4.D.2:
landmark extraction, (478, 3) shape, None on missing face.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from packages.ml_core.face_mesh import FaceMeshProcessor


def _make_mock_landmark(x: float, y: float, z: float) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


@pytest.fixture()
def mock_mediapipe(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock mediapipe into sys.modules."""
    mock_mp = MagicMock()
    monkeypatch.setitem(sys.modules, "mediapipe", mock_mp)
    monkeypatch.setitem(sys.modules, "mediapipe.solutions", mock_mp.solutions)
    monkeypatch.setitem(sys.modules, "mediapipe.solutions.face_mesh", mock_mp.solutions.face_mesh)
    return mock_mp


@pytest.fixture()
def mock_cv2(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock cv2 into sys.modules."""
    mock = MagicMock()
    mock.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", mock)
    return mock


class TestFaceMeshProcessor:
    """§4.D.2 — MediaPipe Face Mesh inference."""

    def test_extract_landmarks_returns_478x3(
        self, mock_mediapipe: MagicMock, mock_cv2: MagicMock
    ) -> None:
        """§4.D.2 — Returns (478, 3) ndarray on face detection."""
        fake_landmarks = [_make_mock_landmark(0.5, 0.5, 0.0) for _ in range(478)]
        mock_face = MagicMock()
        mock_face.landmark = fake_landmarks

        mock_result = MagicMock()
        mock_result.multi_face_landmarks = [mock_face]

        mock_mesh_instance = MagicMock()
        mock_mesh_instance.process.return_value = mock_result
        mock_mediapipe.solutions.face_mesh.FaceMesh.return_value = mock_mesh_instance

        processor = FaceMeshProcessor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.extract_landmarks(frame)

        assert result is not None
        assert result.shape == (478, 3)
        assert result.dtype == np.float64

    def test_extract_landmarks_no_face_returns_none(
        self, mock_mediapipe: MagicMock, mock_cv2: MagicMock
    ) -> None:
        """§4.D contract — Missing face returns None."""
        mock_result = MagicMock()
        mock_result.multi_face_landmarks = None

        mock_mesh_instance = MagicMock()
        mock_mesh_instance.process.return_value = mock_result
        mock_mediapipe.solutions.face_mesh.FaceMesh.return_value = mock_mesh_instance

        processor = FaceMeshProcessor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.extract_landmarks(frame)

        assert result is None

    def test_load_model_initializes_face_mesh(self, mock_mediapipe: MagicMock) -> None:
        """§4.D.2 — load_model creates MediaPipe FaceMesh with refine_landmarks."""
        processor = FaceMeshProcessor()
        processor.load_model()

        mock_mediapipe.solutions.face_mesh.FaceMesh.assert_called_once_with(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
