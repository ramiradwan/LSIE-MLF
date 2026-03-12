"""
Shared test fixtures for LSIE-MLF test suite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pytest


@pytest.fixture
def sample_session_id() -> str:
    """Valid UUID v4 session identifier."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_timestamp() -> datetime:
    """UTC timestamp for test payloads."""
    return datetime.now(UTC)


@pytest.fixture
def sample_landmarks() -> np.ndarray:
    """
    Synthetic 478-vertex 3D landmark array.
    Realistic enough for AU12 computation testing.
    """
    rng = np.random.default_rng(seed=42)
    landmarks = rng.uniform(0.0, 1.0, size=(478, 3)).astype(np.float64)

    # Set eye landmarks to plausible positions for IOD computation
    landmarks[33] = [0.3, 0.3, 0.0]  # left eye outer
    landmarks[133] = [0.4, 0.3, 0.0]  # left eye inner
    landmarks[362] = [0.6, 0.3, 0.0]  # right eye inner
    landmarks[263] = [0.7, 0.3, 0.0]  # right eye outer

    # Set mouth corners for AU12
    landmarks[61] = [0.35, 0.6, 0.0]  # left lip corner
    landmarks[291] = [0.65, 0.6, 0.0]  # right lip corner

    return landmarks


@pytest.fixture
def neutral_landmarks(sample_landmarks: np.ndarray) -> np.ndarray:
    """Landmarks representing a neutral (non-smiling) face."""
    return sample_landmarks.copy()


@pytest.fixture
def smiling_landmarks(sample_landmarks: np.ndarray) -> np.ndarray:
    """Landmarks representing a smiling face (wider mouth)."""
    lm = sample_landmarks.copy()
    lm[61] = [0.25, 0.55, 0.0]  # left lip pulled wider and up
    lm[291] = [0.75, 0.55, 0.0]  # right lip pulled wider and up
    return lm
