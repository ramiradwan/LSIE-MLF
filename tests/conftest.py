"""
Shared test fixtures for LSIE-MLF test suite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

# Type alias for landmark arrays
LandmarkArray = np.ndarray[Any, np.dtype[np.float64]]


@pytest.fixture
def sample_session_id() -> str:
    """Valid UUID v4 session identifier."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_timestamp() -> datetime:
    """UTC timestamp for test payloads."""
    return datetime.now(UTC)


@pytest.fixture
def sample_landmarks() -> LandmarkArray:
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
def neutral_landmarks(sample_landmarks: LandmarkArray) -> LandmarkArray:
    """Landmarks representing a neutral (non-smiling) face."""
    return sample_landmarks.copy()


@pytest.fixture
def smiling_landmarks(sample_landmarks: LandmarkArray) -> LandmarkArray:
    """Landmarks representing a smiling face (wider mouth)."""
    lm = sample_landmarks.copy()
    lm[61] = [0.25, 0.55, 0.0]  # left lip pulled wider and up
    lm[291] = [0.75, 0.55, 0.0]  # right lip pulled wider and up
    return lm


def pytest_addoption(parser: Any) -> None:
    """Add audit-item test selection for executable §13 verifiers."""
    parser.addoption(
        "--audit-item",
        action="store",
        default=None,
        help="Run only tests marked with pytest.mark.audit_item(<item_id>).",
    )


def pytest_configure(config: Any) -> None:
    """Register the audit_item marker for local plugin consumers."""
    config.addinivalue_line(
        "markers",
        "audit_item(item_id): bind a test to a §13 audit checklist item",
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Deselect tests whose audit_item argument does not match --audit-item."""
    requested_item_id = config.getoption("--audit-item")
    if requested_item_id is None:
        return

    selected: list[Any] = []
    deselected: list[Any] = []
    for item in items:
        has_matching_marker = False
        for marker in item.iter_markers(name="audit_item"):
            positional_item_ids = [str(arg) for arg in marker.args]
            keyword_item_id = marker.kwargs.get("item_id")
            if (
                str(requested_item_id) in positional_item_ids
                or str(keyword_item_id) == requested_item_id
            ):
                has_matching_marker = True
                break
        if has_matching_marker:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
