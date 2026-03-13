"""
Tests for packages/ml_core/acoustic.py — Phase 1 validation.

Verifies AcousticAnalyzer against §4.D.3:
pitch F0, jitter, shimmer extraction via parselmouth.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from packages.ml_core.acoustic import AcousticAnalyzer, AcousticMetrics


class TestAcousticMetrics:
    """§11 — Variable Extraction Matrix data class."""

    def test_defaults_are_none(self) -> None:
        m = AcousticMetrics()
        assert m.pitch_f0 is None
        assert m.jitter is None
        assert m.shimmer is None

    def test_with_values(self) -> None:
        m = AcousticMetrics(pitch_f0=150.0, jitter=0.01, shimmer=0.03)
        assert m.pitch_f0 == 150.0
        assert m.jitter == 0.01
        assert m.shimmer == 0.03


@pytest.fixture()
def mock_parselmouth(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install mock parselmouth + parselmouth.praat into sys.modules."""
    mock_pm = MagicMock()
    mock_praat = MagicMock()
    mock_pm.praat = mock_praat

    monkeypatch.setitem(sys.modules, "parselmouth", mock_pm)
    monkeypatch.setitem(sys.modules, "parselmouth.praat", mock_praat)

    return mock_pm, mock_praat


class TestAcousticAnalyzer:
    """§4.D.3 — Praat acoustic feature extraction."""

    def test_analyze_returns_metrics(
        self, mock_parselmouth: tuple[MagicMock, MagicMock]
    ) -> None:
        """§4.D.3 — Returns AcousticMetrics with pitch, jitter, shimmer."""
        mock_pm, mock_praat = mock_parselmouth

        mock_sound = MagicMock()
        mock_pm.Sound.return_value = mock_sound

        mock_pitch = MagicMock()
        mock_point_process = MagicMock()

        def call_side_effect(*args: Any, **kwargs: Any) -> Any:
            cmd = args[1] if len(args) > 1 else ""
            if cmd == "To Pitch":
                return mock_pitch
            if cmd == "Get mean":
                return 180.0
            if cmd == "To PointProcess (periodic, cc)":
                return mock_point_process
            if cmd == "Get jitter (local)":
                return 0.012
            if cmd == "Get shimmer (local)":
                return 0.035
            return None

        mock_praat.call.side_effect = call_side_effect

        samples = np.zeros(16000, dtype=np.int16)
        analyzer = AcousticAnalyzer()
        result = analyzer.analyze(samples.tobytes(), sample_rate=16000)

        assert isinstance(result, AcousticMetrics)
        assert result.pitch_f0 == 180.0
        assert result.jitter == 0.012
        assert result.shimmer == 0.035

    def test_analyze_zero_pitch_becomes_none(
        self, mock_parselmouth: tuple[MagicMock, MagicMock]
    ) -> None:
        """§4.D.3 — Zero pitch (no voiced frames) returns None."""
        mock_pm, mock_praat = mock_parselmouth
        mock_pm.Sound.return_value = MagicMock()

        def call_side_effect(*args: Any, **kwargs: Any) -> Any:
            cmd = args[1] if len(args) > 1 else ""
            if cmd == "Get mean":
                return 0.0
            if cmd == "Get jitter (local)":
                return None
            if cmd == "Get shimmer (local)":
                return None
            return MagicMock()

        mock_praat.call.side_effect = call_side_effect

        samples = np.zeros(16000, dtype=np.int16)
        analyzer = AcousticAnalyzer()
        result = analyzer.analyze(samples.tobytes())

        assert result.pitch_f0 is None

    def test_analyze_uses_correct_sample_rate(
        self, mock_parselmouth: tuple[MagicMock, MagicMock]
    ) -> None:
        """§4.D.3 — Sound object created with specified sample rate."""
        mock_pm, mock_praat = mock_parselmouth
        mock_praat.call.return_value = MagicMock()

        samples = np.zeros(8000, dtype=np.int16)
        analyzer = AcousticAnalyzer()
        analyzer.analyze(samples.tobytes(), sample_rate=8000)

        mock_pm.Sound.assert_called_once()
        call_kwargs = mock_pm.Sound.call_args[1]
        assert call_kwargs["sampling_frequency"] == 8000
