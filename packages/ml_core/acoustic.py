"""
Acoustic Analysis — §4.D.3

Praat engine accessed through parselmouth for pitch (F0),
jitter, and shimmer extraction from speech audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass


@dataclass
class AcousticMetrics:
    """Acoustic feature vector per §11 Variable Extraction Matrix."""

    pitch_f0: float | None = None  # 75–600 Hz
    jitter: float | None = None  # 0–1 ratio
    shimmer: float | None = None  # 0–1 ratio


class AcousticAnalyzer:
    """
    §4.D.3 — Praat acoustic feature extraction via parselmouth.

    Extracts pitch, jitter, and shimmer metrics from speech audio segments.
    """

    def analyze(self, audio_samples: bytes, sample_rate: int = 16000) -> AcousticMetrics:
        """
        Extract acoustic features from PCM audio.

        §4.D.3 — Praat acoustic feature extraction via parselmouth.

        Args:
            audio_samples: Raw PCM s16le bytes.
            sample_rate: Sample rate in Hz (16000 after resampling).

        Returns:
            AcousticMetrics with pitch, jitter, shimmer values.
        """
        import parselmouth as pm
        from parselmouth.praat import call

        # Convert PCM s16le bytes to float64 samples
        samples: npt.NDArray[np.float64] = np.frombuffer(
            audio_samples, dtype=np.int16,
        ).astype(np.float64)
        # Normalize to [-1.0, 1.0] range
        samples = samples / 32768.0

        # §4.D.3 — Create Praat Sound object
        sound: pm.Sound = pm.Sound(samples, sampling_frequency=sample_rate)

        # §4.D.3 — Pitch extraction (F0): 75–600 Hz range
        pitch: pm.Pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
        pitch_f0: float | None = call(pitch, "Get mean", 0.0, 0.0, "Hertz")
        if pitch_f0 == 0.0:
            pitch_f0 = None  # No voiced frames detected

        # §4.D.3 — Jitter and shimmer via PointProcess
        point_process: Any = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)

        jitter: float | None = call(
            point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
        )
        shimmer: float | None = call(
            [sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )

        return AcousticMetrics(pitch_f0=pitch_f0, jitter=jitter, shimmer=shimmer)
