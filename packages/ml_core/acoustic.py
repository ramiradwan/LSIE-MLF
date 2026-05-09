"""
Module D acoustic analysis helpers (§4.D.3, §7D).

Provides deterministic stimulus-locked observational acoustic analytics for
Module D. Public helpers compute canonical §7D validity, voiced coverage, F0,
jitter, and shimmer fields from PCM audio or timestamped pitch frames; raw audio
and reconstructive voiceprint data are not persisted (§13.23).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, log2
from typing import Any

import numpy as np
import numpy.typing as npt

ACOUSTIC_VOICING_THRESHOLD: float = 0.45
ACOUSTIC_MIN_VOICED_COVERAGE_S: float = 1.0
ACOUSTIC_MIN_STABLE_ISLAND_S: float = 0.2
ACOUSTIC_MIN_PERIODIC_PEAKS: int = 4

_ACOUSTIC_STIMULUS_WINDOW_START_OFFSET_S: float = 0.5
_ACOUSTIC_STIMULUS_WINDOW_END_OFFSET_S: float = 5.0
_ACOUSTIC_BASELINE_WINDOW_START_OFFSET_S: float = -5.0
_ACOUSTIC_BASELINE_WINDOW_END_OFFSET_S: float = -2.0

_CANONICAL_ACOUSTIC_FIELD_NAMES: tuple[str, ...] = (
    "f0_valid_measure",
    "f0_valid_baseline",
    "perturbation_valid_measure",
    "perturbation_valid_baseline",
    "voiced_coverage_measure_s",
    "voiced_coverage_baseline_s",
    "f0_mean_measure_hz",
    "f0_mean_baseline_hz",
    "f0_delta_semitones",
    "jitter_mean_measure",
    "jitter_mean_baseline",
    "jitter_delta",
    "shimmer_mean_measure",
    "shimmer_mean_baseline",
    "shimmer_delta",
)

__all__ = [
    "ACOUSTIC_VOICING_THRESHOLD",
    "ACOUSTIC_MIN_VOICED_COVERAGE_S",
    "ACOUSTIC_MIN_STABLE_ISLAND_S",
    "ACOUSTIC_MIN_PERIODIC_PEAKS",
    "AcousticAnalyticsResult",
    "AcousticAnalyzer",
    "AcousticMetrics",
    "AcousticPerturbationMeasurement",
    "TimestampedAcousticFrame",
    "compute_f0_delta_semitones",
    "compute_observational_acoustic_result",
    "compute_stimulus_locked_acoustic_result",
    "extract_baseline_acoustic_window",
    "extract_stimulus_acoustic_window",
    "null_acoustic_result",
]

PerturbationMeasurementProvider = Callable[[float, float], "AcousticPerturbationMeasurement | None"]


@dataclass(frozen=True, slots=True)
class AcousticAnalyticsResult:
    """Exact reward-free §7D observational acoustic output schema.

    The result contains only raw validity flags, voiced coverage, voiced-window
    means, and canonical deltas. Reward values, z-scored features, and any
    other population-normalized statistics are intentionally excluded.
    """

    f0_valid_measure: bool = False
    f0_valid_baseline: bool = False
    perturbation_valid_measure: bool = False
    perturbation_valid_baseline: bool = False
    voiced_coverage_measure_s: float = 0.0
    voiced_coverage_baseline_s: float = 0.0
    f0_mean_measure_hz: float | None = None
    f0_mean_baseline_hz: float | None = None
    f0_delta_semitones: float | None = None
    jitter_mean_measure: float | None = None
    jitter_mean_baseline: float | None = None
    jitter_delta: float | None = None
    shimmer_mean_measure: float | None = None
    shimmer_mean_baseline: float | None = None
    shimmer_delta: float | None = None

    def to_dict(self) -> dict[str, bool | float | None]:
        """Serialize the canonical §7D payload with stable field ordering."""
        return {
            field_name: getattr(self, field_name) for field_name in _CANONICAL_ACOUSTIC_FIELD_NAMES
        }


@dataclass(frozen=True, slots=True)
class TimestampedAcousticFrame:
    """A pitch-analysis frame aligned to the stimulus timeline.

    ``timestamp_s`` is the absolute center time of the frame in the same epoch
    space as ``_stimulus_time``. ``duration_s`` is the frame contribution used
    for voiced-coverage and voiced-island duration calculations.
    """

    timestamp_s: float
    duration_s: float
    f0_hz: float | None
    voicing_strength: float


@dataclass(frozen=True, slots=True)
class AcousticPerturbationMeasurement:
    """Praat perturbation outputs for a candidate voiced island."""

    periodic_peak_count: int
    jitter_local: float | None
    shimmer_local: float | None


@dataclass(frozen=True, slots=True)
class _VoicedIsland:
    start_s: float
    end_s: float
    duration_s: float
    frames: tuple[TimestampedAcousticFrame, ...]


@dataclass(frozen=True, slots=True)
class _WindowSummary:
    f0_valid: bool
    perturbation_valid: bool
    voiced_coverage_s: float
    f0_mean_hz: float | None
    jitter_mean: float | None
    shimmer_mean: float | None


def compute_f0_delta_semitones(
    f0_mean_measure_hz: float | None,
    f0_mean_baseline_hz: float | None,
) -> float | None:
    """Compute the canonical §7D semitone delta.

    Returns ``None`` when either dependent F0 mean is null, when the baseline
    F0 is non-positive, or when the ratio is otherwise not mathematically
    computable.
    """
    if f0_mean_measure_hz is None or f0_mean_baseline_hz is None:
        return None
    if f0_mean_baseline_hz <= 0:
        return None

    try:
        return 12.0 * log2(f0_mean_measure_hz / f0_mean_baseline_hz)
    except ValueError:
        return None


def _difference_or_none(measure: float | None, baseline: float | None) -> float | None:
    if measure is None or baseline is None:
        return None
    return measure - baseline


def _is_finite_number(value: float | None) -> bool:
    return value is not None and isfinite(value)


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric_value):
        return None
    return numeric_value


def _compute_effective_perturbation_outputs(
    *,
    f0_valid: bool,
    perturbation_valid: bool,
    jitter_mean: float | None,
    shimmer_mean: float | None,
) -> tuple[bool, float | None, float | None]:
    normalized_jitter_mean = _normalize_optional_float(jitter_mean)
    normalized_shimmer_mean = _normalize_optional_float(shimmer_mean)
    effective_perturbation_valid = (
        f0_valid
        and perturbation_valid
        and normalized_jitter_mean is not None
        and normalized_shimmer_mean is not None
    )
    if not effective_perturbation_valid:
        return False, None, None
    return True, normalized_jitter_mean, normalized_shimmer_mean


def compute_observational_acoustic_result(
    *,
    f0_valid_measure: bool = False,
    f0_valid_baseline: bool = False,
    perturbation_valid_measure: bool = False,
    perturbation_valid_baseline: bool = False,
    voiced_coverage_measure_s: float = 0.0,
    voiced_coverage_baseline_s: float = 0.0,
    f0_mean_measure_hz: float | None = None,
    f0_mean_baseline_hz: float | None = None,
    jitter_mean_measure: float | None = None,
    jitter_mean_baseline: float | None = None,
    shimmer_mean_measure: float | None = None,
    shimmer_mean_baseline: float | None = None,
) -> AcousticAnalyticsResult:
    """Assemble a deterministic §7D result from window-level summary inputs.

    This helper is pure computation only: it normalizes dependent fields to the
    canonical false/null contract, derives canonical deltas, preserves the spec
    independence between F0 validity and perturbation validity, clamps
    impossible perturbation-valid states whenever F0 or perturbation metrics are
    not computable, and never emits reward-path or z-scored values.
    """
    normalized_f0_mean_measure_hz = (
        _normalize_optional_float(f0_mean_measure_hz) if f0_valid_measure else None
    )
    normalized_f0_mean_baseline_hz = (
        _normalize_optional_float(f0_mean_baseline_hz) if f0_valid_baseline else None
    )
    (
        effective_perturbation_valid_measure,
        normalized_jitter_mean_measure,
        normalized_shimmer_mean_measure,
    ) = _compute_effective_perturbation_outputs(
        f0_valid=f0_valid_measure,
        perturbation_valid=perturbation_valid_measure,
        jitter_mean=jitter_mean_measure,
        shimmer_mean=shimmer_mean_measure,
    )
    (
        effective_perturbation_valid_baseline,
        normalized_jitter_mean_baseline,
        normalized_shimmer_mean_baseline,
    ) = _compute_effective_perturbation_outputs(
        f0_valid=f0_valid_baseline,
        perturbation_valid=perturbation_valid_baseline,
        jitter_mean=jitter_mean_baseline,
        shimmer_mean=shimmer_mean_baseline,
    )

    return AcousticAnalyticsResult(
        f0_valid_measure=f0_valid_measure,
        f0_valid_baseline=f0_valid_baseline,
        perturbation_valid_measure=effective_perturbation_valid_measure,
        perturbation_valid_baseline=effective_perturbation_valid_baseline,
        voiced_coverage_measure_s=voiced_coverage_measure_s,
        voiced_coverage_baseline_s=voiced_coverage_baseline_s,
        f0_mean_measure_hz=normalized_f0_mean_measure_hz,
        f0_mean_baseline_hz=normalized_f0_mean_baseline_hz,
        f0_delta_semitones=compute_f0_delta_semitones(
            normalized_f0_mean_measure_hz,
            normalized_f0_mean_baseline_hz,
        ),
        jitter_mean_measure=normalized_jitter_mean_measure,
        jitter_mean_baseline=normalized_jitter_mean_baseline,
        jitter_delta=_difference_or_none(
            normalized_jitter_mean_measure,
            normalized_jitter_mean_baseline,
        ),
        shimmer_mean_measure=normalized_shimmer_mean_measure,
        shimmer_mean_baseline=normalized_shimmer_mean_baseline,
        shimmer_delta=_difference_or_none(
            normalized_shimmer_mean_measure,
            normalized_shimmer_mean_baseline,
        ),
    )


def null_acoustic_result() -> AcousticAnalyticsResult:
    """Return the deterministic §7D false/null acoustic contract."""
    return AcousticAnalyticsResult()


def _extract_acoustic_window(
    frames: Sequence[TimestampedAcousticFrame],
    *,
    stimulus_time_s: float,
    start_offset_s: float,
    end_offset_s: float,
) -> list[TimestampedAcousticFrame]:
    window_start_s = stimulus_time_s + start_offset_s
    window_end_s = stimulus_time_s + end_offset_s
    return [frame for frame in frames if window_start_s <= frame.timestamp_s <= window_end_s]


def extract_stimulus_acoustic_window(
    frames: Sequence[TimestampedAcousticFrame],
    stimulus_time_s: float,
) -> list[TimestampedAcousticFrame]:
    """Return the §7B/§7D measurement window ``[t+0.5s, t+5.0s]``."""
    return _extract_acoustic_window(
        frames,
        stimulus_time_s=stimulus_time_s,
        start_offset_s=_ACOUSTIC_STIMULUS_WINDOW_START_OFFSET_S,
        end_offset_s=_ACOUSTIC_STIMULUS_WINDOW_END_OFFSET_S,
    )


def extract_baseline_acoustic_window(
    frames: Sequence[TimestampedAcousticFrame],
    stimulus_time_s: float,
) -> list[TimestampedAcousticFrame]:
    """Return the §7B/§7D baseline window ``[t-5.0s, t-2.0s]``."""
    return _extract_acoustic_window(
        frames,
        stimulus_time_s=stimulus_time_s,
        start_offset_s=_ACOUSTIC_BASELINE_WINDOW_START_OFFSET_S,
        end_offset_s=_ACOUSTIC_BASELINE_WINDOW_END_OFFSET_S,
    )


def _is_voiced(frame: TimestampedAcousticFrame) -> bool:
    return frame.voicing_strength >= ACOUSTIC_VOICING_THRESHOLD


def _frame_span_start(frame: TimestampedAcousticFrame) -> float:
    return frame.timestamp_s - max(frame.duration_s, 0.0) / 2.0


def _frame_span_end(frame: TimestampedAcousticFrame) -> float:
    return frame.timestamp_s + max(frame.duration_s, 0.0) / 2.0


def _frames_are_contiguous(
    previous: TimestampedAcousticFrame,
    current: TimestampedAcousticFrame,
) -> bool:
    gap_s = current.timestamp_s - previous.timestamp_s
    max_step_s = max(previous.duration_s, current.duration_s, 0.0)
    if max_step_s <= 0.0:
        return False
    return gap_s <= (max_step_s * 1.5)


def _compute_voiced_coverage_s(frames: Sequence[TimestampedAcousticFrame]) -> float:
    return float(sum(max(frame.duration_s, 0.0) for frame in frames if _is_voiced(frame)))


def _build_voiced_islands(frames: Sequence[TimestampedAcousticFrame]) -> list[_VoicedIsland]:
    islands: list[_VoicedIsland] = []
    current_frames: list[TimestampedAcousticFrame] = []

    def flush_current() -> None:
        if not current_frames:
            return
        islands.append(
            _VoicedIsland(
                start_s=_frame_span_start(current_frames[0]),
                end_s=_frame_span_end(current_frames[-1]),
                duration_s=float(sum(max(frame.duration_s, 0.0) for frame in current_frames)),
                frames=tuple(current_frames),
            )
        )
        current_frames.clear()

    for frame in sorted(frames, key=lambda item: item.timestamp_s):
        if not _is_voiced(frame):
            flush_current()
            continue

        if current_frames and not _frames_are_contiguous(current_frames[-1], frame):
            flush_current()
        current_frames.append(frame)

    flush_current()
    return islands


def _compute_f0_mean_hz(
    frames: Sequence[TimestampedAcousticFrame],
    *,
    f0_valid: bool,
) -> float | None:
    if not f0_valid:
        return None

    positive_voiced_f0 = [
        frame.f0_hz
        for frame in frames
        if _is_voiced(frame) and frame.f0_hz is not None and frame.f0_hz > 0.0
    ]
    if not positive_voiced_f0:
        return None
    return float(np.mean(positive_voiced_f0))


def _select_eligible_perturbation_span(
    islands: Sequence[_VoicedIsland],
    *,
    measurement_provider: PerturbationMeasurementProvider | None,
) -> AcousticPerturbationMeasurement | None:
    if measurement_provider is None:
        return None

    ordered_islands = sorted(islands, key=lambda island: (-island.duration_s, island.start_s))
    for island in ordered_islands:
        try:
            measurement = measurement_provider(island.start_s, island.end_s)
        except Exception:
            continue
        if measurement is None:
            continue
        if measurement.periodic_peak_count < ACOUSTIC_MIN_PERIODIC_PEAKS:
            continue
        if not (
            _is_finite_number(measurement.jitter_local)
            and _is_finite_number(measurement.shimmer_local)
        ):
            continue
        return measurement
    return None


def _summarize_window(
    frames: Sequence[TimestampedAcousticFrame],
    *,
    perturbation_measurement_provider: PerturbationMeasurementProvider | None,
) -> _WindowSummary:
    voiced_coverage_s = _compute_voiced_coverage_s(frames)
    voiced_islands = _build_voiced_islands(frames)
    longest_voiced_island_s = max((island.duration_s for island in voiced_islands), default=0.0)

    f0_valid = (
        voiced_coverage_s >= ACOUSTIC_MIN_VOICED_COVERAGE_S
        and longest_voiced_island_s >= ACOUSTIC_MIN_STABLE_ISLAND_S
    )
    f0_mean_hz = _compute_f0_mean_hz(frames, f0_valid=f0_valid)

    perturbation_measurement = None
    if f0_valid:
        perturbation_measurement = _select_eligible_perturbation_span(
            voiced_islands,
            measurement_provider=perturbation_measurement_provider,
        )

    jitter_mean = None
    shimmer_mean = None
    if perturbation_measurement is not None:
        jitter_mean = perturbation_measurement.jitter_local
        shimmer_mean = perturbation_measurement.shimmer_local

    return _WindowSummary(
        f0_valid=f0_valid,
        perturbation_valid=perturbation_measurement is not None,
        voiced_coverage_s=voiced_coverage_s,
        f0_mean_hz=f0_mean_hz,
        jitter_mean=jitter_mean,
        shimmer_mean=shimmer_mean,
    )


def compute_stimulus_locked_acoustic_result(
    frames: Sequence[TimestampedAcousticFrame],
    *,
    stimulus_time_s: float | None,
    perturbation_measurement_provider: PerturbationMeasurementProvider | None = None,
) -> AcousticAnalyticsResult:
    """Compute the canonical §7D result from timestamped pitch-analysis frames.

    The function is deterministic for null stimulus, silence, and unvoiced
    windows. It uses local §7B window boundaries, voiced-only masking,
    independent F0/perturbation validity gates, and canonical summary math.
    """
    if stimulus_time_s is None:
        return null_acoustic_result()

    measure_frames = extract_stimulus_acoustic_window(frames, stimulus_time_s)
    baseline_frames = extract_baseline_acoustic_window(frames, stimulus_time_s)

    measure_summary = _summarize_window(
        measure_frames,
        perturbation_measurement_provider=perturbation_measurement_provider,
    )
    baseline_summary = _summarize_window(
        baseline_frames,
        perturbation_measurement_provider=perturbation_measurement_provider,
    )

    return compute_observational_acoustic_result(
        f0_valid_measure=measure_summary.f0_valid,
        f0_valid_baseline=baseline_summary.f0_valid,
        perturbation_valid_measure=measure_summary.perturbation_valid,
        perturbation_valid_baseline=baseline_summary.perturbation_valid,
        voiced_coverage_measure_s=measure_summary.voiced_coverage_s,
        voiced_coverage_baseline_s=baseline_summary.voiced_coverage_s,
        f0_mean_measure_hz=measure_summary.f0_mean_hz,
        f0_mean_baseline_hz=baseline_summary.f0_mean_hz,
        jitter_mean_measure=measure_summary.jitter_mean,
        jitter_mean_baseline=baseline_summary.jitter_mean,
        shimmer_mean_measure=measure_summary.shimmer_mean,
        shimmer_mean_baseline=baseline_summary.shimmer_mean,
    )


@dataclass
class AcousticMetrics:
    """Canonical §7D acoustic output schema for Module D."""

    f0_valid_measure: bool = False
    f0_valid_baseline: bool = False
    perturbation_valid_measure: bool = False
    perturbation_valid_baseline: bool = False
    voiced_coverage_measure_s: float = 0.0
    voiced_coverage_baseline_s: float = 0.0
    f0_mean_measure_hz: float | None = None
    f0_mean_baseline_hz: float | None = None
    f0_delta_semitones: float | None = None
    jitter_mean_measure: float | None = None
    jitter_mean_baseline: float | None = None
    jitter_delta: float | None = None
    shimmer_mean_measure: float | None = None
    shimmer_mean_baseline: float | None = None
    shimmer_delta: float | None = None

    @classmethod
    def from_observational_result(
        cls,
        result: AcousticAnalyticsResult,
    ) -> AcousticMetrics:
        """Create an AcousticMetrics instance from a canonical §7D result."""
        return cls(
            f0_valid_measure=result.f0_valid_measure,
            f0_valid_baseline=result.f0_valid_baseline,
            perturbation_valid_measure=result.perturbation_valid_measure,
            perturbation_valid_baseline=result.perturbation_valid_baseline,
            voiced_coverage_measure_s=result.voiced_coverage_measure_s,
            voiced_coverage_baseline_s=result.voiced_coverage_baseline_s,
            f0_mean_measure_hz=result.f0_mean_measure_hz,
            f0_mean_baseline_hz=result.f0_mean_baseline_hz,
            f0_delta_semitones=result.f0_delta_semitones,
            jitter_mean_measure=result.jitter_mean_measure,
            jitter_mean_baseline=result.jitter_mean_baseline,
            jitter_delta=result.jitter_delta,
            shimmer_mean_measure=result.shimmer_mean_measure,
            shimmer_mean_baseline=result.shimmer_mean_baseline,
            shimmer_delta=result.shimmer_delta,
        )


class AcousticAnalyzer:
    """
    §4.D.3 / §7D — parselmouth-backed acoustic analyzer for PCM audio.

    ``analyze()`` accepts mono PCM s16le bytes and a sample rate. When both
    ``stimulus_time_s`` and ``segment_start_time_s`` are provided, it returns
    canonical stimulus-locked observational acoustic fields; otherwise it returns
    deterministic false/null §7D defaults. It does not compute rewards or persist
    raw audio or reconstructive voiceprint data (§13.23).
    """

    def _extract_pitch_frame_times(
        self,
        pitch: Any,
        *,
        n_frames: int,
        frame_duration_s: float,
    ) -> npt.NDArray[np.float64]:
        if n_frames <= 0:
            return np.array([], dtype=np.float64)

        raw_times: Any = None
        xs_attr = getattr(pitch, "xs", None)
        if callable(xs_attr):
            raw_times = xs_attr()
        elif xs_attr is not None:
            raw_times = xs_attr

        if raw_times is not None:
            try:
                times = np.asarray(raw_times, dtype=np.float64)
            except (TypeError, ValueError):
                times = np.array([], dtype=np.float64)
            if len(times) >= n_frames:
                return times[:n_frames]

        return np.arange(n_frames, dtype=np.float64) * frame_duration_s + (frame_duration_s / 2.0)

    def _estimate_pitch_frame_duration_s(
        self,
        pitch: Any,
        *,
        frame_times_s: npt.NDArray[np.float64],
        segment_duration_s: float,
        n_frames: int,
    ) -> float:
        get_time_step = getattr(pitch, "get_time_step", None)
        if callable(get_time_step):
            maybe_time_step = _normalize_optional_float(get_time_step())
            if maybe_time_step is not None and maybe_time_step > 0.0:
                return maybe_time_step

        if len(frame_times_s) >= 2:
            diffs = np.diff(frame_times_s)
            positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            if len(positive_diffs) > 0:
                return float(np.median(positive_diffs))

        if segment_duration_s > 0.0 and n_frames > 0:
            return segment_duration_s / n_frames
        return 0.0

    def _extract_pitch_frames(
        self,
        pitch: Any,
        *,
        segment_start_time_s: float,
        segment_duration_s: float,
    ) -> list[TimestampedAcousticFrame]:
        selected_array = getattr(pitch, "selected_array", None)
        if selected_array is None:
            return []

        try:
            raw_frequencies = selected_array["frequency"]
        except (KeyError, TypeError, IndexError):
            return []

        try:
            raw_strengths = selected_array["strength"]
        except (KeyError, TypeError, IndexError):
            raw_strengths = np.zeros_like(raw_frequencies)

        frequencies = np.asarray(raw_frequencies, dtype=np.float64)
        strengths = np.asarray(raw_strengths, dtype=np.float64)
        n_frames = min(len(frequencies), len(strengths))
        if n_frames <= 0:
            return []

        frequencies = frequencies[:n_frames]
        strengths = strengths[:n_frames]
        provisional_duration_s = segment_duration_s / n_frames if segment_duration_s > 0.0 else 0.0
        frame_times_s = self._extract_pitch_frame_times(
            pitch,
            n_frames=n_frames,
            frame_duration_s=provisional_duration_s,
        )
        n_frames = min(n_frames, len(frame_times_s))
        if n_frames <= 0:
            return []

        frequencies = frequencies[:n_frames]
        strengths = strengths[:n_frames]
        frame_times_s = frame_times_s[:n_frames]
        frame_duration_s = self._estimate_pitch_frame_duration_s(
            pitch,
            frame_times_s=frame_times_s,
            segment_duration_s=segment_duration_s,
            n_frames=n_frames,
        )

        frames: list[TimestampedAcousticFrame] = []
        for frame_time_s, frequency_hz, strength in zip(
            frame_times_s,
            frequencies,
            strengths,
            strict=True,
        ):
            frames.append(
                TimestampedAcousticFrame(
                    timestamp_s=segment_start_time_s + float(frame_time_s),
                    duration_s=frame_duration_s,
                    f0_hz=float(frequency_hz),
                    voicing_strength=float(strength),
                )
            )
        return frames

    def _build_perturbation_measurement_provider(
        self,
        *,
        sound: Any,
        point_process: Any,
        praat_call: Callable[..., Any],
        segment_start_time_s: float,
        segment_duration_s: float,
    ) -> PerturbationMeasurementProvider:
        def provider(start_s: float, end_s: float) -> AcousticPerturbationMeasurement | None:
            relative_start_s = max(0.0, start_s - segment_start_time_s)
            relative_end_s = min(segment_duration_s, end_s - segment_start_time_s)
            if relative_end_s <= relative_start_s:
                return None

            first_peak = praat_call(point_process, "Get low index", relative_start_s)
            last_peak = praat_call(point_process, "Get high index", relative_end_s)
            try:
                periodic_peak_count = max(
                    0,
                    int(round(float(last_peak))) - int(round(float(first_peak))) + 1,
                )
            except (TypeError, ValueError):
                return None

            jitter_local = _normalize_optional_float(
                praat_call(
                    point_process,
                    "Get jitter (local)",
                    relative_start_s,
                    relative_end_s,
                    0.0001,
                    0.02,
                    1.3,
                )
            )
            shimmer_local = _normalize_optional_float(
                praat_call(
                    [sound, point_process],
                    "Get shimmer (local)",
                    relative_start_s,
                    relative_end_s,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
            )
            return AcousticPerturbationMeasurement(
                periodic_peak_count=periodic_peak_count,
                jitter_local=jitter_local,
                shimmer_local=shimmer_local,
            )

        return provider

    def analyze_stimulus_locked(
        self,
        audio_samples: bytes,
        *,
        stimulus_time_s: float | None,
        segment_start_time_s: float,
        sample_rate: int = 16000,
    ) -> AcousticMetrics:
        """Explicit stimulus-locked entry point for canonical §7D analytics."""
        return self.analyze(
            audio_samples,
            sample_rate=sample_rate,
            stimulus_time_s=stimulus_time_s,
            segment_start_time_s=segment_start_time_s,
        )

    def analyze(
        self,
        audio_samples: bytes,
        sample_rate: int = 16000,
        *,
        stimulus_time_s: float | None = None,
        segment_start_time_s: float | None = None,
    ) -> AcousticMetrics:
        """
        Extract acoustic features from PCM audio.

        §4.D.3 — Praat acoustic feature extraction via parselmouth.

        Args:
            audio_samples: Raw PCM s16le bytes.
            sample_rate: Sample rate in Hz (16000 after resampling).
            stimulus_time_s: Optional drift-corrected stimulus epoch used for
                canonical §7D windowing.
            segment_start_time_s: Optional absolute segment start epoch in the
                same time base as ``stimulus_time_s``.

        Returns:
            ``AcousticMetrics`` with canonical §7D fields. Fields remain at
            deterministic false/null defaults unless both timing arguments are
            supplied.
        """
        import parselmouth as pm
        from parselmouth.praat import call

        # Convert PCM s16le bytes to float64 samples
        samples: npt.NDArray[np.float64] = np.frombuffer(
            audio_samples,
            dtype=np.int16,
        ).astype(np.float64)
        # Normalize to [-1.0, 1.0] range
        samples = samples / 32768.0
        segment_duration_s = len(samples) / sample_rate if sample_rate > 0 else 0.0

        # §4.D.3 — Create Praat Sound object
        sound: pm.Sound = pm.Sound(samples, sampling_frequency=sample_rate)

        # §4.D.3 — Pitch extraction for canonical voiced-window F0 frames.
        pitch: pm.Pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
        point_process: Any = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)

        observational_result = null_acoustic_result()
        if stimulus_time_s is not None and segment_start_time_s is not None:
            frames = self._extract_pitch_frames(
                pitch,
                segment_start_time_s=segment_start_time_s,
                segment_duration_s=segment_duration_s,
            )
            perturbation_provider = self._build_perturbation_measurement_provider(
                sound=sound,
                point_process=point_process,
                praat_call=call,
                segment_start_time_s=segment_start_time_s,
                segment_duration_s=segment_duration_s,
            )
            observational_result = compute_stimulus_locked_acoustic_result(
                frames,
                stimulus_time_s=stimulus_time_s,
                perturbation_measurement_provider=perturbation_provider,
            )

        return AcousticMetrics.from_observational_result(observational_result)
