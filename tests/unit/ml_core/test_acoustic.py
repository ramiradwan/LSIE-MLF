"""
Tests for packages/ml_core/acoustic.py — Phase 1 validation.

Verifies AcousticAnalyzer against §4.D.3 legacy compatibility extraction,
pure §7D observational acoustic helpers, and AcousticMetrics as the
canonical v3.3 acoustic payload envelope.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from packages.ml_core.acoustic import (
    ACOUSTIC_MIN_PERIODIC_PEAKS,
    ACOUSTIC_MIN_STABLE_ISLAND_S,
    ACOUSTIC_MIN_VOICED_COVERAGE_S,
    ACOUSTIC_VOICING_THRESHOLD,
    AcousticAnalyticsResult,
    AcousticAnalyzer,
    AcousticMetrics,
    AcousticPerturbationMeasurement,
    TimestampedAcousticFrame,
    compute_f0_delta_semitones,
    compute_observational_acoustic_result,
    compute_stimulus_locked_acoustic_result,
    extract_baseline_acoustic_window,
    extract_stimulus_acoustic_window,
    null_acoustic_result,
)


def _make_acoustic_frames(
    span_start_s: float,
    span_end_s: float,
    *,
    frame_duration_s: float = 0.1,
    f0_hz: float | None = 180.0,
    voicing_strength: float = 0.8,
) -> list[TimestampedAcousticFrame]:
    """Create evenly-spaced frame centers covering the requested span."""
    frames: list[TimestampedAcousticFrame] = []
    frame_center_s = span_start_s + (frame_duration_s / 2.0)
    while frame_center_s <= (span_end_s - (frame_duration_s / 2.0) + 1e-9):
        frames.append(
            TimestampedAcousticFrame(
                timestamp_s=frame_center_s,
                duration_s=frame_duration_s,
                f0_hz=f0_hz,
                voicing_strength=voicing_strength,
            )
        )
        frame_center_s += frame_duration_s
    return frames


def _assert_false_null_contract(result: AcousticAnalyticsResult) -> None:
    assert result.f0_valid_measure is False
    assert result.f0_valid_baseline is False
    assert result.perturbation_valid_measure is False
    assert result.perturbation_valid_baseline is False
    assert result.voiced_coverage_measure_s == 0.0
    assert result.voiced_coverage_baseline_s == 0.0
    assert result.f0_mean_measure_hz is None
    assert result.f0_mean_baseline_hz is None
    assert result.f0_delta_semitones is None


class TestAcousticAnalyticsResult:
    """§7D / §12.4 — pure observational acoustic result helpers."""

    def test_section_7d_constants_match_spec(self) -> None:
        assert ACOUSTIC_VOICING_THRESHOLD == 0.45
        assert ACOUSTIC_MIN_VOICED_COVERAGE_S == 1.0
        assert ACOUSTIC_MIN_STABLE_ISLAND_S == 0.2
        assert ACOUSTIC_MIN_PERIODIC_PEAKS == 4

    def test_null_acoustic_result_follows_false_null_contract(self) -> None:
        result = null_acoustic_result()

        assert isinstance(result, AcousticAnalyticsResult)
        _assert_false_null_contract(result)

    def test_to_dict_emits_only_canonical_section_7d_fields(self) -> None:
        payload = null_acoustic_result().to_dict()

        assert tuple(payload.keys()) == (
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
        assert len(payload) == 15
        for forbidden_field in (
            "gated_reward",
            "p90_intensity",
            "semantic_gate",
            "reward_z",
            "z_score",
            "z_scored",
        ):
            assert forbidden_field not in payload

    def test_compute_observational_result_derives_deltas_and_keeps_f0_independent(
        self,
    ) -> None:
        """F0 outputs remain valid even when perturbation validity fails."""
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=False,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=1.8,
            voiced_coverage_baseline_s=1.2,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            jitter_mean_measure=0.012,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=0.035,
            shimmer_mean_baseline=0.03,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.voiced_coverage_measure_s == 1.8
        assert result.voiced_coverage_baseline_s == 1.2
        assert result.f0_mean_measure_hz == 180.0
        assert result.f0_mean_baseline_hz == 150.0
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))

        # Perturbation means are nulled only for invalid perturbation windows.

    def test_compute_observational_result_nulls_baseline_f0_and_delta_when_baseline_invalid(
        self,
    ) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=False,
            perturbation_valid_measure=False,
            perturbation_valid_baseline=False,
            voiced_coverage_measure_s=1.2,
            voiced_coverage_baseline_s=0.8,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
        )

        assert result.f0_mean_measure_hz == 180.0
        assert result.f0_mean_baseline_hz is None
        assert result.f0_delta_semitones is None

    def test_compute_observational_result_clamps_perturbation_valid_when_f0_invalid(
        self,
    ) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=False,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=0.6,
            voiced_coverage_baseline_s=1.4,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            jitter_mean_measure=0.012,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=0.035,
            shimmer_mean_baseline=0.03,
        )

        assert result.f0_valid_measure is False
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.f0_mean_measure_hz is None
        assert result.f0_mean_baseline_hz == 150.0
        assert result.f0_delta_semitones is None

    def test_compute_observational_result_requires_computable_perturbation_metrics(
        self,
    ) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=1.5,
            voiced_coverage_baseline_s=1.3,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            jitter_mean_measure=0.012,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=None,
            shimmer_mean_baseline=0.03,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.f0_mean_measure_hz == 180.0
        assert result.f0_mean_baseline_hz == 150.0
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))

    @pytest.mark.parametrize(
        ("jitter_mean_measure", "shimmer_mean_measure"),
        [
            (np.nan, 0.03),
            (0.012, np.nan),
            (np.inf, 0.03),
            (0.012, np.inf),
        ],
    )
    def test_compute_observational_result_treats_non_finite_perturbation_metrics_as_not_computable(
        self,
        jitter_mean_measure: float,
        shimmer_mean_measure: float,
    ) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=1.5,
            voiced_coverage_baseline_s=1.3,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            jitter_mean_measure=jitter_mean_measure,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=shimmer_mean_measure,
            shimmer_mean_baseline=0.03,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.f0_mean_measure_hz == 180.0
        assert result.f0_mean_baseline_hz == 150.0
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))

    def test_compute_f0_delta_semitones_matches_canonical_formula(self) -> None:
        assert compute_f0_delta_semitones(180.0, 150.0) == pytest.approx(
            12.0 * np.log2(180.0 / 150.0)
        )

    def test_compute_f0_delta_semitones_null_when_not_computable(self) -> None:
        assert compute_f0_delta_semitones(None, 150.0) is None
        assert compute_f0_delta_semitones(180.0, None) is None
        assert compute_f0_delta_semitones(180.0, 0.0) is None
        assert compute_f0_delta_semitones(180.0, -10.0) is None

    def test_compute_observational_result_uses_additive_perturbation_delta_math(self) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=1.6,
            voiced_coverage_baseline_s=1.4,
            f0_mean_measure_hz=220.0,
            f0_mean_baseline_hz=110.0,
            jitter_mean_measure=0.014,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=0.041,
            shimmer_mean_baseline=0.033,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is True
        assert result.perturbation_valid_baseline is True
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(220.0 / 110.0))


class TestStimulusLockedAcousticAnalytics:
    """§7D.2–§7D.5 — windowing, gating, and summary math."""

    def test_extracts_windows_with_existing_section_7b_boundaries(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            TimestampedAcousticFrame(94.99, 0.1, 150.0, 0.8),
            TimestampedAcousticFrame(95.0, 0.1, 150.0, 0.8),
            TimestampedAcousticFrame(98.0, 0.1, 150.0, 0.8),
            TimestampedAcousticFrame(98.01, 0.1, 150.0, 0.8),
            TimestampedAcousticFrame(100.49, 0.1, 180.0, 0.8),
            TimestampedAcousticFrame(100.5, 0.1, 180.0, 0.8),
            TimestampedAcousticFrame(105.0, 0.1, 180.0, 0.8),
            TimestampedAcousticFrame(105.01, 0.1, 180.0, 0.8),
        ]

        baseline = extract_baseline_acoustic_window(frames, stimulus_time_s)
        measure = extract_stimulus_acoustic_window(frames, stimulus_time_s)

        assert [frame.timestamp_s for frame in baseline] == [95.0, 98.0]
        assert [frame.timestamp_s for frame in measure] == [100.5, 105.0]

    def test_compute_stimulus_locked_result_is_deterministic_for_null_stimulus(self) -> None:
        frames = _make_acoustic_frames(95.0, 97.0)
        result = compute_stimulus_locked_acoustic_result(frames, stimulus_time_s=None)
        _assert_false_null_contract(result)

    def test_silent_windows_are_deterministic_and_do_not_probe_perturbation(self) -> None:
        stimulus_time_s = 100.0

        def provider(_start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            raise AssertionError("provider should not be called for silent windows")

        result = compute_stimulus_locked_acoustic_result(
            [],
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        _assert_false_null_contract(result)

    def test_keeps_f0_validity_independent_from_perturbation_validity(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 96.2, f0_hz=150.0),
            *_make_acoustic_frames(100.6, 101.8, f0_hz=180.0),
        ]

        def provider(start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            if start_s < 99.0:
                return AcousticPerturbationMeasurement(
                    periodic_peak_count=6,
                    jitter_local=0.008,
                    shimmer_local=0.018,
                )
            return AcousticPerturbationMeasurement(
                periodic_peak_count=3,
                jitter_local=0.010,
                shimmer_local=0.020,
            )

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.voiced_coverage_measure_s == pytest.approx(1.2)
        assert result.voiced_coverage_baseline_s == pytest.approx(1.2)
        assert result.f0_mean_measure_hz == pytest.approx(180.0)
        assert result.f0_mean_baseline_hz == pytest.approx(150.0)
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))

    def test_invalid_f0_window_nulls_mean_and_semitone_delta(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 95.8, f0_hz=150.0),
            *_make_acoustic_frames(100.6, 101.8, f0_hz=180.0),
        ]

        def provider(_start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            return AcousticPerturbationMeasurement(
                periodic_peak_count=6,
                jitter_local=0.010,
                shimmer_local=0.020,
            )

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is False
        assert result.perturbation_valid_measure is True
        assert result.perturbation_valid_baseline is False
        assert result.f0_mean_measure_hz == pytest.approx(180.0)
        assert result.f0_mean_baseline_hz is None
        assert result.f0_delta_semitones is None

    def test_uses_longest_eligible_voiced_island_for_perturbation(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 97.0, f0_hz=180.0),
            *_make_acoustic_frames(100.6, 102.2, f0_hz=220.0),
            *_make_acoustic_frames(102.8, 104.0, f0_hz=220.0),
            *_make_acoustic_frames(104.3, 104.8, f0_hz=220.0),
        ]

        def provider(start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            if start_s < 99.0:
                return AcousticPerturbationMeasurement(
                    periodic_peak_count=6,
                    jitter_local=0.008,
                    shimmer_local=0.018,
                )
            if start_s < 102.5:
                return AcousticPerturbationMeasurement(
                    periodic_peak_count=3,
                    jitter_local=0.030,
                    shimmer_local=0.040,
                )
            if start_s < 104.1:
                return AcousticPerturbationMeasurement(
                    periodic_peak_count=6,
                    jitter_local=0.011,
                    shimmer_local=0.021,
                )
            return AcousticPerturbationMeasurement(
                periodic_peak_count=6,
                jitter_local=0.015,
                shimmer_local=0.025,
            )

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        assert result.f0_valid_measure is True
        assert result.perturbation_valid_measure is True
        assert result.f0_mean_measure_hz == pytest.approx(220.0)

    def test_unvoiced_windows_are_deterministic_and_do_not_probe_perturbation(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 97.0, f0_hz=0.0, voicing_strength=0.2),
            *_make_acoustic_frames(100.6, 101.8, f0_hz=0.0, voicing_strength=0.2),
        ]

        def provider(_start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            raise AssertionError("provider should not be called when F0 validity is false")

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        _assert_false_null_contract(result)

    def test_voicing_threshold_masking_preserves_window_means_and_coverage(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            TimestampedAcousticFrame(95.1, 0.2, 100.0, ACOUSTIC_VOICING_THRESHOLD),
            TimestampedAcousticFrame(95.3, 0.2, 110.0, 0.8),
            TimestampedAcousticFrame(95.5, 0.2, 120.0, 0.8),
            TimestampedAcousticFrame(95.7, 0.2, 130.0, 0.8),
            TimestampedAcousticFrame(95.9, 0.2, 140.0, 0.8),
            TimestampedAcousticFrame(96.1, 0.2, 999.0, ACOUSTIC_VOICING_THRESHOLD - 0.01),
            TimestampedAcousticFrame(100.7, 0.2, 200.0, ACOUSTIC_VOICING_THRESHOLD),
            TimestampedAcousticFrame(100.9, 0.2, 210.0, 0.8),
            TimestampedAcousticFrame(101.1, 0.2, 220.0, 0.8),
            TimestampedAcousticFrame(101.3, 0.2, 230.0, 0.8),
            TimestampedAcousticFrame(101.5, 0.2, 240.0, 0.8),
            TimestampedAcousticFrame(101.7, 0.2, 999.0, ACOUSTIC_VOICING_THRESHOLD - 0.01),
        ]
        expected_measure_mean_hz = np.mean([200.0, 210.0, 220.0, 230.0, 240.0])
        expected_baseline_mean_hz = np.mean([100.0, 110.0, 120.0, 130.0, 140.0])

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is False
        assert result.voiced_coverage_measure_s == pytest.approx(1.0)
        assert result.voiced_coverage_baseline_s == pytest.approx(1.0)
        assert result.f0_mean_measure_hz == pytest.approx(expected_measure_mean_hz)
        assert result.f0_mean_baseline_hz == pytest.approx(expected_baseline_mean_hz)
        assert result.f0_delta_semitones == pytest.approx(
            12.0 * np.log2(expected_measure_mean_hz / expected_baseline_mean_hz)
        )

    def test_insufficient_voiced_coverage_is_deterministic_and_skips_perturbation(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 95.8, f0_hz=150.0),
            *_make_acoustic_frames(100.6, 101.2, f0_hz=180.0),
        ]
        provider_called = False

        def provider(_start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            nonlocal provider_called
            provider_called = True
            return AcousticPerturbationMeasurement(
                periodic_peak_count=6,
                jitter_local=0.010,
                shimmer_local=0.020,
            )

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        assert provider_called is False
        assert result.f0_valid_measure is False
        assert result.f0_valid_baseline is False
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is False
        assert result.voiced_coverage_measure_s == pytest.approx(0.6)
        assert result.voiced_coverage_baseline_s == pytest.approx(0.8)
        assert result.f0_mean_measure_hz is None
        assert result.f0_mean_baseline_hz is None
        assert result.f0_delta_semitones is None

    def test_undefined_perturbation_outputs_null_only_perturbation_fields(self) -> None:
        stimulus_time_s = 100.0
        frames = [
            *_make_acoustic_frames(95.0, 96.2, f0_hz=150.0),
            *_make_acoustic_frames(100.6, 101.8, f0_hz=180.0),
        ]

        def provider(start_s: float, _end_s: float) -> AcousticPerturbationMeasurement:
            if start_s < 99.0:
                return AcousticPerturbationMeasurement(
                    periodic_peak_count=6,
                    jitter_local=0.008,
                    shimmer_local=0.018,
                )
            return AcousticPerturbationMeasurement(
                periodic_peak_count=6,
                jitter_local=None,
                shimmer_local=np.nan,
            )

        result = compute_stimulus_locked_acoustic_result(
            frames,
            stimulus_time_s=stimulus_time_s,
            perturbation_measurement_provider=provider,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is False
        assert result.perturbation_valid_baseline is True
        assert result.voiced_coverage_measure_s == pytest.approx(1.2)
        assert result.voiced_coverage_baseline_s == pytest.approx(1.2)
        assert result.f0_mean_measure_hz == pytest.approx(180.0)
        assert result.f0_mean_baseline_hz == pytest.approx(150.0)
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))


class TestAcousticMetrics:
    """§11 / §7D — acoustic payload data model."""

    def test_defaults_follow_canonical_false_null_contract(self) -> None:
        m = AcousticMetrics()

        assert m.f0_valid_measure is False
        assert m.f0_valid_baseline is False
        assert m.perturbation_valid_measure is False
        assert m.perturbation_valid_baseline is False
        assert m.voiced_coverage_measure_s == 0.0
        assert m.voiced_coverage_baseline_s == 0.0
        assert m.f0_mean_measure_hz is None
        assert m.f0_mean_baseline_hz is None
        assert m.f0_delta_semitones is None

    def test_with_canonical_values(self) -> None:
        m = AcousticMetrics(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=False,
            voiced_coverage_measure_s=2.5,
            voiced_coverage_baseline_s=1.75,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            f0_delta_semitones=3.157376,
            jitter_mean_measure=0.012,
            jitter_mean_baseline=0.009,
            jitter_delta=0.003,
            shimmer_mean_measure=0.035,
            shimmer_mean_baseline=0.03,
            shimmer_delta=0.005,
        )

        assert m.f0_valid_measure is True
        assert m.f0_valid_baseline is True
        assert m.perturbation_valid_measure is True
        assert m.perturbation_valid_baseline is False
        assert m.voiced_coverage_measure_s == 2.5
        assert m.voiced_coverage_baseline_s == 1.75
        assert m.f0_mean_measure_hz == 180.0
        assert m.f0_mean_baseline_hz == 150.0
        assert m.f0_delta_semitones == 3.157376

    def test_from_observational_result_preserves_canonical_values(self) -> None:
        result = compute_observational_acoustic_result(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=2.0,
            voiced_coverage_baseline_s=1.5,
            f0_mean_measure_hz=180.0,
            f0_mean_baseline_hz=150.0,
            jitter_mean_measure=0.012,
            jitter_mean_baseline=0.009,
            shimmer_mean_measure=0.035,
            shimmer_mean_baseline=0.03,
        )

        metrics = AcousticMetrics.from_observational_result(result)

        assert metrics.f0_valid_measure is True
        assert metrics.f0_valid_baseline is True
        assert metrics.perturbation_valid_measure is True
        assert metrics.perturbation_valid_baseline is True
        assert metrics.f0_delta_semitones == pytest.approx(12.0 * np.log2(180.0 / 150.0))


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

    def test_analyze_returns_metrics(self, mock_parselmouth: tuple[MagicMock, MagicMock]) -> None:
        """§4.D.3 — Returns AcousticMetrics with canonical defaults."""
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
        assert result.f0_valid_measure is False
        assert result.voiced_coverage_measure_s == 0.0
        assert result.f0_mean_measure_hz is None

    def test_analyze_stimulus_locked_populates_section_7d_fields(
        self,
        mock_parselmouth: tuple[MagicMock, MagicMock],
    ) -> None:
        """Stimulus-locked analysis computes canonical §7D validity and deltas."""
        mock_pm, mock_praat = mock_parselmouth
        mock_sound = MagicMock()
        mock_pm.Sound.return_value = mock_sound

        mock_point_process = MagicMock()

        class FakePitch:
            def __init__(self) -> None:
                frame_times_s = np.arange(110, dtype=np.float64) * 0.1 + 0.05
                frequencies = np.zeros_like(frame_times_s)
                strengths = np.zeros_like(frame_times_s)

                baseline_mask = (frame_times_s >= 0.05) & (frame_times_s < 2.05)
                measure_mask = (frame_times_s >= 5.55) & (frame_times_s < 6.85)
                frequencies[baseline_mask] = 180.0
                strengths[baseline_mask] = 0.8
                frequencies[measure_mask] = 220.0
                strengths[measure_mask] = 0.8

                self.selected_array = {
                    "frequency": frequencies,
                    "strength": strengths,
                }
                self._frame_times_s = frame_times_s

            def xs(self) -> npt.NDArray[np.float64]:
                return self._frame_times_s

            def get_time_step(self) -> float:
                return 0.1

        fake_pitch = FakePitch()

        def call_side_effect(*args: Any, **kwargs: Any) -> Any:
            cmd = args[1] if len(args) > 1 else ""
            if cmd == "To Pitch":
                return fake_pitch
            if cmd == "Get mean":
                return 200.0
            if cmd == "To PointProcess (periodic, cc)":
                return mock_point_process
            if cmd == "Get low index":
                return 1
            if cmd == "Get high index":
                return 6
            if cmd == "Get jitter (local)":
                start_s = float(args[2])
                end_s = float(args[3])
                if start_s == 0.0 and end_s == 0.0:
                    return 0.012
                if start_s < 3.0:
                    return 0.008
                return 0.010
            if cmd == "Get shimmer (local)":
                start_s = float(args[2])
                end_s = float(args[3])
                if start_s == 0.0 and end_s == 0.0:
                    return 0.035
                if start_s < 3.0:
                    return 0.018
                return 0.020
            return None

        mock_praat.call.side_effect = call_side_effect

        samples = np.zeros(110, dtype=np.int16)
        analyzer = AcousticAnalyzer()
        result = analyzer.analyze(
            samples.tobytes(),
            sample_rate=10,
            stimulus_time_s=100.0,
            segment_start_time_s=95.0,
        )

        assert result.f0_valid_measure is True
        assert result.f0_valid_baseline is True
        assert result.perturbation_valid_measure is True
        assert result.perturbation_valid_baseline is True
        assert result.voiced_coverage_measure_s == pytest.approx(1.3)
        assert result.voiced_coverage_baseline_s == pytest.approx(2.0)
        assert result.f0_mean_measure_hz == pytest.approx(220.0)
        assert result.f0_mean_baseline_hz == pytest.approx(180.0)
        assert result.f0_delta_semitones == pytest.approx(12.0 * np.log2(220.0 / 180.0))

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
