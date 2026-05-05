"""Integration coverage for synthetic replay capture fixtures.

The tests generate small deterministic fixtures, instantiate the orchestrator
through ``REPLAY_CAPTURE_FIXTURE``, and assemble replay-backed segments without
opening live IPC pipes.  The segment test intentionally uses the production
``FaceMeshProcessor``/``AU12Normalizer`` path; it skips only when the worker ML
FaceMesh dependencies are unavailable in the local test environment.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import uuid
import wave
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.generate_capture_fixture import (
    EMBEDDED_SPEECH_BACKEND_ID,
    EMBEDDED_SPEECH_BACKEND_VERSION,
    SPEECH_BACKEND_EMBEDDED,
    _parse_args,
)
from scripts.generate_capture_fixture import (
    main as generate_capture_fixture,
)
from services.worker.pipeline.orchestrator import Orchestrator
from services.worker.pipeline.replay_capture import (
    ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    ReplayCaptureSource,
)


def _generate_fixture(
    path: Path,
    *,
    segments: int = 3,
    segment_duration_s: float = 3.0,
    width: int = 320,
    height: int = 240,
    speech_backend: str = SPEECH_BACKEND_EMBEDDED,
) -> None:
    exit_code = generate_capture_fixture(
        [
            str(path),
            "--segments",
            str(segments),
            "--segment-duration-s",
            str(segment_duration_s),
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            "2026",
            "--speech-backend",
            speech_backend,
            "--overwrite",
        ]
    )
    assert exit_code == 0


def _fixture_bytes(path: Path) -> dict[str, bytes]:
    return {
        name: (path / name).read_bytes()
        for name in ("video.mkv", "audio.wav", "stimulus_script.json")
    }


def _load_script(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((path / "stimulus_script.json").read_text(encoding="utf-8")),
    )


def _assert_embedded_speech_backend(script: dict[str, Any]) -> None:
    """Assert fixture metadata fully pins the deterministic embedded backend."""
    assert script["audio_synthesis"] == "deterministic_offline_lexical_speech:embedded"
    assert script["speech_backend"] == {
        "deterministic": True,
        "identifier": EMBEDDED_SPEECH_BACKEND_ID,
        "requested": SPEECH_BACKEND_EMBEDDED,
        "used": SPEECH_BACKEND_EMBEDDED,
        "version": EMBEDDED_SPEECH_BACKEND_VERSION,
    }


def _read_fixture_audio(path: Path) -> tuple[np.ndarray[Any, np.dtype[np.int16]], int]:
    with wave.open(str(path / "audio.wav"), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == SAMPLE_WIDTH_BYTES
        assert wav_file.getframerate() == 48_000
        sample_rate = wav_file.getframerate()
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return samples, sample_rate


def _assert_scripted_speech_energy(path: Path, script: dict[str, Any]) -> None:
    """Verify lexical speech energy is present at every scripted greeting offset."""
    samples, sample_rate = _read_fixture_audio(path)
    segment_duration_s = float(script["segment_duration_s"])
    for stimulus in script["stimuli"]:
        segment_start_s = int(stimulus["segment_index"]) * segment_duration_s
        stimulus_s = segment_start_s + float(stimulus["stimulus_offset_s"])
        start = int(round(stimulus_s * sample_rate))
        stop = min(samples.size, start + int(round(0.45 * sample_rate)))
        speech_window = samples[start:stop].astype(np.float64)
        assert speech_window.size > 0
        rms = float(np.sqrt(np.mean(np.square(speech_window))))
        peak = int(np.max(np.abs(speech_window)))
        assert rms > 300.0
        assert peak > 1_000


def _require_production_face_mesh_dependencies() -> None:
    """Skip narrowly when production MediaPipe/OpenCV dependencies are absent."""
    missing = [
        module_name
        for module_name in ("mediapipe", "cv2")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        pytest.skip(
            "production FaceMesh replay coverage requires worker ML dependencies "
            f"({', '.join(missing)} unavailable); no synthetic landmark shim is used"
        )


def _normalize_text(value: str) -> str:
    """Normalize text with the repository's semantic gate tolerance kept strict here."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _record_stimulus(orchestrator: Orchestrator, timestamp_s: float) -> None:
    with patch("services.worker.pipeline.orchestrator.time.time", return_value=timestamp_s):
        orchestrator.record_stimulus_injection()


def _process_frame(orchestrator: Orchestrator, timestamp_s: float) -> None:
    with patch("services.worker.pipeline.orchestrator.time.time", return_value=timestamp_s):
        orchestrator._process_video_frame()


def _assemble(orchestrator: Orchestrator, audio_data: bytes, timestamp_s: float) -> dict[str, Any]:
    with patch("services.worker.pipeline.orchestrator.time.time", return_value=timestamp_s):
        return orchestrator.assemble_segment(audio_data, [])


def _assert_downstream_semantic_grounded(
    payload: dict[str, Any],
    expected_greeting_text: str,
    bytes_per_segment: int,
) -> None:
    """Exercise Module D semantic wiring without GPU ASR or Azure network calls.

    The fake transcriber is intentionally narrow: it asserts that Module D wraps
    the replay PCM as a 16 kHz WAV, then returns the literal fixture greeting.
    The fake semantic evaluator accepts only the fixture greeting after the same
    preprocessing Module D applies before semantic evaluation.
    """
    try:
        from packages.ml_core.preprocessing import TextPreprocessor
        from services.worker.tasks import inference as inference_mod
    except ModuleNotFoundError as exc:
        pytest.skip(f"downstream semantic check requires worker task dependencies: {exc}")

    normalized_expected = _normalize_text(expected_greeting_text)
    normalized_semantic_input = _normalize_text(
        TextPreprocessor().preprocess(expected_greeting_text)
    )

    class LiteralTranscriptionEngine:
        def transcribe(self, audio_path: str, language: str | None = None) -> str:
            del language
            with wave.open(audio_path, "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == SAMPLE_WIDTH_BYTES
                assert wav_file.getframerate() == ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ
                assert wav_file.getnframes() * SAMPLE_WIDTH_BYTES == bytes_per_segment
            return expected_greeting_text

    class LiteralSemanticEvaluator:
        def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any]:
            assert _normalize_text(expected_greeting) == normalized_expected
            assert _normalize_text(actual_utterance) == normalized_semantic_input
            return {
                "reasoning": "cross_encoder_high_match",
                "is_match": True,
                "confidence_score": 1.0,
            }

    payload_for_inference = dict(payload)
    payload_for_inference["_frame_data"] = None
    payload_for_inference["_stimulus_time"] = None

    def pcm_to_wav_bytes(pcm: bytes) -> bytes:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ)
            wav_file.writeframes(pcm)
        return wav_buffer.getvalue()

    mock_persist = MagicMock()
    with (
        patch.object(inference_mod, "persist_metrics", mock_persist),
        patch("packages.ml_core.audio_pipe.pcm_to_wav_bytes", side_effect=pcm_to_wav_bytes),
        patch("packages.ml_core.transcription.TranscriptionEngine", LiteralTranscriptionEngine),
        patch("packages.ml_core.semantic.SemanticEvaluator", LiteralSemanticEvaluator),
    ):
        inference_mod._TRANSCRIPTION_ENGINE = None
        inference_mod._TRANSCRIPTION_ENGINE_FACTORY = None
        task = inference_mod.process_segment
        if hasattr(task, "run"):
            result = task.run(payload_for_inference)
        else:
            result = task(MagicMock(), payload_for_inference)

    assert _normalize_text(result["transcription"]) == normalized_expected
    assert result["semantic"]["is_match"] is True
    mock_persist.delay.assert_called_once()


@pytest.mark.integration
def test_capture_fixture_cli_defaults_to_embedded_speech_backend(tmp_path: Path) -> None:
    args = _parse_args([str(tmp_path / "fixture")])

    assert args.speech_backend == SPEECH_BACKEND_EMBEDDED


@pytest.mark.integration
def test_explicit_espeak_backend_requires_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("scripts.generate_capture_fixture.shutil.which", lambda name: None)

    fixture = tmp_path / "fixture"
    exit_code = generate_capture_fixture([str(fixture), "--speech-backend", "espeak-ng"])

    assert exit_code == 1
    assert "--speech-backend espeak-ng requires espeak-ng" in capsys.readouterr().err
    assert not fixture.exists()


@pytest.mark.integration
def test_capture_fixture_generation_is_byte_deterministic_and_lexical(tmp_path: Path) -> None:
    fixture_a = tmp_path / "fixture-a"
    fixture_b = tmp_path / "fixture-b"
    _generate_fixture(fixture_a, segments=2, segment_duration_s=2.0, width=160, height=120)
    _generate_fixture(fixture_b, segments=2, segment_duration_s=2.0, width=160, height=120)

    assert _fixture_bytes(fixture_a) == _fixture_bytes(fixture_b)
    script = _load_script(fixture_a)
    _assert_embedded_speech_backend(script)
    assert [stimulus["expected_greeting_text"] for stimulus in script["stimuli"]]
    _assert_scripted_speech_energy(fixture_a, script)


@pytest.mark.integration
def test_replay_fixture_drives_orchestrator_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_production_face_mesh_dependencies()

    fixture = tmp_path / "fixture"
    _generate_fixture(fixture, segments=3, segment_duration_s=3.0, width=320, height=240)
    script = _load_script(fixture)
    _assert_embedded_speech_backend(script)
    _assert_scripted_speech_energy(fixture, script)

    monkeypatch.setenv("REPLAY_CAPTURE_FIXTURE", str(fixture))
    monkeypatch.setenv("REPLAY_CAPTURE_REALTIME", "0")

    orchestrator = Orchestrator(
        stream_url="replay://synthetic-fixture",
        session_id=str(uuid.uuid4()),
    )
    assert isinstance(orchestrator.video_capture, ReplayCaptureSource)
    assert orchestrator.audio_resampler is orchestrator.video_capture
    assert orchestrator._using_replay_capture is True

    # Disable the retained broker-backed physiology/stimulus client to prevent
    # synchronous DNS hangs on Windows while this replay test exercises the
    # local segment-assembly path only.
    orchestrator._redis = None

    replay = orchestrator.video_capture
    replay.start()

    segment_duration_s = float(script["segment_duration_s"])
    fps = int(script["fps"])
    frames_per_segment = int(round(segment_duration_s * fps))
    bytes_per_segment = int(
        segment_duration_s * ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
    )
    epoch_s = 1_800_000_000.0

    try:
        for stimulus in script["stimuli"]:
            segment_index = int(stimulus["segment_index"])
            segment_start_s = segment_index * segment_duration_s
            stimulus_elapsed_s = replay.elapsed_for_stimulus(stimulus)
            stimulus_timestamp_s = epoch_s + stimulus_elapsed_s
            stimulus_recorded = False
            expected_greeting_text = str(stimulus["expected_greeting_text"])
            orchestrator._active_arm = str(stimulus["expected_arm_id"])
            orchestrator._expected_greeting = expected_greeting_text

            for frame_offset in range(frames_per_segment):
                elapsed_s = segment_start_s + (frame_offset / fps)
                replay.seek(elapsed_s)
                if not stimulus_recorded and elapsed_s >= stimulus_elapsed_s:
                    _record_stimulus(orchestrator, stimulus_timestamp_s)
                    stimulus_recorded = True
                _process_frame(orchestrator, epoch_s + elapsed_s)

            if not stimulus_recorded:
                _record_stimulus(orchestrator, stimulus_timestamp_s)

            audio_data = replay.read_chunk(bytes_per_segment)
            assert len(audio_data) == bytes_per_segment

            replay.seek(segment_start_s + segment_duration_s - (1.0 / fps))
            payload = _assemble(
                orchestrator,
                audio_data,
                epoch_s + segment_start_s + segment_duration_s,
            )

            payload_audio = payload["_audio_data"]
            assert payload_audio == audio_data
            audio_duration_s = len(payload_audio) / (
                ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
            )
            assert audio_duration_s == pytest.approx(segment_duration_s, abs=1 / fps)
            assert payload["segments"][0]["audio_bytes"] == bytes_per_segment
            assert payload["_stimulus_time"] == pytest.approx(stimulus_timestamp_s, abs=1e-6)
            assert payload["_active_arm"] == stimulus["expected_arm_id"]
            assert payload["_expected_greeting"] == expected_greeting_text
            assert payload["media_source"]["codec"] == "h264"
            assert payload["media_source"]["resolution"] == [
                int(script["video_width"]),
                int(script["video_height"]),
            ]

            au12_series = payload["_au12_series"]
            assert au12_series, "AU12 series should contain post-stimulus replay frames"
            peak_au12 = max(point["intensity"] for point in au12_series)
            assert peak_au12 == pytest.approx(float(stimulus["expected_peak_au12"]), abs=0.18)
            _assert_downstream_semantic_grounded(
                payload,
                expected_greeting_text,
                bytes_per_segment,
            )
    finally:
        orchestrator.stop()
