from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.gate0_gpu_replay as replay
from packages.ml_core.gpu_probe import GpuInfo


class _Engine:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[str, str | None]] = []

    def transcribe(self, audio: str, language: str | None = None) -> str:
        self.calls.append((audio, language))
        return self.transcript


class _EngineFactory:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.instances: list[_Engine] = []
        self.calls: list[dict[str, str | None]] = []

    def __call__(self, model_size: str, device: str | None = None) -> _Engine:
        self.calls.append({"model_size": model_size, "device": device})
        engine = _Engine(self.transcript)
        self.instances.append(engine)
        return engine


def _write_gate0_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    session_id = "11111111-1111-4111-8111-111111111111"
    start = "2026-04-17T12:00:00Z"
    end = "2026-04-17T12:00:30Z"
    expected_segment_id = replay._derive_segment_id(
        {
            "session_id": session_id,
            "segment_window_start_utc": start,
            "segment_window_end_utc": end,
        },
        fixture_dir / "segment_000.json",
    )
    fixture = {
        "session_id": session_id,
        "segment_id": expected_segment_id,
        "segment_window_start_utc": start,
        "segment_window_end_utc": end,
        "_stimulus_modality": "question",
        "_stimulus_payload": {
            "content_type": "text",
            "text": "Hi! What's the best advice today?",
        },
        "_expected_stimulus_rule": (
            "Deliver the direct question to the live streamer exactly as written."
        ),
        "_expected_response_rule": (
            "The live streamer answers or clearly acknowledges the direct question on stream."
        ),
    }
    (fixture_dir / "segment_000.json").write_text(json.dumps(fixture), encoding="utf-8")
    stimulus_script = tmp_path / "stimulus_script.json"
    stimulus_script.write_text(
        json.dumps(
            {
                "speech_backend": {"used": "espeak-ng"},
                "stimuli": [
                    {
                        "segment_index": 0,
                        "stimulus_definition": {
                            "stimulus_modality": "question",
                            "stimulus_payload": {
                                "content_type": "text",
                                "text": "Hi! What's the best advice today?",
                            },
                            "expected_stimulus_rule": (
                                "Deliver the direct question to the live streamer "
                                "exactly as written."
                            ),
                            "expected_response_rule": (
                                "The live streamer answers or clearly acknowledges "
                                "the direct question on stream."
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capture_audio = tmp_path / "audio.wav"
    capture_audio.write_bytes(b"RIFFsynthetic")
    return fixture_dir, stimulus_script, capture_audio, expected_segment_id


def test_require_production_gpu_rejects_missing_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "query_gpu_inventory", lambda: [])

    with pytest.raises(replay.Gate0ReplayError, match="nvidia-smi-visible"):
        replay._require_production_gpu()


def test_require_production_gpu_rejects_pascal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replay,
        "query_gpu_inventory",
        lambda: [GpuInfo("NVIDIA GeForce GTX 1080 Ti", 6.1)],
    )

    with pytest.raises(replay.Gate0ReplayError) as exc_info:
        replay._require_production_gpu()

    assert "Turing SM 7.5+" in str(exc_info.value)
    assert "NVIDIA GeForce GTX 1080 Ti SM 6.1" in str(exc_info.value)


def test_require_production_gpu_accepts_turing(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = [GpuInfo("NVIDIA T4", 7.5)]
    monkeypatch.setattr(replay, "query_gpu_inventory", lambda: inventory)

    assert replay._require_production_gpu() == inventory


def test_require_cuda_speech_device_rejects_dev_cpu_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LSIE_DEV_FORCE_CPU_SPEECH", "1")

    with pytest.raises(replay.Gate0ReplayError, match="forbidden"):
        replay._require_cuda_speech_device()


def test_require_cuda_speech_device_rejects_cpu_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    monkeypatch.setattr(replay, "resolve_speech_device", lambda: "cpu")

    with pytest.raises(replay.Gate0ReplayError, match="expected speech device 'cuda'"):
        replay._require_cuda_speech_device()


def test_run_gate0_replay_validates_fixture_and_transcribes_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_dir, stimulus_script, capture_audio, expected_segment_id = _write_gate0_inputs(tmp_path)
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    monkeypatch.setattr(replay, "query_gpu_inventory", lambda: [GpuInfo("NVIDIA T4", 7.5)])
    monkeypatch.setattr(replay, "resolve_speech_device", lambda: "cuda")
    engine_factory = _EngineFactory("Hi, the best advice today is to keep shipping carefully.")

    report = replay.run_gate0_replay(
        fixture_dir=fixture_dir,
        capture_audio=capture_audio,
        stimulus_script=stimulus_script,
        model_size="tiny",
        language="en",
        min_token_recall=0.60,
        engine_factory=engine_factory,
    )

    assert report.passed is True
    assert report.speech_device == "cuda"
    assert report.gpu_inventory == (GpuInfo("NVIDIA T4", 7.5),)
    assert report.fixture_checks[0].expected_segment_id == expected_segment_id
    assert report.fixture_checks[0].segment_id_matches is True
    assert report.fixture_checks[0].expected_stimulus_text == "Hi! What's the best advice today?"
    assert report.phrase_checks[0].passed is True
    assert engine_factory.calls == [{"model_size": "tiny", "device": "cuda"}]
    assert engine_factory.instances[0].calls == [(str(capture_audio), "en")]


def test_run_gate0_replay_raises_on_fixture_segment_id_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_dir, stimulus_script, capture_audio, _ = _write_gate0_inputs(tmp_path)
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    path = fixture_dir / "segment_000.json"
    fixture: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    fixture["segment_id"] = "0" * 64
    path.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(replay, "query_gpu_inventory", lambda: [GpuInfo("NVIDIA T4", 7.5)])
    monkeypatch.setattr(replay, "resolve_speech_device", lambda: "cuda")

    with pytest.raises(replay.Gate0ReplayError) as exc_info:
        replay.run_gate0_replay(
            fixture_dir=fixture_dir,
            capture_audio=capture_audio,
            stimulus_script=stimulus_script,
            model_size="tiny",
            engine_factory=_EngineFactory("Hi best advice today"),
        )

    assert "segment_id mismatch segment_000.json" in str(exc_info.value)


def test_validate_fixture_contract_orders_hashed_fixtures_by_segment_start(tmp_path: Path) -> None:
    fixture_dir, stimulus_script, _capture_audio, _expected_segment_id = _write_gate0_inputs(
        tmp_path
    )
    path = fixture_dir / "segment_000.json"
    first_fixture: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()

    second_fixture = dict(first_fixture)
    second_fixture["segment_window_start_utc"] = "2026-04-17T12:00:30Z"
    second_fixture["segment_window_end_utc"] = "2026-04-17T12:01:00Z"
    second_fixture["segment_id"] = replay._derive_segment_id(
        second_fixture,
        fixture_dir / "segment_000_first_by_name.json",
    )
    first_fixture["segment_id"] = replay._derive_segment_id(
        first_fixture,
        fixture_dir / "segment_zzz_second_by_name.json",
    )

    (fixture_dir / "segment_000_first_by_name.json").write_text(
        json.dumps(second_fixture),
        encoding="utf-8",
    )
    (fixture_dir / "segment_zzz_second_by_name.json").write_text(
        json.dumps(first_fixture),
        encoding="utf-8",
    )
    second_fixture["_stimulus_payload"] = {"content_type": "text", "text": "second chronological"}
    first_fixture["_stimulus_payload"] = {"content_type": "text", "text": "first chronological"}
    (fixture_dir / "segment_000_first_by_name.json").write_text(
        json.dumps(second_fixture),
        encoding="utf-8",
    )
    (fixture_dir / "segment_zzz_second_by_name.json").write_text(
        json.dumps(first_fixture),
        encoding="utf-8",
    )

    checks = replay._validate_fixture_contract(fixture_dir, stimulus_script)

    assert [check.expected_stimulus_text for check in checks] == [
        "first chronological",
        "second chronological",
    ]


def test_main_writes_failure_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "gate0_report.json"
    failure_report = replay.Gate0ReplayReport(
        gpu_inventory=(GpuInfo("NVIDIA T4", 7.5),),
        speech_device="cuda",
        model_size="tiny",
        transcript="unrelated transcript",
        fixture_checks=(
            replay.FixtureCheck(
                fixture_path=Path("segment_000.json"),
                segment_id="abc",
                expected_segment_id="abc",
                segment_id_matches=True,
                expected_stimulus_text="Hi best advice today",
            ),
        ),
        phrase_checks=(
            replay.PhraseCheck(
                expected_text="Hi best advice today",
                matched_tokens=(),
                missing_tokens=("hi", "best", "advice", "today"),
                recall=0.0,
                passed=False,
            ),
        ),
    )

    def fail_replay(**kwargs: Any) -> replay.Gate0ReplayReport:
        del kwargs
        raise replay.Gate0ReplayError("failed", report=failure_report)

    monkeypatch.setattr(replay, "run_gate0_replay", fail_replay)

    exit_code = replay.main(["--report-path", str(report_path)])

    assert exit_code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["transcript"] == "unrelated transcript"
    assert report["phrase_checks"][0]["passed"] is False


def test_run_gate0_replay_raises_on_low_token_recall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_dir, stimulus_script, capture_audio, _ = _write_gate0_inputs(tmp_path)
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    monkeypatch.setattr(replay, "query_gpu_inventory", lambda: [GpuInfo("NVIDIA T4", 7.5)])
    monkeypatch.setattr(replay, "resolve_speech_device", lambda: "cuda")

    with pytest.raises(replay.Gate0ReplayError) as exc_info:
        replay.run_gate0_replay(
            fixture_dir=fixture_dir,
            capture_audio=capture_audio,
            stimulus_script=stimulus_script,
            model_size="tiny",
            min_token_recall=0.90,
            engine_factory=_EngineFactory("unrelated transcript"),
        )

    assert "transcript token recall below threshold" in str(exc_info.value)
    assert "best" in str(exc_info.value)


def test_write_report_persists_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "gate0.json"
    report = replay.Gate0ReplayReport(
        gpu_inventory=(GpuInfo("NVIDIA T4", 7.5),),
        speech_device="cuda",
        model_size="tiny",
        transcript="Hi best advice today",
        fixture_checks=(
            replay.FixtureCheck(
                fixture_path=Path("segment_000.json"),
                segment_id="abc",
                expected_segment_id="abc",
                segment_id_matches=True,
                expected_stimulus_text="Hi best advice today",
            ),
        ),
        phrase_checks=(
            replay.PhraseCheck(
                expected_text="Hi best advice today",
                matched_tokens=("hi", "best", "advice", "today"),
                missing_tokens=(),
                recall=1.0,
                passed=True,
            ),
        ),
    )

    replay.write_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["speech_device"] == "cuda"
    assert data["gpu_inventory"] == [{"compute_cap": 7.5, "name": "NVIDIA T4"}]
