#!/usr/bin/env python3
"""Gate 0 replay parity on the production CUDA speech path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.ml_core.gpu_probe import GpuInfo, query_gpu_inventory  # noqa: E402
from packages.ml_core.transcription import (  # noqa: E402
    TranscriptionEngine,
    resolve_speech_device,
)

DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "v4_gate0"
DEFAULT_CAPTURE_AUDIO = REPO_ROOT / "tests" / "fixtures" / "capture" / "baseline" / "audio.wav"
DEFAULT_STIMULUS_SCRIPT = (
    REPO_ROOT / "tests" / "fixtures" / "capture" / "baseline" / "stimulus_script.json"
)
TURING_COMPUTE_CAP = 7.5
DEFAULT_MIN_TOKEN_RECALL = 0.60
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"a", "an", "and", "are", "be", "been", "for", "s", "the", "to", "you"})


class Gate0ReplayError(RuntimeError):
    """Raised when the Gate 0 replay cannot prove production parity."""


class SpeechEngine(Protocol):
    def transcribe(self, audio: str, language: str | None = None) -> str: ...


class SpeechEngineFactory(Protocol):
    def __call__(self, model_size: str, device: str | None = None) -> SpeechEngine: ...


@dataclass(frozen=True)
class FixtureCheck:
    fixture_path: Path
    segment_id: str
    expected_segment_id: str
    segment_id_matches: bool
    expected_greeting: str

    def to_json(self) -> dict[str, Any]:
        return {
            "fixture_path": str(self.fixture_path),
            "segment_id": self.segment_id,
            "expected_segment_id": self.expected_segment_id,
            "segment_id_matches": self.segment_id_matches,
            "expected_greeting": self.expected_greeting,
        }


@dataclass(frozen=True)
class PhraseCheck:
    expected_text: str
    matched_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    recall: float
    passed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "expected_text": self.expected_text,
            "matched_tokens": list(self.matched_tokens),
            "missing_tokens": list(self.missing_tokens),
            "recall": self.recall,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class Gate0ReplayReport:
    gpu_inventory: tuple[GpuInfo, ...]
    speech_device: str
    model_size: str
    transcript: str
    fixture_checks: tuple[FixtureCheck, ...]
    phrase_checks: tuple[PhraseCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.segment_id_matches for check in self.fixture_checks) and all(
            check.passed for check in self.phrase_checks
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gpu_inventory": [
                {"name": gpu.name, "compute_cap": gpu.compute_cap} for gpu in self.gpu_inventory
            ],
            "speech_device": self.speech_device,
            "model_size": self.model_size,
            "transcript": self.transcript,
            "fixture_checks": [check.to_json() for check in self.fixture_checks],
            "phrase_checks": [check.to_json() for check in self.phrase_checks],
        }


def run_gate0_replay(
    *,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    capture_audio: Path = DEFAULT_CAPTURE_AUDIO,
    stimulus_script: Path = DEFAULT_STIMULUS_SCRIPT,
    model_size: str = "large-v3",
    language: str | None = "en",
    min_token_recall: float = DEFAULT_MIN_TOKEN_RECALL,
    engine_factory: SpeechEngineFactory = TranscriptionEngine,
) -> Gate0ReplayReport:
    inventory = tuple(_require_production_gpu())
    speech_device = _require_cuda_speech_device()
    fixture_checks = tuple(_validate_fixture_contract(fixture_dir, stimulus_script))
    expected_phrases = tuple(check.expected_greeting for check in fixture_checks)
    transcript = _run_cuda_transcription(
        capture_audio,
        model_size=model_size,
        language=language,
        engine_factory=engine_factory,
    )
    phrase_checks = tuple(
        _check_phrase(transcript, phrase, min_token_recall=min_token_recall)
        for phrase in expected_phrases
    )
    report = Gate0ReplayReport(
        gpu_inventory=inventory,
        speech_device=speech_device,
        model_size=model_size,
        transcript=transcript,
        fixture_checks=fixture_checks,
        phrase_checks=phrase_checks,
    )
    if not report.passed:
        raise Gate0ReplayError(_format_failure(report))
    return report


def write_report(report: Gate0ReplayReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--capture-audio", type=Path, default=DEFAULT_CAPTURE_AUDIO)
    parser.add_argument("--stimulus-script", type=Path, default=DEFAULT_STIMULUS_SCRIPT)
    parser.add_argument(
        "--model-size",
        default=os.environ.get("LSIE_GATE0_WHISPER_MODEL", "large-v3"),
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-token-recall", type=float, default=DEFAULT_MIN_TOKEN_RECALL)
    parser.add_argument("--report-path", type=Path, default=Path("gate0_gpu_replay_report.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_gate0_replay(
            fixture_dir=args.fixture_dir,
            capture_audio=args.capture_audio,
            stimulus_script=args.stimulus_script,
            model_size=args.model_size,
            language=args.language or None,
            min_token_recall=args.min_token_recall,
        )
    except Gate0ReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    write_report(report, args.report_path)
    print(f"Gate 0 GPU replay passed; report written to {args.report_path}")
    return 0


def _require_production_gpu() -> list[GpuInfo]:
    inventory = query_gpu_inventory()
    if not inventory:
        raise Gate0ReplayError("Gate 0 GPU replay requires nvidia-smi-visible GPU inventory")
    best = max(gpu.compute_cap for gpu in inventory)
    if best < TURING_COMPUTE_CAP:
        observed = ", ".join(f"{gpu.name} SM {gpu.compute_cap:.1f}" for gpu in inventory)
        raise Gate0ReplayError(
            f"Gate 0 GPU replay requires NVIDIA Turing SM 7.5+; observed {observed}"
        )
    return inventory


def _require_cuda_speech_device() -> str:
    if os.environ.get("LSIE_DEV_FORCE_CPU_SPEECH") == "1":
        raise Gate0ReplayError("LSIE_DEV_FORCE_CPU_SPEECH=1 is forbidden in Gate 0 GPU replay")
    device = resolve_speech_device()
    if device != "cuda":
        raise Gate0ReplayError(f"Gate 0 expected speech device 'cuda', got {device!r}")
    return device


def _validate_fixture_contract(fixture_dir: Path, stimulus_script: Path) -> list[FixtureCheck]:
    fixtures = _load_segment_fixtures(fixture_dir)
    expected_by_segment = _expected_greetings_by_segment(stimulus_script)
    checks: list[FixtureCheck] = []
    for fixture_path, fixture in fixtures:
        segment_id = _require_str(fixture, "segment_id", fixture_path)
        expected_segment_id = _derive_segment_id(fixture, fixture_path)
        segment_index = len(checks)
        expected_greeting = expected_by_segment.get(segment_index) or _require_str(
            fixture, "_expected_greeting", fixture_path
        )
        checks.append(
            FixtureCheck(
                fixture_path=fixture_path,
                segment_id=segment_id,
                expected_segment_id=expected_segment_id,
                segment_id_matches=segment_id == expected_segment_id,
                expected_greeting=expected_greeting,
            )
        )
    return checks


def _load_segment_fixtures(fixture_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not fixture_dir.exists():
        raise Gate0ReplayError(f"Gate 0 fixture directory does not exist: {fixture_dir}")
    paths = sorted(fixture_dir.glob("segment_*.json"))
    if not paths:
        raise Gate0ReplayError(
            f"Gate 0 fixture directory has no segment_*.json files: {fixture_dir}"
        )
    fixtures: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise Gate0ReplayError(f"Gate 0 fixture is not a JSON object: {path}")
        fixtures.append((path, data))
    return fixtures


def _expected_greetings_by_segment(stimulus_script: Path) -> dict[int, str]:
    script = json.loads(stimulus_script.read_text(encoding="utf-8"))
    stimuli = script.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise Gate0ReplayError(f"stimulus script has no stimuli: {stimulus_script}")
    expected: dict[int, str] = {}
    for raw in stimuli:
        if not isinstance(raw, dict):
            continue
        segment_index = raw.get("segment_index")
        expected_text = raw.get("expected_greeting_text")
        if isinstance(segment_index, int) and isinstance(expected_text, str):
            expected[segment_index] = expected_text
    if not expected:
        raise Gate0ReplayError(f"stimulus script has no expected greeting text: {stimulus_script}")
    return expected


def _derive_segment_id(fixture: dict[str, Any], fixture_path: Path) -> str:
    stable_identity = "|".join(
        (
            f"{uuid.UUID(_require_str(fixture, 'session_id', fixture_path))}",
            _canonical_utc_timestamp(
                _require_str(fixture, "segment_window_start_utc", fixture_path)
            ),
            _canonical_utc_timestamp(_require_str(fixture, "segment_window_end_utc", fixture_path)),
        )
    )
    return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()


def _canonical_utc_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_str(fixture: dict[str, Any], key: str, fixture_path: Path) -> str:
    value = fixture.get(key)
    if not isinstance(value, str):
        raise Gate0ReplayError(f"Gate 0 fixture {fixture_path} missing string key {key}")
    return value


def _run_cuda_transcription(
    capture_audio: Path,
    *,
    model_size: str,
    language: str | None,
    engine_factory: SpeechEngineFactory,
) -> str:
    if not capture_audio.exists():
        raise Gate0ReplayError(f"Gate 0 capture audio does not exist: {capture_audio}")
    engine = engine_factory(model_size=model_size, device="cuda")
    transcript = engine.transcribe(str(capture_audio), language=language)
    if not transcript.strip():
        raise Gate0ReplayError("Gate 0 CUDA transcription returned empty text")
    return transcript


def _check_phrase(transcript: str, expected: str, *, min_token_recall: float) -> PhraseCheck:
    transcript_tokens = set(_tokens(transcript))
    expected_tokens = tuple(_tokens(expected))
    if not expected_tokens:
        raise Gate0ReplayError(f"expected phrase has no comparable tokens: {expected!r}")
    matched = tuple(token for token in expected_tokens if token in transcript_tokens)
    missing = tuple(token for token in expected_tokens if token not in transcript_tokens)
    recall = len(matched) / len(expected_tokens)
    return PhraseCheck(
        expected_text=expected,
        matched_tokens=matched,
        missing_tokens=missing,
        recall=recall,
        passed=recall >= min_token_recall,
    )


def _tokens(value: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(value.lower()) if token not in _STOPWORDS]


def _format_failure(report: Gate0ReplayReport) -> str:
    lines = ["Gate 0 GPU replay parity failed"]
    for fixture_check in report.fixture_checks:
        if not fixture_check.segment_id_matches:
            lines.append(
                "segment_id mismatch "
                f"{fixture_check.fixture_path.name}: expected {fixture_check.expected_segment_id}, "
                f"got {fixture_check.segment_id}"
            )
    for phrase_check in report.phrase_checks:
        if not phrase_check.passed:
            lines.append(
                "transcript token recall below threshold for "
                f"{phrase_check.expected_text!r}: recall={phrase_check.recall:.3f}, "
                f"missing={list(phrase_check.missing_tokens)}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
