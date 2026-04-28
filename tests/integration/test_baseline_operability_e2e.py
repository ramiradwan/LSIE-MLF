"""Deterministic baseline replay E2E proof for Thompson Sampling posterior movement."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.schemas.operator_console import SessionLifecycleAccepted
from services.api.routes import experiments as experiments_route
from services.api.routes import sessions as sessions_route
from services.api.services.session_lifecycle_service import _stable_session_id_for_action
from services.worker.pipeline.orchestrator import SEGMENT_WINDOW_SECONDS, Orchestrator
from services.worker.pipeline.replay_capture import (
    ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    ReplayCaptureSource,
)

BASELINE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "capture"
    / "baseline"
    / "stimulus_script.json"
)
EXPERIMENT_ID = "baseline_operability_two_arm_proof"
STREAM_URL = "replay://baseline-operability/posterior-proof"
EPOCH_S = 1_900_000_000.0
CALIBRATION_FRAMES_REQUIRED = 45


@dataclass
class _InMemoryE2EState:
    """Shared in-memory backing store for the endpoint and worker seams."""

    experiments: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    arm_order: dict[str, list[str]] = field(default_factory=dict)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    encounters: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_intents: list[dict[str, Any]] = field(default_factory=list)

    def create_experiment(self, request: Any) -> dict[str, Any]:
        now = datetime.now(UTC)
        experiment_id = str(request.experiment_id)
        self.experiments[experiment_id] = {}
        self.arm_order[experiment_id] = []
        for arm in request.arms:
            arm_id = str(arm.arm)
            self.arm_order[experiment_id].append(arm_id)
            self.experiments[experiment_id][arm_id] = {
                "experiment_id": experiment_id,
                "label": str(request.label),
                "arm": arm_id,
                "greeting_text": str(arm.greeting_text),
                "alpha_param": 1.0,
                "beta_param": 1.0,
                "enabled": True,
                "end_dated_at": None,
                "updated_at": now,
            }
        return {
            "experiment_id": experiment_id,
            "label": str(request.label),
            "arms": [self.experiments[experiment_id][arm] for arm in self.arm_order[experiment_id]],
        }

    def experiment_rows(self, experiment_id: str) -> list[dict[str, Any]]:
        return [
            {
                "experiment_id": row["experiment_id"],
                "arm": row["arm"],
                "alpha_param": row["alpha_param"],
                "beta_param": row["beta_param"],
                "updated_at": row["updated_at"],
            }
            for row in sorted(
                self.experiments.get(experiment_id, {}).values(),
                key=lambda candidate: str(candidate["arm"]),
            )
        ]

    def register_session(self, session_id: str, stream_url: str) -> None:
        self.sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "stream_url": stream_url or "unknown",
                "started_at": datetime.now(UTC),
                "ended_at": None,
            },
        )

    def end_session(self, session_id: str) -> None:
        session = self.sessions[session_id]
        if session["ended_at"] is None:
            session["ended_at"] = datetime.now(UTC)

    def active_session_id(self) -> str | None:
        active = [
            session_id
            for session_id, session in self.sessions.items()
            if session.get("ended_at") is None
        ]
        return active[-1] if active else None


class _FakeExperimentAdminService:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state

    def create_experiment(self, request: Any) -> dict[str, Any]:
        return self._state.create_experiment(request)


class _FakeSessionLifecycleService:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state

    def request_session_start(self, request: Any) -> SessionLifecycleAccepted:
        session_id = _stable_session_id_for_action(request.client_action_id)
        intent = {
            "action": "start",
            "session_id": str(session_id),
            "stream_url": str(request.stream_url),
            "experiment_id": str(request.experiment_id),
        }
        self._state.lifecycle_intents.append(intent)
        return SessionLifecycleAccepted(
            action="start",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=datetime.now(UTC),
            message=None,
        )

    def request_session_end(self, session_id: uuid.UUID, request: Any) -> SessionLifecycleAccepted:
        if self._state.active_session_id() != str(session_id):
            raise AssertionError(f"session {session_id} is not the active replay session")
        intent = {
            "action": "end",
            "session_id": str(session_id),
            "stream_url": self._state.sessions[str(session_id)]["stream_url"],
            "experiment_id": EXPERIMENT_ID,
        }
        self._state.lifecycle_intents.append(intent)
        return SessionLifecycleAccepted(
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=datetime.now(UTC),
            message=None,
        )


class _FakeCursor:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state
        self.description: list[tuple[str]] | None = None
        self._rows: list[tuple[Any, ...]] = []
        self._index = 0

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        normalized = " ".join(sql.lower().split())
        self._index = 0

        if normalized.startswith("select experiment_id, arm, alpha_param, beta_param"):
            rows = self._state.experiment_rows(str(params["experiment_id"]))
            columns = ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"]
            self.description = [(column,) for column in columns]
            self._rows = [tuple(row[column] for column in columns) for row in rows]
            return

        if normalized.startswith("select distinct experiment_id"):
            columns = ["experiment_id"]
            self.description = [(column,) for column in columns]
            self._rows = [(experiment_id,) for experiment_id in sorted(self._state.experiments)]
            return

        if "from sessions s where s.session_id" in normalized:
            session = self._state.sessions.get(str(params["session_id"]))
            columns = ["session_id", "stream_url", "started_at", "ended_at"]
            self.description = [(column,) for column in columns]
            self._rows = [tuple(session[column] for column in columns)] if session else []
            return

        if normalized.startswith("select count(m.id) as total_segments"):
            session_id = str(params["session_id"])
            session_metrics = [
                metric for metric in self._state.metrics if metric.get("session_id") == session_id
            ]
            timestamps = [
                str(metric["timestamp_utc"])
                for metric in session_metrics
                if metric.get("timestamp_utc") is not None
            ]
            columns = [
                "total_segments",
                "avg_au12",
                "avg_pitch_f0",
                "avg_jitter",
                "avg_shimmer",
                "first_segment_at",
                "last_segment_at",
            ]
            self.description = [(column,) for column in columns]
            self._rows = [
                (
                    len(session_metrics),
                    None,
                    None,
                    None,
                    None,
                    min(timestamps) if timestamps else None,
                    max(timestamps) if timestamps else None,
                )
            ]
            return

        if normalized.startswith("select s.session_id, s.stream_url, s.started_at, s.ended_at"):
            columns = ["session_id", "stream_url", "started_at", "ended_at", "metric_count"]
            self.description = [(column,) for column in columns]
            self._rows = [
                (
                    session["session_id"],
                    session["stream_url"],
                    session["started_at"],
                    session["ended_at"],
                    len(
                        [
                            metric
                            for metric in self._state.metrics
                            if metric.get("session_id") == session["session_id"]
                        ]
                    ),
                )
                for session in self._state.sessions.values()
            ]
            return

        raise AssertionError(f"unexpected SQL in baseline E2E fake cursor: {sql}")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row


class _FakeConnection:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeEncounterCursor:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state

    def __enter__(self) -> _FakeEncounterCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        assert "INSERT INTO encounter_log" in sql
        self._state.encounters.append(dict(params))


class _FakeEncounterConnection:
    def __init__(self, state: _InMemoryE2EState) -> None:
        self._state = state

    def cursor(self) -> _FakeEncounterCursor:
        return _FakeEncounterCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ScriptedFaceMeshProcessor:
    def extract_landmarks(self, frame: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        del frame
        return np.zeros((478, 3), dtype=np.float64)

    def close(self) -> None:
        return None


class _ScriptedAU12Driver:
    current_intensity: float = 0.0


class _ScriptedAU12Normalizer:
    def __init__(self, alpha: float = 6.0) -> None:
        self.alpha = alpha
        self.b_neutral: float | None = None
        self.calibration_buffer: list[float] = []

    def compute_bounded_intensity(self, landmarks: Any, is_calibrating: bool) -> float:
        del landmarks
        if is_calibrating:
            self.calibration_buffer.append(0.0)
            self.b_neutral = 0.0
            return 0.0
        return float(_ScriptedAU12Driver.current_intensity)

    def compute_intensity(self, landmarks: Any) -> float:
        del landmarks
        return float(_ScriptedAU12Driver.current_intensity)


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _load_baseline_script() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(BASELINE_SCRIPT_PATH.read_text(encoding="utf-8")))


def _materialize_baseline_replay_fixture(tmp_path: Path, script: dict[str, Any]) -> Path:
    """Install a lightweight replay fixture honoring the 30-second script contract.

    The proof exercises the real replay/orchestrator audio path, so the WAV is
    a valid 48 kHz mono PCM stream with the scripted duration. Video pixels are
    provided by a deterministic in-memory replay seam because this E2E asserts
    orchestrator timing/reward semantics, not production MediaPipe detection.
    """
    fixture_dir = tmp_path / "baseline-replay"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    duration_s = float(script["duration_s"])
    sample_rate = int(script["audio_sample_rate_hz"])
    sample_width = int(script["audio_sample_width_bytes"])
    channels = int(script["audio_channels"])
    assert sample_rate == 48_000
    assert sample_width == SAMPLE_WIDTH_BYTES
    assert channels == 1

    total_frames = int(round(duration_s * sample_rate))
    one_second_silence = b"\x00" * (sample_rate * sample_width * channels)
    with wave.open(str(fixture_dir / "audio.wav"), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        remaining = total_frames
        while remaining > 0:
            frames = min(sample_rate, remaining)
            wav_file.writeframes(one_second_silence[: frames * sample_width * channels])
            remaining -= frames

    # ReplayCaptureSource validates file presence before loading. The test
    # replaces video decoding with a one-frame deterministic array to avoid
    # making this reward-path E2E depend on H.264 decode throughput.
    (fixture_dir / "video.mkv").write_bytes(b"baseline-operability-video-placeholder\n")
    (fixture_dir / "stimulus_script.json").write_text(
        json.dumps(script, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fixture_dir


def _build_api(state: _InMemoryE2EState, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    app.include_router(sessions_route.router, prefix="/api/v1")
    app.include_router(experiments_route.router, prefix="/api/v1")

    def experiment_admin_service() -> _FakeExperimentAdminService:
        return _FakeExperimentAdminService(state)

    def session_lifecycle_service() -> _FakeSessionLifecycleService:
        return _FakeSessionLifecycleService(state)

    app.dependency_overrides[experiments_route.get_admin_service] = experiment_admin_service
    app.dependency_overrides[sessions_route.get_session_lifecycle_service] = (
        session_lifecycle_service
    )
    monkeypatch.setattr(experiments_route, "get_connection", lambda: _FakeConnection(state))
    monkeypatch.setattr(experiments_route, "put_connection", lambda conn: None)
    monkeypatch.setattr(sessions_route, "get_connection", lambda: _FakeConnection(state))
    monkeypatch.setattr(sessions_route, "put_connection", lambda conn: None)
    return app


def _fake_metrics_store_class(state: _InMemoryE2EState) -> type[Any]:
    class FakeMetricsStore:
        def connect(self, minconn: int = 2, maxconn: int = 10) -> None:
            del minconn, maxconn

        def close(self) -> None:
            return None

        def insert_metrics(self, metrics: dict[str, Any]) -> None:
            state.metrics.append(dict(metrics))

        def get_experiment_arms(self, experiment_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "arm": arm,
                    "alpha_param": state.experiments[experiment_id][arm]["alpha_param"],
                    "beta_param": state.experiments[experiment_id][arm]["beta_param"],
                }
                for arm in state.arm_order[experiment_id]
            ]

        def get_experiment_arm(self, experiment_id: str, arm: str) -> dict[str, Any] | None:
            row = state.experiments.get(experiment_id, {}).get(arm)
            if row is None:
                return None
            return {
                "arm": arm,
                "alpha_param": row["alpha_param"],
                "beta_param": row["beta_param"],
            }

        def update_experiment_arm(
            self,
            experiment_id: str,
            arm: str,
            alpha: float,
            beta: float,
        ) -> None:
            row = state.experiments[experiment_id][arm]
            row["alpha_param"] = float(alpha)
            row["beta_param"] = float(beta)
            row["updated_at"] = datetime.now(UTC)

        def _get_conn(self) -> _FakeEncounterConnection:
            return _FakeEncounterConnection(state)

        def _put_conn(self, conn: _FakeEncounterConnection) -> None:
            del conn

    return FakeMetricsStore


def _install_worker_seams(
    monkeypatch: pytest.MonkeyPatch,
    state: _InMemoryE2EState,
    script: dict[str, Any],
    observed_reward_calls: list[dict[str, Any]],
    processed_results: list[dict[str, Any]],
    dispatched_payloads: list[dict[str, Any]],
    on_last_dispatch: Callable[[Orchestrator], None],
) -> None:
    from services.worker.pipeline import analytics as analytics_mod
    from services.worker.pipeline import reward as reward_mod
    from services.worker.tasks import inference as inference_mod

    proof = script["posterior_proof"]
    stimuli = list(script["stimuli"])
    current_stimulus: dict[str, Any] = {"row": None}

    class FixtureTranscriptionEngine:
        def transcribe(self, audio_path: str, language: str | None = None) -> str:
            del language
            with wave.open(audio_path, "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == SAMPLE_WIDTH_BYTES
                assert wav_file.getframerate() == ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ
            row = current_stimulus["row"]
            assert row is not None
            return str(row["expected_greeting_text"])

    class FixtureSemanticEvaluator:
        def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any]:
            row = current_stimulus["row"]
            assert row is not None
            assert _normalize_text(expected_greeting) == _normalize_text(
                str(row["expected_greeting_text"])
            )
            assert _normalize_text(actual_utterance) == _normalize_text(
                str(row["expected_greeting_text"])
            )
            return {
                "reasoning": "fixture-scripted deterministic semantic gate",
                "is_match": bool(row["expected_semantic_match"]),
                "confidence_score": float(row["expected_confidence_score"]),
            }

    class FixtureTextPreprocessor:
        def preprocess(self, text: str) -> str:
            return text

    real_compute_reward = reward_mod.compute_reward

    def observing_compute_reward(**kwargs: Any) -> Any:
        observed_reward_calls.append(
            {
                "au12_series": list(kwargs["au12_series"]),
                "stimulus_time_s": kwargs["stimulus_time_s"],
                "is_match": kwargs["is_match"],
                "confidence_score_present": "confidence_score" in kwargs,
                "x_max_present": "x_max" in kwargs,
            }
        )
        return real_compute_reward(**kwargs)

    def sync_persist_metrics(metrics: dict[str, Any]) -> None:
        inference_mod.persist_metrics.run(metrics)

    def sync_dispatch(self: Orchestrator, payload: dict[str, Any]) -> None:
        dispatch_index = len(processed_results)
        assert dispatch_index < len(stimuli), "orchestrator dispatched beyond fixture truth"
        stimulus = stimuli[dispatch_index]
        tolerance = float(proof["reward_input_tolerance"])
        expected_stimulus_time = EPOCH_S + (
            int(stimulus["segment_index"]) * SEGMENT_WINDOW_SECONDS
            + float(stimulus["stimulus_offset_s"])
        )

        assert payload["_active_arm"] == stimulus["expected_arm_id"]
        assert payload["_expected_greeting"] == stimulus["expected_greeting_text"]
        assert payload["_stimulus_time"] == pytest.approx(expected_stimulus_time, abs=tolerance)
        assert payload["_au12_series"], "each encounter must carry reward AU12 inputs"
        dispatched_payloads.append(dict(payload))

        current_stimulus["row"] = stimulus
        payload_for_inference = dict(payload)
        # Reward proof lives in orchestrator-provided AU12 series. Keeping the
        # representative frame out avoids loading production MediaPipe in CI.
        payload_for_inference["_frame_data"] = None
        result = inference_mod.process_segment.run(payload_for_inference)
        processed_results.append(result)

        if len(processed_results) < len(stimuli):
            self._select_experiment_arm()
        else:
            on_last_dispatch(self)

    monkeypatch.setattr(analytics_mod, "MetricsStore", _fake_metrics_store_class(state))
    monkeypatch.setattr(reward_mod, "compute_reward", observing_compute_reward)
    monkeypatch.setattr(
        "packages.ml_core.transcription.TranscriptionEngine",
        FixtureTranscriptionEngine,
    )
    monkeypatch.setattr("packages.ml_core.semantic.SemanticEvaluator", FixtureSemanticEvaluator)
    monkeypatch.setattr("packages.ml_core.preprocessing.TextPreprocessor", FixtureTextPreprocessor)
    monkeypatch.setattr(
        "packages.ml_core.face_mesh.FaceMeshProcessor",
        _ScriptedFaceMeshProcessor,
    )
    monkeypatch.setattr("packages.ml_core.au12.AU12Normalizer", _ScriptedAU12Normalizer)
    monkeypatch.setattr(inference_mod.persist_metrics, "delay", sync_persist_metrics)
    monkeypatch.setattr(Orchestrator, "_dispatch_payload", sync_dispatch)

    rng = np.random.default_rng(int(script["seed"]))
    ts_draw_inputs: list[tuple[float, float]] = []

    def seeded_beta_rvs(alpha: Any, beta: Any, *args: Any, **kwargs: Any) -> float:
        assert args == ()
        assert not kwargs
        alpha_f = float(alpha)
        beta_f = float(beta)
        ts_draw_inputs.append((alpha_f, beta_f))
        return float(rng.beta(alpha_f, beta_f))

    monkeypatch.setattr("scipy.stats.beta.rvs", seeded_beta_rvs)
    cast(Any, _install_worker_seams).ts_draw_inputs = ts_draw_inputs


def _patch_orchestrator_persistence(
    monkeypatch: pytest.MonkeyPatch,
    state: _InMemoryE2EState,
) -> None:
    def register_session(self: Orchestrator) -> None:
        state.register_session(self._session_id, self._stream_url)

    def mark_session_ended(self: Orchestrator, session_id: str) -> None:
        state.end_session(session_id)
        self.stop()

    def skip_session_lifecycle_listener(self: Orchestrator) -> None:
        del self

    monkeypatch.setattr(Orchestrator, "_register_session", register_session)
    monkeypatch.setattr(Orchestrator, "_mark_session_ended", mark_session_ended)
    monkeypatch.setattr(
        Orchestrator,
        "_start_session_lifecycle_listener",
        skip_session_lifecycle_listener,
    )


class _ReplayRunDriver:
    """Stimulus driver that lets Orchestrator.run own segment dispatch."""

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.proof = script["posterior_proof"]
        self.stimuli = list(script["stimuli"])
        self.fps = int(script["fps"])
        self.next_stimulus_index = 0
        self.current_time_s = EPOCH_S
        self.recorded_stimuli: list[dict[str, Any]] = []
        self.calibration_completed_before_first_stimulus = False

    def time(self) -> float:
        return self.current_time_s

    async def sleep(self, delay: float) -> None:
        del delay

    def load_blank_video_frames(
        self,
        source: ReplayCaptureSource,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return np.zeros((1, source.height, source.width, 3), dtype=np.uint8)

    def process_video_frame(self, orchestrator: Orchestrator) -> None:
        replay = orchestrator.video_capture
        elapsed_s = float(getattr(replay, "_sequential_frame_index", 0)) / self.fps
        self.current_time_s = EPOCH_S + elapsed_s
        self._record_due_stimulus(orchestrator, replay, elapsed_s)
        self._set_scripted_intensity(replay, elapsed_s)
        self.current_time_s = EPOCH_S + elapsed_s
        self._real_process_video_frame(orchestrator)

    def _record_due_stimulus(
        self,
        orchestrator: Orchestrator,
        replay: ReplayCaptureSource,
        elapsed_s: float,
    ) -> None:
        if self.next_stimulus_index >= len(self.stimuli):
            return

        stimulus = self.stimuli[self.next_stimulus_index]
        stimulus_elapsed_s = replay.elapsed_for_stimulus(stimulus)
        if elapsed_s < stimulus_elapsed_s:
            return

        if self.next_stimulus_index == 0:
            assert orchestrator._is_calibrating is True
            assert orchestrator._stimulus_time is None
            assert orchestrator._au12_series == []
            assert orchestrator._calibration_frames_accumulated() >= CALIBRATION_FRAMES_REQUIRED
            self.calibration_completed_before_first_stimulus = True

        assert orchestrator._active_arm == stimulus["expected_arm_id"]
        assert orchestrator._expected_greeting == stimulus["expected_greeting_text"]
        self.current_time_s = EPOCH_S + stimulus_elapsed_s
        orchestrator.record_stimulus_injection()
        assert orchestrator._is_calibrating is False
        assert orchestrator._stimulus_time == pytest.approx(
            EPOCH_S + stimulus_elapsed_s,
            abs=float(self.proof["reward_input_tolerance"]),
        )
        self.recorded_stimuli.append(stimulus)
        self.next_stimulus_index += 1

    def _set_scripted_intensity(self, replay: ReplayCaptureSource, elapsed_s: float) -> None:
        _ScriptedAU12Driver.current_intensity = 0.0
        active_index = self.next_stimulus_index - 1
        if active_index < 0:
            return

        stimulus = self.stimuli[active_index]
        stimulus_elapsed_s = replay.elapsed_for_stimulus(stimulus)
        window_start_s = stimulus_elapsed_s + float(self.proof["stimulus_window_start_offset_s"])
        window_end_s = stimulus_elapsed_s + float(self.proof["stimulus_window_end_offset_s"])
        if window_start_s <= elapsed_s <= window_end_s:
            _ScriptedAU12Driver.current_intensity = float(stimulus["expected_peak_au12"])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._real_process_video_frame = Orchestrator._process_video_frame

        def load_blank_video_frames(
            source: ReplayCaptureSource,
        ) -> np.ndarray[Any, np.dtype[np.uint8]]:
            return self.load_blank_video_frames(source)

        def process_video_frame(orchestrator: Orchestrator) -> None:
            self.process_video_frame(orchestrator)

        monkeypatch.setattr(
            "services.worker.pipeline.orchestrator.time.time",
            self.time,
        )
        monkeypatch.setattr(
            "services.worker.pipeline.orchestrator.asyncio.sleep",
            self.sleep,
        )
        monkeypatch.setattr(ReplayCaptureSource, "_load_video_frames", load_blank_video_frames)
        monkeypatch.setattr(Orchestrator, "_process_video_frame", process_video_frame)


def _posterior_mean(arm_row: dict[str, Any]) -> float:
    alpha = float(arm_row["alpha_param"])
    beta = float(arm_row["beta_param"])
    return alpha / (alpha + beta)


@pytest.mark.integration
def test_baseline_replay_e2e_proves_strong_arm_posterior_improves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_baseline_script()
    proof = script["posterior_proof"]
    strong_arm = str(proof["strong_arm_id"])
    weak_arm = str(proof["weak_arm_id"])
    stimuli = list(script["stimuli"])
    assert len(stimuli) >= int(proof["minimum_encounters"])
    assert {strong_arm, weak_arm} <= {str(row["expected_arm_id"]) for row in stimuli}

    state = _InMemoryE2EState()
    app = _build_api(state, monkeypatch)
    client = TestClient(app)

    create_experiment_response = client.post(
        "/api/v1/experiments",
        json={
            "experiment_id": EXPERIMENT_ID,
            "label": "Baseline operability deterministic two-arm proof",
            "arms": [
                {
                    "arm": strong_arm,
                    "greeting_text": "Hey! Thanks for streaming, you're awesome!",
                },
                {
                    "arm": weak_arm,
                    "greeting_text": "Hi! What's the best advice you've gotten today?",
                },
            ],
        },
    )
    assert create_experiment_response.status_code == 201
    assert state.arm_order[EXPERIMENT_ID] == [strong_arm, weak_arm]

    assert float(script["segment_duration_s"]) == pytest.approx(float(SEGMENT_WINDOW_SECONDS))
    assert float(script["duration_s"]) == pytest.approx(
        len(stimuli) * float(SEGMENT_WINDOW_SECONDS)
    )

    initial_experiment_response = client.get(f"/api/v1/experiments/{EXPERIMENT_ID}")
    assert initial_experiment_response.status_code == 200
    initial_arms = {arm["arm"]: arm for arm in initial_experiment_response.json()["arms"]}
    for arm_id in (strong_arm, weak_arm):
        assert float(initial_arms[arm_id]["alpha_param"]) == pytest.approx(1.0)
        assert float(initial_arms[arm_id]["beta_param"]) == pytest.approx(1.0)

    fixture_dir = _materialize_baseline_replay_fixture(tmp_path, script)
    monkeypatch.setenv("REPLAY_CAPTURE_FIXTURE", str(fixture_dir))
    monkeypatch.setenv("REPLAY_CAPTURE_REALTIME", "0")

    start_response = client.post(
        "/api/v1/sessions",
        json={
            "stream_url": STREAM_URL,
            "experiment_id": EXPERIMENT_ID,
            "client_action_id": str(uuid.uuid4()),
        },
    )
    assert start_response.status_code == 200
    session_id = start_response.json()["session_id"]
    assert state.lifecycle_intents[-1]["action"] == "start"

    observed_reward_calls: list[dict[str, Any]] = []
    processed_results: list[dict[str, Any]] = []
    dispatched_payloads: list[dict[str, Any]] = []
    end_responses: list[dict[str, Any]] = []

    def request_end_from_run(orchestrator_for_end: Orchestrator) -> None:
        # The run loop has just dispatched the final full 30-second segment.
        # Clear chunk-rounding residue so lifecycle end does not flush an extra
        # partial proof segment unrelated to fixture truth.
        orchestrator_for_end._audio_buffer.clear()
        end_response = client.post(
            f"/api/v1/sessions/{session_id}/end",
            json={"client_action_id": str(uuid.uuid4())},
        )
        assert end_response.status_code == 200
        end_responses.append(end_response.json())
        assert state.lifecycle_intents[-1]["action"] == "end"
        orchestrator_for_end._session_lifecycle_queue.put(state.lifecycle_intents[-1])

    _install_worker_seams(
        monkeypatch,
        state,
        script,
        observed_reward_calls,
        processed_results,
        dispatched_payloads,
        request_end_from_run,
    )
    _patch_orchestrator_persistence(monkeypatch, state)
    driver = _ReplayRunDriver(script)
    driver.install(monkeypatch)

    orchestrator = Orchestrator(
        stream_url=STREAM_URL,
        session_id=session_id,
        experiment_id=EXPERIMENT_ID,
    )
    orchestrator._redis = None
    assert isinstance(orchestrator.video_capture, ReplayCaptureSource)
    assert orchestrator.audio_resampler is orchestrator.video_capture
    assert orchestrator._using_replay_capture is True
    replay = orchestrator.video_capture
    assert replay.segment_duration_s == pytest.approx(float(SEGMENT_WINDOW_SECONDS))

    try:
        asyncio.run(orchestrator.run())
    finally:
        orchestrator.stop()

    assert end_responses and end_responses[-1]["action"] == "end"
    assert driver.calibration_completed_before_first_stimulus is True
    assert driver.recorded_stimuli == stimuli
    assert len(dispatched_payloads) == len(stimuli)
    assert len(processed_results) == len(stimuli)
    assert len(observed_reward_calls) == len(stimuli)
    assert len(state.encounters) == len(stimuli)
    assert len(state.metrics) == len(stimuli)

    ts_draw_inputs = cast(Any, _install_worker_seams).ts_draw_inputs
    assert ts_draw_inputs[: len(state.arm_order[EXPERIMENT_ID])] == [
        (1.0, 1.0),
        (1.0, 1.0),
    ]
    assert len(ts_draw_inputs) == len(stimuli) * len(state.arm_order[EXPERIMENT_ID])
    assert any(draw != (1.0, 1.0) for draw in ts_draw_inputs[2:])

    session_response = client.get(f"/api/v1/sessions/{session_id}")
    assert session_response.status_code == 200
    assert session_response.json()["ended_at"] is not None

    tolerance = float(proof["reward_input_tolerance"])

    from services.worker.pipeline import reward as reward_mod

    for stimulus, metrics, reward_call, encounter in zip(
        stimuli,
        state.metrics,
        observed_reward_calls,
        state.encounters,
        strict=True,
    ):
        expected_stimulus_time = EPOCH_S + replay.elapsed_for_stimulus(stimulus)
        assert metrics["_active_arm"] == stimulus["expected_arm_id"]
        assert metrics["_expected_greeting"] == stimulus["expected_greeting_text"]
        assert reward_call["stimulus_time_s"] == pytest.approx(
            expected_stimulus_time,
            abs=tolerance,
        )
        assert reward_call["is_match"] is bool(stimulus["expected_semantic_match"])
        assert reward_call["confidence_score_present"] is False
        assert reward_call["x_max_present"] is False
        assert metrics["semantic"]["confidence_score"] == pytest.approx(
            float(stimulus["expected_confidence_score"]),
            abs=tolerance,
        )
        assert metrics["_x_max"] == stimulus["expected_x_max"]

        au12_window = reward_mod.extract_stimulus_window(
            reward_call["au12_series"],
            reward_call["stimulus_time_s"],
        )
        assert len(au12_window) >= 10
        assert [point.intensity for point in au12_window] == pytest.approx(
            [float(stimulus["expected_peak_au12"])] * len(au12_window),
            abs=tolerance,
        )
        assert encounter["arm"] == stimulus["expected_arm_id"]
        assert encounter["p90_intensity"] == pytest.approx(
            float(stimulus["expected_p90_intensity"]),
            abs=tolerance,
        )
        assert encounter["gated_reward"] == pytest.approx(
            float(stimulus["expected_reward"]),
            abs=tolerance,
        )
        assert encounter["semantic_gate"] == int(bool(stimulus["expected_semantic_match"]))
        assert encounter["is_valid"] is True
        assert encounter["n_frames"] == len(au12_window)

    experiment_response = client.get(f"/api/v1/experiments/{EXPERIMENT_ID}")
    assert experiment_response.status_code == 200
    arms = {arm["arm"]: arm for arm in experiment_response.json()["arms"]}
    strong_mean = _posterior_mean(arms[strong_arm])
    weak_mean = _posterior_mean(arms[weak_arm])
    assert strong_mean > weak_mean
    assert strong_mean - weak_mean >= float(proof["posterior_mean_margin_min"])
    assert float(arms[strong_arm]["alpha_param"]) > float(arms[weak_arm]["alpha_param"])
    assert float(arms[weak_arm]["beta_param"]) > 1.0
