from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import json
import math
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Self, cast
from unittest.mock import MagicMock, patch

import pytest

from packages.ml_core.acoustic import AcousticMetrics
from packages.schemas import inference_handoff as inference_handoff_schema
from packages.schemas.inference_handoff import InferenceHandoffPayload
from packages.schemas.physiology import PhysiologicalChunkEvent
from services.api.routes import physiology as physiology_route
from services.api.services.oura_hydration_service import OuraHydrationService
from services.worker.pipeline.analytics import MetricsStore
from services.worker.pipeline.orchestrator import (
    PHYSIO_STALENESS_THRESHOLD_S,
    Orchestrator,
)
from services.worker.pipeline.reward import TimestampedAU12, compute_reward

DRAFT_07_SCHEMA_URI = "http://json-schema.org/draft-07/schema#"
PCM_AUDIO = b"\x00\x01" * 64
STIMULUS_TIME_S = 100.0
STREAMER_RMSSD_SERIES = [30.0, 40.0, 50.0, 60.0]
OPERATOR_RMSSD_SERIES = [15.0, 20.0, 25.0, 30.0]


def _assert_canonical_acoustic_payload(payload: dict[str, Any]) -> None:
    """Assert the canonical §7D acoustic payload at the Module D → E boundary."""
    assert payload["f0_valid_measure"] is True
    assert payload["f0_valid_baseline"] is True
    assert payload["perturbation_valid_measure"] is True
    assert payload["perturbation_valid_baseline"] is True
    assert payload["voiced_coverage_measure_s"] == pytest.approx(2.5)
    assert payload["voiced_coverage_baseline_s"] == pytest.approx(2.0)
    assert payload["f0_mean_measure_hz"] == pytest.approx(220.0)
    assert payload["f0_mean_baseline_hz"] == pytest.approx(180.0)
    assert payload["f0_delta_semitones"] == pytest.approx(12.0 * math.log2(220.0 / 180.0))
    assert payload["jitter_mean_measure"] == pytest.approx(0.010)
    assert payload["jitter_mean_baseline"] == pytest.approx(0.008)
    assert payload["jitter_delta"] == pytest.approx(0.002)
    assert payload["shimmer_mean_measure"] == pytest.approx(0.020)
    assert payload["shimmer_mean_baseline"] == pytest.approx(0.018)
    assert payload["shimmer_delta"] == pytest.approx(0.002)

    # Legacy scalar dual-write remains optional/deprecated.
    if payload.get("pitch_f0") is not None:
        assert payload["pitch_f0"] == pytest.approx(220.0)
    if payload.get("jitter") is not None:
        assert payload["jitter"] == pytest.approx(0.010)
    if payload.get("shimmer") is not None:
        assert payload["shimmer"] == pytest.approx(0.020)


def _assert_null_canonical_acoustic_payload(payload: dict[str, Any]) -> None:
    """Assert deterministic false/null §7D outputs for null or locally failed acoustics."""
    assert payload["f0_valid_measure"] is False
    assert payload["f0_valid_baseline"] is False
    assert payload["perturbation_valid_measure"] is False
    assert payload["perturbation_valid_baseline"] is False
    assert payload["voiced_coverage_measure_s"] == pytest.approx(0.0)
    assert payload["voiced_coverage_baseline_s"] == pytest.approx(0.0)
    assert payload["f0_mean_measure_hz"] is None
    assert payload["f0_mean_baseline_hz"] is None
    assert payload["f0_delta_semitones"] is None
    assert payload["jitter_mean_measure"] is None
    assert payload["jitter_mean_baseline"] is None
    assert payload["jitter_delta"] is None
    assert payload["shimmer_mean_measure"] is None
    assert payload["shimmer_mean_baseline"] is None
    assert payload["shimmer_delta"] is None
    if "pitch_f0" in payload:
        assert payload["pitch_f0"] is None
    if "jitter" in payload:
        assert payload["jitter"] is None
    if "shimmer" in payload:
        assert payload["shimmer"] is None


class _FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeRedis:
    """Queue-first fake Redis reusing the repo's existing lpop/rpush test pattern."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}
        self._values: dict[str, str] = {}
        self.rpush_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, bool, int | None]] = []
        self.delete_calls: list[str] = []

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self._values:
            return False
        self._values[key] = value
        return True

    def delete(self, key: str) -> int:
        self.delete_calls.append(key)
        existed = key in self._values
        self._values.pop(key, None)
        return 1 if existed else 0

    def lpop(self, queue_name: str) -> str | None:
        queue = self._lists.setdefault(queue_name, [])
        if not queue:
            return None
        return queue.pop(0)

    def rpush(self, queue_name: str, payload: str) -> None:
        self._lists.setdefault(queue_name, []).append(payload)
        self.rpush_calls.append((queue_name, payload))

    def peek(self, queue_name: str) -> list[str]:
        return list(self._lists.get(queue_name, []))

    def close(self) -> None:
        return None


class _FakeOuraClient:
    def __init__(self, resources: list[dict[str, Any]]) -> None:
        self._resources = list(resources)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, dict(query)))
        if not self._resources:
            raise AssertionError("Unexpected extra hydration fetch")
        return self._resources.pop(0)


class _RecordingCursor:
    def __init__(self, store: _RecordingMetricsStore) -> None:
        self._store = store
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        normalized_sql = " ".join(sql.split())
        if (
            "SELECT subject_role, rmssd_ms, source_timestamp_utc FROM physiology_log"
            in normalized_sql
        ):
            self._rows = self._store.query_recent_physiology(params)
            return
        if "INSERT INTO physiology_log" in normalized_sql:
            self._store.physiology_rows.append(dict(params))
            return
        if "INSERT INTO comodulation_log" in normalized_sql:
            self._store.comodulation_rows.append(dict(params))
            return
        if "INSERT INTO encounter_log" in normalized_sql:
            self._store.encounter_rows.append(dict(params))
            return
        raise AssertionError(f"Unexpected SQL in physiology integration test: {normalized_sql}")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _CursorContextManager:
    def __init__(self, store: _RecordingMetricsStore) -> None:
        self._cursor = _RecordingCursor(store)

    def __enter__(self) -> _RecordingCursor:
        return self._cursor

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


class _RecordingConnection:
    def __init__(self, store: _RecordingMetricsStore) -> None:
        self._store = store
        self.isolation_levels: list[int] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _CursorContextManager:
        return _CursorContextManager(self._store)

    def set_isolation_level(self, level: int) -> None:
        self.isolation_levels.append(level)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _RecordingMetricsStore(MetricsStore):
    """In-memory stand-in that still executes the production physiology methods."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics_rows: list[dict[str, Any]] = []
        self.physiology_rows: list[dict[str, Any]] = []
        self.comodulation_rows: list[dict[str, Any]] = []
        self.encounter_rows: list[dict[str, Any]] = []
        self.arm_updates: list[dict[str, Any]] = []
        self._connection = _RecordingConnection(self)
        self._arms: dict[str, dict[str, float]] = {"arm-a": {"alpha_param": 1.0, "beta_param": 1.0}}

    def connect(self, minconn: int = 2, maxconn: int = 10) -> None:
        del minconn, maxconn

    def close(self) -> None:
        return None

    def _get_conn(self) -> _RecordingConnection:
        return self._connection

    def _put_conn(self, conn: Any) -> None:
        assert conn is self._connection

    def insert_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics_rows.append(dict(metrics))

    def get_experiment_arms(self, experiment_id: str) -> list[dict[str, Any]]:
        return [
            {
                "arm": arm,
                "alpha_param": state["alpha_param"],
                "beta_param": state["beta_param"],
            }
            for arm, state in self._arms.items()
        ]

    def get_experiment_arm(self, experiment_id: str, arm: str) -> dict[str, Any] | None:
        del experiment_id
        state = self._arms.get(arm)
        if state is None:
            return None
        return {
            "arm": arm,
            "alpha_param": state["alpha_param"],
            "beta_param": state["beta_param"],
        }

    def update_experiment_arm(
        self,
        experiment_id: str,
        arm: str,
        alpha: float,
        beta: float,
    ) -> None:
        self._arms[arm] = {"alpha_param": alpha, "beta_param": beta}
        self.arm_updates.append(
            {
                "experiment_id": experiment_id,
                "arm": arm,
                "alpha": alpha,
                "beta": beta,
            }
        )

    def query_recent_physiology(self, params: dict[str, Any]) -> list[tuple[str, float, datetime]]:
        session_id = params["session_id"]
        window_start = params["window_start_utc"]
        window_end = params["window_end_utc"]
        rows: list[tuple[str, float, datetime]] = []
        for row in self.physiology_rows:
            if row["session_id"] != session_id:
                continue
            if (
                row["rmssd_ms"] is None
                or row["is_valid"] is not True
                or row["is_stale"] is not False
            ):
                continue
            source_ts = _parse_dt(row["source_timestamp_utc"])
            if not (window_start < source_ts <= window_end):
                continue
            rows.append((row["subject_role"], row["rmssd_ms"], source_ts))
        rows.sort(key=lambda item: item[2])
        return rows


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_segment_id(payload: dict[str, Any]) -> str:
    stable_identity = (
        f"{uuid.UUID(payload['session_id'])}"
        f"{_iso_z(_parse_dt(payload['segment_window_start_utc']))}"
        f"{_iso_z(_parse_dt(payload['segment_window_end_utc']))}"
    )
    return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _notification_payload(*, role: str, start: datetime, end: datetime) -> dict[str, Any]:
    event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{role}:{_iso_z(start)}:{_iso_z(end)}"))
    return {
        "event_id": event_id,
        "subject_role": role,
        "event_type": "daily_update",
        "data_type": "heartrate",
        "start_datetime": _iso_z(start),
        "end_datetime": _iso_z(end),
    }


def _ibi_resource(*, role: str, start: datetime, end: datetime, rmssd_ms: float) -> dict[str, Any]:
    midpoint = 1000.0 + float(rmssd_ms)
    return {
        "data": [
            {
                "id": f"{role}-{_iso_z(end)}",
                "timestamp": _iso_z(start),
                "end_datetime": _iso_z(end),
                "ibi_ms": [1000.0, midpoint, 1000.0],
                "heart_rate_bpm": [70, 71, 72] if role == "streamer" else [64, 65, 66],
                "sample_interval_s": 1,
            }
        ]
    }


def _reward_telemetry() -> list[dict[str, float]]:
    telemetry: list[dict[str, float]] = []
    fps = 30.0
    for index in range(int(5.0 * fps)):
        telemetry.append(
            {
                "timestamp_s": STIMULUS_TIME_S - 5.0 + (index / fps),
                "intensity": 0.1,
            }
        )
    for index in range(int(6.0 * fps)):
        telemetry.append(
            {
                "timestamp_s": STIMULUS_TIME_S + (index / fps),
                "intensity": 0.8,
            }
        )
    return telemetry


def _rolling_ibi_rmssd(chunk_rmssd_values: list[float]) -> float:
    numerator = 2 * sum(value**2 for value in chunk_rmssd_values)
    denominator = (3 * len(chunk_rmssd_values)) - 1
    return round(math.sqrt(numerator / denominator), 3)


def _fake_ml_modules(
    *,
    acoustic_analyzer_cls: type[Any] | None = None,
) -> dict[str, ModuleType]:
    transcription: Any = ModuleType("packages.ml_core.transcription")

    class TranscriptionEngine:
        def transcribe(self, wav_path: str) -> str:
            assert wav_path.endswith(".wav")
            return "hello welcome to the stream"

    transcription.TranscriptionEngine = TranscriptionEngine

    acoustic: Any = ModuleType("packages.ml_core.acoustic")

    if acoustic_analyzer_cls is None:

        class AcousticAnalyzer:
            def analyze(
                self,
                audio_data: bytes,
                sample_rate: int = 16000,
                *,
                stimulus_time_s: float | None = None,
                segment_start_time_s: float | None = None,
            ) -> Any:
                assert audio_data
                assert sample_rate == 16000
                assert stimulus_time_s is not None
                assert segment_start_time_s is not None
                return AcousticMetrics(
                    pitch_f0=220.0,
                    jitter=0.01,
                    shimmer=0.02,
                    f0_valid_measure=True,
                    f0_valid_baseline=True,
                    perturbation_valid_measure=True,
                    perturbation_valid_baseline=True,
                    voiced_coverage_measure_s=2.5,
                    voiced_coverage_baseline_s=2.0,
                    f0_mean_measure_hz=220.0,
                    f0_mean_baseline_hz=180.0,
                    f0_delta_semitones=12.0 * math.log2(220.0 / 180.0),
                    jitter_mean_measure=0.01,
                    jitter_mean_baseline=0.008,
                    jitter_delta=0.002,
                    shimmer_mean_measure=0.02,
                    shimmer_mean_baseline=0.018,
                    shimmer_delta=0.002,
                )

        acoustic_analyzer_cls = AcousticAnalyzer

    acoustic.AcousticAnalyzer = acoustic_analyzer_cls

    preprocessing: Any = ModuleType("packages.ml_core.preprocessing")

    class TextPreprocessor:
        def preprocess(self, text: str) -> str:
            return text.lower().strip()

    preprocessing.TextPreprocessor = TextPreprocessor

    semantic: Any = ModuleType("packages.ml_core.semantic")

    class SemanticEvaluator:
        def evaluate(self, expected_greeting: str, actual_text: str) -> dict[str, Any]:
            del expected_greeting
            return {
                "reasoning": "integration-test match",
                "is_match": actual_text == "hello welcome to the stream",
                "confidence": 0.91,
                "confidence_score": 0.91,
            }

    semantic.SemanticEvaluator = SemanticEvaluator

    return {
        "packages.ml_core.transcription": transcription,
        "packages.ml_core.acoustic": acoustic,
        "packages.ml_core.preprocessing": preprocessing,
        "packages.ml_core.semantic": semantic,
    }


def _fake_ffmpeg_run(cmd: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
    del args, kwargs
    Path(cmd[-1]).write_bytes(b"RIFF")
    return SimpleNamespace(returncode=0)


def _load_inference_module() -> Any:
    mock_app = MagicMock()
    mock_app.task.return_value = lambda fn: fn
    module_name = "services.worker.tasks.inference"
    worker_pkg = importlib.import_module("services.worker")
    fake_celery_module = ModuleType("services.worker.celery_app")
    cast(Any, fake_celery_module).celery_app = mock_app
    fake_celery = ModuleType("celery")
    cast(Any, fake_celery).Task = type("Task", (), {})

    with (
        patch.dict(
            sys.modules,
            {
                "celery": fake_celery,
                "services.worker.celery_app": fake_celery_module,
            },
        ),
        patch.object(worker_pkg, "celery_app", fake_celery_module, create=True),
    ):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)


def _exercise_module_d_to_e(
    payload: dict[str, Any],
    store: _RecordingMetricsStore,
    *,
    acoustic_analyzer_cls: type[Any] | None = None,
) -> dict[str, Any]:
    inference_mod = _load_inference_module()
    persist_metrics_fn = inference_mod.persist_metrics
    sync_dispatch = SimpleNamespace(delay=lambda metrics: persist_metrics_fn(MagicMock(), metrics))
    fake_psycopg2 = SimpleNamespace(
        extensions=SimpleNamespace(
            ISOLATION_LEVEL_READ_COMMITTED=1,
            ISOLATION_LEVEL_SERIALIZABLE=6,
        )
    )

    with (
        patch.dict(
            sys.modules,
            _fake_ml_modules(acoustic_analyzer_cls=acoustic_analyzer_cls),
            clear=False,
        ),
        patch("subprocess.run", side_effect=_fake_ffmpeg_run),
        patch("services.worker.pipeline.analytics.MetricsStore", return_value=store),
        patch("services.worker.pipeline.analytics._import_psycopg2", return_value=fake_psycopg2),
        patch.object(inference_mod, "persist_metrics", sync_dispatch),
    ):
        return cast(dict[str, Any], inference_mod.process_segment(MagicMock(), payload))


def _invoke_webhook(
    fake_redis: _FakeRedis,
    payload: dict[str, Any],
    *,
    secret: str,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    signature = _sign(body, secret)
    with (
        patch.dict("os.environ", {"OURA_WEBHOOK_SECRET": secret}, clear=False),
        patch.object(physiology_route, "_get_redis", return_value=fake_redis),
    ):
        return asyncio.run(
            physiology_route.oura_webhook(
                cast(Any, _FakeRequest(body)),
                x_oura_signature=signature,
            )
        )


def _run_physiology_cycle(
    *,
    fake_redis: _FakeRedis,
    orchestrator: Orchestrator,
    cycle_end: datetime,
    streamer_rmssd: float,
    operator_rmssd: float,
    secret: str,
) -> tuple[dict[str, Any], list[PhysiologicalChunkEvent]]:
    cycle_start = cycle_end - timedelta(seconds=3)
    streamer_notification = _notification_payload(
        role="streamer",
        start=cycle_start,
        end=cycle_end,
    )
    operator_notification = _notification_payload(
        role="operator",
        start=cycle_start,
        end=cycle_end,
    )

    assert _invoke_webhook(fake_redis, streamer_notification, secret=secret)["status"] == "accepted"
    assert _invoke_webhook(fake_redis, operator_notification, secret=secret)["status"] == "accepted"

    hydrate_items = [json.loads(item) for item in fake_redis.peek("physio:hydrate")]
    assert len(hydrate_items) == 2
    for item in hydrate_items:
        assert sorted(item.keys()) == [
            "data_type",
            "end_datetime",
            "event_type",
            "notification_received_utc",
            "start_datetime",
            "subject_role",
            "unique_id",
        ]

    oura_client = _FakeOuraClient(
        [
            _ibi_resource(
                role="streamer",
                start=cycle_start,
                end=cycle_end,
                rmssd_ms=streamer_rmssd,
            ),
            _ibi_resource(
                role="operator",
                start=cycle_start,
                end=cycle_end,
                rmssd_ms=operator_rmssd,
            ),
        ]
    )
    hydration_service = OuraHydrationService(
        redis_client=fake_redis,
        oura_client=cast(Any, oura_client),
        clock=lambda: cycle_end + timedelta(seconds=1),
    )

    assert hydration_service.drain_once() == 2
    chunk_events = [
        PhysiologicalChunkEvent.model_validate_json(payload)
        for payload in fake_redis.peek("physio:events")
    ]
    assert len(chunk_events) == 2
    assert {event.subject_role for event in chunk_events} == {"streamer", "operator"}
    assert all(event.event_type == "physiological_chunk" for event in chunk_events)

    orchestrator._drain_physio_events()
    orchestrator._au12_series = _reward_telemetry()
    orchestrator._stimulus_time = STIMULUS_TIME_S

    with patch(
        "services.worker.pipeline.orchestrator.time.time",
        return_value=cycle_end.timestamp() + 5.0,
    ):
        payload = orchestrator.assemble_segment(PCM_AUDIO, [])

    return payload, chunk_events


def test_notification_hydration_chunk_derivation_and_scalar_persistence_end_to_end() -> None:
    """Full webhook -> hydration -> chunk -> derivation -> scalar persistence regression."""
    secret = "integration-secret"
    fake_redis = _FakeRedis()
    session_id = "00000000-0000-4000-8000-000000000123"
    store = _RecordingMetricsStore()
    orchestrator = Orchestrator(session_id=session_id, experiment_id="exp-1")
    orchestrator._redis = fake_redis
    orchestrator._active_arm = "arm-a"
    orchestrator._expected_greeting = "hello welcome to the stream"

    empty_payload = Orchestrator(session_id=session_id, experiment_id="exp-1").assemble_segment(
        PCM_AUDIO,
        [],
    )
    schema_empty_payload = {
        key: value
        for key, value in empty_payload.items()
        if key not in {"_audio_data", "_frame_data", "_experiment_code"}
    }
    validated_empty_payload = InferenceHandoffPayload.model_validate(schema_empty_payload)
    expected_physio_event_schema = PhysiologicalChunkEvent.model_json_schema()

    assert hasattr(inference_handoff_schema, "physiological_sample_event_schema")
    exported_physio_event_schema = inference_handoff_schema.physiological_sample_event_schema
    assert "_physiological_context" not in empty_payload
    assert str(validated_empty_payload.session_id) == session_id
    assert exported_physio_event_schema["$schema"] == DRAFT_07_SCHEMA_URI
    if exported_physio_event_schema.get("title") == "PhysiologicalChunkEvent":
        assert exported_physio_event_schema == expected_physio_event_schema
    else:
        assert exported_physio_event_schema["$ref"] == "#/$defs/PhysiologicalChunkEvent"
        assert (
            exported_physio_event_schema["$defs"]["PhysiologicalChunkEvent"]
            == expected_physio_event_schema
        )

    first_cycle_end = datetime.now(UTC).replace(second=3, microsecond=0) - timedelta(minutes=4)
    expected_streamer_snapshots: list[float] = []
    expected_operator_snapshots: list[float] = []

    for index, (streamer_rmssd, operator_rmssd) in enumerate(
        zip(STREAMER_RMSSD_SERIES, OPERATOR_RMSSD_SERIES, strict=True)
    ):
        cycle_end = first_cycle_end + timedelta(minutes=index)
        payload, chunk_events = _run_physiology_cycle(
            fake_redis=fake_redis,
            orchestrator=orchestrator,
            cycle_end=cycle_end,
            streamer_rmssd=streamer_rmssd,
            operator_rmssd=operator_rmssd,
            secret=secret,
        )

        assert payload["session_id"] == session_id
        assert payload["segment_id"] == _expected_segment_id(payload)
        assert "_segment_id" not in payload
        assert payload["_au12_series"] == _reward_telemetry()
        assert payload["_stimulus_time"] == STIMULUS_TIME_S
        assert len(chunk_events) == 2

        expected_streamer = _rolling_ibi_rmssd(STREAMER_RMSSD_SERIES[: index + 1])
        expected_operator = _rolling_ibi_rmssd(OPERATOR_RMSSD_SERIES[: index + 1])
        expected_streamer_snapshots.append(expected_streamer)
        expected_operator_snapshots.append(expected_operator)

        context = payload["_physiological_context"]
        assert context["streamer"]["rmssd_ms"] == pytest.approx(expected_streamer)
        assert context["operator"]["rmssd_ms"] == pytest.approx(expected_operator)
        assert context["streamer"]["source_kind"] == "ibi"
        assert context["operator"]["source_kind"] == "ibi"
        assert context["streamer"]["derivation_method"] == "server"
        assert context["operator"]["derivation_method"] == "server"
        assert context["streamer"]["is_valid"] is True
        assert context["operator"]["is_valid"] is True
        assert context["streamer"]["is_stale"] is False
        assert context["operator"]["is_stale"] is False
        assert context["streamer"]["freshness_s"] == 5.0
        assert context["operator"]["freshness_s"] == 5.0

        result = _exercise_module_d_to_e(payload, store)
        assert result["semantic"]["is_match"] is True
        assert result["transcription"] == "hello welcome to the stream"
        assert result["_physiological_context"] == context
        _assert_canonical_acoustic_payload(result)

        assert len(store.metrics_rows) == index + 1
        _assert_canonical_acoustic_payload(store.metrics_rows[-1])
        assert len(store.physiology_rows) == (index + 1) * 2
        assert len(store.comodulation_rows) == index + 1
        assert len(store.arm_updates) == index + 1
        assert len(store.encounter_rows) == index + 1

        for row in store.physiology_rows[-2:]:
            assert sorted(row.keys()) == [
                "derivation_method",
                "freshness_s",
                "heart_rate_bpm",
                "is_stale",
                "is_valid",
                "provider",
                "rmssd_ms",
                "segment_id",
                "session_id",
                "source_kind",
                "source_timestamp_utc",
                "subject_role",
                "validity_ratio",
                "window_s",
            ]
            assert "payload" not in row
            assert "provider_body" not in row
            assert "data_type" not in row
            assert row["session_id"] == session_id
            assert row["subject_role"] in {"streamer", "operator"}
            assert row["rmssd_ms"] is not None
            assert row["is_stale"] is False

        comod_row = store.comodulation_rows[-1]
        assert sorted(comod_row.keys()) == [
            "co_modulation_index",
            "coverage_ratio",
            "n_paired_observations",
            "operator_rmssd_mean",
            "session_id",
            "streamer_rmssd_mean",
            "window_end_utc",
            "window_minutes",
            "window_start_utc",
        ]
        assert comod_row["session_id"] == session_id
        assert comod_row["window_minutes"] == 10
        assert comod_row["n_paired_observations"] == index + 1
        if index < 3:
            assert comod_row["co_modulation_index"] is None
        else:
            assert comod_row["co_modulation_index"] == pytest.approx(1.0)
            assert comod_row["coverage_ratio"] == pytest.approx(1.0)
            assert comod_row["streamer_rmssd_mean"] == pytest.approx(
                sum(expected_streamer_snapshots) / len(expected_streamer_snapshots)
            )
            assert comod_row["operator_rmssd_mean"] == pytest.approx(
                sum(expected_operator_snapshots) / len(expected_operator_snapshots)
            )

    pre_stale_comod_count = len(store.comodulation_rows)
    stale_now = first_cycle_end.timestamp() + (3 * 60) + PHYSIO_STALENESS_THRESHOLD_S + 5
    orchestrator._au12_series = _reward_telemetry()
    orchestrator._stimulus_time = STIMULUS_TIME_S

    with patch("services.worker.pipeline.orchestrator.time.time", return_value=stale_now):
        stale_payload = orchestrator.assemble_segment(PCM_AUDIO, [])

    stale_context = stale_payload["_physiological_context"]
    assert stale_context["streamer"]["is_stale"] is True
    assert stale_context["operator"]["is_stale"] is True
    assert stale_context["streamer"]["rmssd_ms"] is None
    assert stale_context["operator"]["rmssd_ms"] is None
    assert stale_context["streamer"]["freshness_s"] > PHYSIO_STALENESS_THRESHOLD_S
    assert stale_context["operator"]["freshness_s"] > PHYSIO_STALENESS_THRESHOLD_S

    stale_result = _exercise_module_d_to_e(stale_payload, store)
    assert stale_result["_physiological_context"] == stale_context
    _assert_canonical_acoustic_payload(stale_result)
    assert len(store.metrics_rows) == 5
    _assert_canonical_acoustic_payload(store.metrics_rows[-1])
    assert len(store.physiology_rows) == 10
    assert len(store.comodulation_rows) == pre_stale_comod_count
    assert store.physiology_rows[-1]["is_stale"] is True
    assert store.physiology_rows[-2]["is_stale"] is True
    assert store.physiology_rows[-1]["rmssd_ms"] is None
    assert store.physiology_rows[-2]["rmssd_ms"] is None


def test_reward_pipeline_is_invariant_with_optional_physiology_context() -> None:
    """§13.17 / §13.21 — payloads without physiology stay valid and TS reward is unchanged."""
    session_id = "00000000-0000-4000-8000-000000000321"

    plain_orchestrator = Orchestrator(session_id=session_id, experiment_id="exp-1")
    plain_orchestrator._active_arm = "arm-a"
    plain_orchestrator._expected_greeting = "hello welcome to the stream"
    plain_orchestrator._au12_series = _reward_telemetry()
    plain_orchestrator._stimulus_time = STIMULUS_TIME_S
    payload_without_physio = plain_orchestrator.assemble_segment(PCM_AUDIO, [])
    assert "_physiological_context" not in payload_without_physio

    physio_store = _RecordingMetricsStore()
    plain_store = _RecordingMetricsStore()

    physio_redis = _FakeRedis()
    physio_orchestrator = Orchestrator(session_id=session_id, experiment_id="exp-1")
    physio_orchestrator._redis = physio_redis
    physio_orchestrator._active_arm = "arm-a"
    physio_orchestrator._expected_greeting = "hello welcome to the stream"

    payload_with_physio, _ = _run_physiology_cycle(
        fake_redis=physio_redis,
        orchestrator=physio_orchestrator,
        cycle_end=datetime.now(UTC).replace(second=3, microsecond=0) - timedelta(minutes=1),
        streamer_rmssd=STREAMER_RMSSD_SERIES[0],
        operator_rmssd=OPERATOR_RMSSD_SERIES[0],
        secret="reward-secret",
    )
    assert payload_with_physio["_physiological_context"]["streamer"] is not None
    assert payload_with_physio["_physiological_context"]["operator"] is not None

    result_without_physio = _exercise_module_d_to_e(payload_without_physio, plain_store)
    result_with_physio = _exercise_module_d_to_e(payload_with_physio, physio_store)
    _assert_canonical_acoustic_payload(result_without_physio)
    _assert_canonical_acoustic_payload(result_with_physio)
    _assert_canonical_acoustic_payload(plain_store.metrics_rows[0])
    _assert_canonical_acoustic_payload(physio_store.metrics_rows[0])

    expected_reward = compute_reward(
        au12_series=[TimestampedAU12(**point) for point in _reward_telemetry()],
        stimulus_time_s=STIMULUS_TIME_S,
        is_match=True,
        confidence_score=0.91,
        x_max=None,
    )

    assert result_without_physio["semantic"]["is_match"] is True
    assert result_with_physio["semantic"]["is_match"] is True
    assert "_physiological_context" not in plain_store.metrics_rows[0]
    assert (
        physio_store.metrics_rows[0]["_physiological_context"]
        == result_with_physio["_physiological_context"]
    )
    assert plain_store.arm_updates == [
        {
            "experiment_id": "exp-1",
            "arm": "arm-a",
            "alpha": pytest.approx(1.0 + expected_reward.gated_reward),
            "beta": pytest.approx(2.0 - expected_reward.gated_reward),
        }
    ]
    assert physio_store.arm_updates == [
        {
            "experiment_id": "exp-1",
            "arm": "arm-a",
            "alpha": pytest.approx(1.0 + expected_reward.gated_reward),
            "beta": pytest.approx(2.0 - expected_reward.gated_reward),
        }
    ]
    assert plain_store.encounter_rows[0]["gated_reward"] == pytest.approx(
        expected_reward.gated_reward
    )
    assert physio_store.encounter_rows[0]["gated_reward"] == pytest.approx(
        expected_reward.gated_reward
    )
    assert len(plain_store.physiology_rows) == 0
    assert len(physio_store.physiology_rows) == 2


def test_acoustic_invalidity_is_local_and_reward_is_preserved_end_to_end() -> None:
    """Local acoustic invalidity degrades only §7D outputs; reward math remains unchanged."""
    session_id = "00000000-0000-4000-8000-000000000654"
    orchestrator = Orchestrator(session_id=session_id, experiment_id="exp-1")
    orchestrator._active_arm = "arm-a"
    orchestrator._expected_greeting = "hello welcome to the stream"
    orchestrator._au12_series = _reward_telemetry()
    orchestrator._stimulus_time = STIMULUS_TIME_S
    payload = orchestrator.assemble_segment(PCM_AUDIO, [])
    expected_reward = compute_reward(
        au12_series=[TimestampedAU12(**point) for point in _reward_telemetry()],
        stimulus_time_s=STIMULUS_TIME_S,
        is_match=True,
        confidence_score=0.91,
        x_max=None,
    )

    class SilentAcousticAnalyzer:
        def analyze(
            self,
            audio_data: bytes,
            sample_rate: int = 16000,
            *,
            stimulus_time_s: float | None = None,
            segment_start_time_s: float | None = None,
        ) -> Any:
            assert audio_data
            assert sample_rate == 16000
            assert stimulus_time_s == STIMULUS_TIME_S
            assert segment_start_time_s is not None
            return AcousticMetrics()

    class PerturbationInvalidAcousticAnalyzer:
        def analyze(
            self,
            audio_data: bytes,
            sample_rate: int = 16000,
            *,
            stimulus_time_s: float | None = None,
            segment_start_time_s: float | None = None,
        ) -> Any:
            assert audio_data
            assert sample_rate == 16000
            assert stimulus_time_s == STIMULUS_TIME_S
            assert segment_start_time_s is not None
            return AcousticMetrics(
                pitch_f0=210.0,
                jitter=0.010,
                shimmer=0.020,
                f0_valid_measure=True,
                f0_valid_baseline=True,
                voiced_coverage_measure_s=2.4,
                voiced_coverage_baseline_s=2.0,
                f0_mean_measure_hz=220.0,
                f0_mean_baseline_hz=180.0,
                f0_delta_semitones=12.0 * math.log2(220.0 / 180.0),
            )

    class FailingAcousticAnalyzer:
        def analyze(
            self,
            audio_data: bytes,
            sample_rate: int = 16000,
            *,
            stimulus_time_s: float | None = None,
            segment_start_time_s: float | None = None,
        ) -> Any:
            assert audio_data
            assert sample_rate == 16000
            assert stimulus_time_s == STIMULUS_TIME_S
            assert segment_start_time_s is not None
            raise RuntimeError("Praat returned undefined perturbation values")

    cases = [
        ("silence", SilentAcousticAnalyzer),
        ("perturbation_invalidity", PerturbationInvalidAcousticAnalyzer),
        ("exception", FailingAcousticAnalyzer),
    ]

    for label, acoustic_analyzer_cls in cases:
        store = _RecordingMetricsStore()
        result = _exercise_module_d_to_e(
            payload,
            store,
            acoustic_analyzer_cls=acoustic_analyzer_cls,
        )

        assert result["semantic"]["is_match"] is True
        assert result["transcription"] == "hello welcome to the stream"
        assert len(store.metrics_rows) == 1
        assert len(store.arm_updates) == 1
        assert len(store.encounter_rows) == 1

        if label == "perturbation_invalidity":
            assert result["f0_valid_measure"] is True
            assert result["f0_valid_baseline"] is True
            assert result["perturbation_valid_measure"] is False
            assert result["perturbation_valid_baseline"] is False
            assert result["voiced_coverage_measure_s"] == pytest.approx(2.4)
            assert result["voiced_coverage_baseline_s"] == pytest.approx(2.0)
            assert result["f0_mean_measure_hz"] == pytest.approx(220.0)
            assert result["f0_mean_baseline_hz"] == pytest.approx(180.0)
            assert result["f0_delta_semitones"] == pytest.approx(12.0 * math.log2(220.0 / 180.0))
            assert result["jitter_mean_measure"] is None
            assert result["jitter_mean_baseline"] is None
            assert result["jitter_delta"] is None
            assert result["shimmer_mean_measure"] is None
            assert result["shimmer_mean_baseline"] is None
            assert result["shimmer_delta"] is None
            assert result["pitch_f0"] == pytest.approx(210.0)
            assert result["jitter"] == pytest.approx(0.010)
            assert result["shimmer"] == pytest.approx(0.020)
            assert store.metrics_rows[0]["perturbation_valid_measure"] is False
            assert store.metrics_rows[0]["jitter_mean_measure"] is None
            assert store.metrics_rows[0]["shimmer_mean_measure"] is None
        else:
            _assert_null_canonical_acoustic_payload(result)
            _assert_null_canonical_acoustic_payload(store.metrics_rows[0])

        assert store.arm_updates == [
            {
                "experiment_id": "exp-1",
                "arm": "arm-a",
                "alpha": pytest.approx(1.0 + expected_reward.gated_reward),
                "beta": pytest.approx(2.0 - expected_reward.gated_reward),
            }
        ]
        assert store.encounter_rows[0]["gated_reward"] == pytest.approx(
            expected_reward.gated_reward
        )
        assert len(store.physiology_rows) == 0


def _frozen_analytics_datetime(frozen_now: datetime) -> type[datetime]:
    class FrozenAnalyticsDateTime(datetime):
        @classmethod
        def now(cls, tz: Any | None = None) -> Self:
            if tz is None:
                return cls.fromtimestamp(frozen_now.timestamp())
            return cls.fromtimestamp(frozen_now.timestamp(), tz=UTC)

    return FrozenAnalyticsDateTime


def _run_physiology_sequence_through_module_e(
    *,
    session_id: str,
    secret: str,
    first_cycle_end: datetime,
    streamer_rmssd_series: list[float],
    operator_rmssd_series: list[float],
) -> _RecordingMetricsStore:
    fake_redis = _FakeRedis()
    store = _RecordingMetricsStore()
    orchestrator = Orchestrator(session_id=session_id, experiment_id="exp-1")
    orchestrator._redis = fake_redis
    orchestrator._active_arm = "arm-a"
    orchestrator._expected_greeting = "hello welcome to the stream"

    for index, (streamer_rmssd, operator_rmssd) in enumerate(
        zip(streamer_rmssd_series, operator_rmssd_series, strict=True)
    ):
        cycle_end = first_cycle_end + timedelta(minutes=index)
        payload, _ = _run_physiology_cycle(
            fake_redis=fake_redis,
            orchestrator=orchestrator,
            cycle_end=cycle_end,
            streamer_rmssd=streamer_rmssd,
            operator_rmssd=operator_rmssd,
            secret=secret,
        )
        _exercise_module_d_to_e(payload, store)

    return store


def test_comodulation_returns_null_for_zero_variance_series_end_to_end() -> None:
    """§13.20 — four aligned valid bins with zero variance persist a null co-mod row."""
    analytics_now = datetime.now(UTC).replace(second=0, microsecond=0)
    first_cycle_end = analytics_now.replace(second=3, microsecond=0) - timedelta(minutes=4)
    zero_variance_streamer_series = [30.0] + ([30.0 * math.sqrt(1.5)] * 3)
    varying_operator_series = [15.0, 20.0, 25.0, 30.0]
    expected_operator_snapshots = [
        _rolling_ibi_rmssd(varying_operator_series[: index + 1]) for index in range(4)
    ]

    with patch(
        "services.worker.pipeline.analytics.datetime",
        _frozen_analytics_datetime(analytics_now),
    ):
        store = _run_physiology_sequence_through_module_e(
            session_id="00000000-0000-4000-8000-000000000420",
            secret="zero-variance-secret",
            first_cycle_end=first_cycle_end,
            streamer_rmssd_series=zero_variance_streamer_series,
            operator_rmssd_series=varying_operator_series,
        )

    streamer_snapshots = [
        row["rmssd_ms"] for row in store.physiology_rows if row["subject_role"] == "streamer"
    ]
    operator_snapshots = [
        row["rmssd_ms"] for row in store.physiology_rows if row["subject_role"] == "operator"
    ]
    final_comod_row = store.comodulation_rows[-1]

    assert len(store.physiology_rows) == 8
    assert len(store.comodulation_rows) == 4
    assert streamer_snapshots == pytest.approx([30.0, 30.0, 30.0, 30.0])
    assert operator_snapshots == pytest.approx(expected_operator_snapshots)
    assert final_comod_row["session_id"] == "00000000-0000-4000-8000-000000000420"
    assert final_comod_row["window_start_utc"] == analytics_now - timedelta(minutes=10)
    assert final_comod_row["window_end_utc"] == analytics_now
    assert final_comod_row["window_minutes"] == 10
    assert final_comod_row["co_modulation_index"] is None
    assert final_comod_row["n_paired_observations"] == 4
    assert final_comod_row["coverage_ratio"] == pytest.approx(1.0)
    assert final_comod_row["streamer_rmssd_mean"] == pytest.approx(30.0)
    assert final_comod_row["operator_rmssd_mean"] == pytest.approx(
        sum(expected_operator_snapshots) / len(expected_operator_snapshots)
    )


def test_comodulation_is_deterministic_and_routes_through_pearsonr_end_to_end() -> None:
    """§13.20 — identical physiology input persists identical co-mod rows.

    The integration path must route the correlation through scipy.stats.pearsonr.
    """
    from scipy.stats import pearsonr as scipy_pearsonr

    analytics_now = datetime.now(UTC).replace(second=0, microsecond=0)
    first_cycle_end = analytics_now.replace(second=3, microsecond=0) - timedelta(minutes=4)
    expected_streamer_snapshots = [
        _rolling_ibi_rmssd(STREAMER_RMSSD_SERIES[: index + 1]) for index in range(4)
    ]
    expected_operator_snapshots = [
        _rolling_ibi_rmssd(OPERATOR_RMSSD_SERIES[: index + 1]) for index in range(4)
    ]

    with (
        patch(
            "services.worker.pipeline.analytics.datetime",
            _frozen_analytics_datetime(analytics_now),
        ),
        patch("scipy.stats.pearsonr", wraps=scipy_pearsonr) as pearsonr_spy,
    ):
        first_store = _run_physiology_sequence_through_module_e(
            session_id="00000000-0000-4000-8000-000000000421",
            secret="determinism-secret",
            first_cycle_end=first_cycle_end,
            streamer_rmssd_series=STREAMER_RMSSD_SERIES,
            operator_rmssd_series=OPERATOR_RMSSD_SERIES,
        )
        second_store = _run_physiology_sequence_through_module_e(
            session_id="00000000-0000-4000-8000-000000000421",
            secret="determinism-secret",
            first_cycle_end=first_cycle_end,
            streamer_rmssd_series=STREAMER_RMSSD_SERIES,
            operator_rmssd_series=OPERATOR_RMSSD_SERIES,
        )

    first_rows = [dict(row) for row in first_store.comodulation_rows]
    second_rows = [dict(row) for row in second_store.comodulation_rows]
    final_comod_row = first_rows[-1]

    assert len(first_rows) == 4
    assert first_rows == second_rows
    assert pearsonr_spy.call_count == 2
    for call in pearsonr_spy.call_args_list:
        streamer_arg, operator_arg = call.args
        assert list(streamer_arg) == pytest.approx(expected_streamer_snapshots)
        assert list(operator_arg) == pytest.approx(expected_operator_snapshots)

    assert final_comod_row["session_id"] == "00000000-0000-4000-8000-000000000421"
    assert final_comod_row["window_start_utc"] == analytics_now - timedelta(minutes=10)
    assert final_comod_row["window_end_utc"] == analytics_now
    assert final_comod_row["window_minutes"] == 10
    assert final_comod_row["co_modulation_index"] == pytest.approx(1.0)
    assert final_comod_row["n_paired_observations"] == 4
    assert final_comod_row["coverage_ratio"] == pytest.approx(1.0)
    assert final_comod_row["streamer_rmssd_mean"] == pytest.approx(
        sum(expected_streamer_snapshots) / len(expected_streamer_snapshots)
    )
    assert final_comod_row["operator_rmssd_mean"] == pytest.approx(
        sum(expected_operator_snapshots) / len(expected_operator_snapshots)
    )
