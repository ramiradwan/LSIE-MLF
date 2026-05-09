"""Tests for the v4 desktop Operator CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from typer.testing import CliRunner

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    AttributionSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    ObservationalAcousticSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from scripts.lsie_cli import app
from services.operator_console.api_client import ApiError

runner = CliRunner()
_SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
_ACTION_ID = UUID("11111111-1111-4111-8111-111111111111")
_NOW = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)


def _build_fake_client() -> FakeClient:
    return FakeClient()


class FakeClient:
    def __init__(self) -> None:
        self.started_stream_url: str | None = None
        self.started_experiment_id: str | None = None
        self.ended_session_id: str | None = None
        self.stimulus_session_id: str | None = None
        self.stimulus_note: str | None = None
        self.created_experiment_id: str | None = None
        self.created_arm_ids: list[str] = []
        self.added_arm_id: str | None = None
        self.updated_arm_id: str | None = None
        self.disabled_arm_id: str | None = None
        self.deleted_arm_id: str | None = None
        self.encounter_limit: int | None = None
        self.list_session_limit: int | None = None

    def get_overview(self) -> OverviewSnapshot:
        return _overview()

    def list_sessions(self, *, limit: int = 50) -> list[SessionSummary]:
        self.list_session_limit = limit
        return [_session()]

    def get_session(self, session_id: UUID | str) -> SessionSummary:
        assert str(session_id) == str(_SESSION_ID)
        return _session()

    def list_session_encounters(
        self,
        session_id: UUID | str,
        *,
        limit: int = 100,
        before_utc: datetime | None = None,
    ) -> list[EncounterSummary]:
        del before_utc
        assert str(session_id) == str(_SESSION_ID)
        self.encounter_limit = limit
        return [_encounter()]

    def list_experiments(self) -> list[ExperimentSummary]:
        return [_experiment_summary()]

    def get_experiment_detail(self, experiment_id: str) -> ExperimentDetail:
        assert experiment_id == "greeting_line_v1"
        return _experiment_detail()

    def get_session_physiology(self, session_id: UUID | str) -> SessionPhysiologySnapshot:
        assert str(session_id) == str(_SESSION_ID)
        return _physiology()

    def get_health(self) -> HealthSnapshot:
        return _health()

    def list_alerts(
        self,
        *,
        limit: int = 50,
        since_utc: datetime | None = None,
    ) -> list[AlertEvent]:
        del limit, since_utc
        return [_alert()]

    def post_stimulus(self, session_id: UUID | str, request: StimulusRequest) -> StimulusAccepted:
        self.stimulus_session_id = str(session_id)
        self.stimulus_note = request.operator_note
        return StimulusAccepted(
            session_id=_SESSION_ID,
            client_action_id=_ACTION_ID,
            accepted=True,
            received_at_utc=_NOW,
            stimulus_time_utc=_NOW,
            message="stimulus accepted",
        )

    def post_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        self.started_stream_url = request.stream_url
        self.started_experiment_id = request.experiment_id
        return SessionLifecycleAccepted(
            action="start",
            session_id=_SESSION_ID,
            client_action_id=_ACTION_ID,
            accepted=True,
            received_at_utc=_NOW,
            message="session start accepted",
        )

    def post_session_end(
        self,
        session_id: UUID | str,
        request: SessionEndRequest,
    ) -> SessionLifecycleAccepted:
        del request
        self.ended_session_id = str(session_id)
        return SessionLifecycleAccepted(
            action="end",
            session_id=_SESSION_ID,
            client_action_id=_ACTION_ID,
            accepted=True,
            received_at_utc=_NOW,
            message="session end accepted",
        )

    def create_experiment(self, request: ExperimentCreateRequest) -> ExperimentAdminResponse:
        self.created_experiment_id = request.experiment_id
        self.created_arm_ids = [arm.arm for arm in request.arms]
        return ExperimentAdminResponse(
            experiment_id="new_exp",
            label="New experiment",
            arms=[_admin_arm("warm")],
        )

    def add_experiment_arm(
        self,
        experiment_id: str,
        request: ExperimentArmCreateRequest,
    ) -> ExperimentArmAdminResponse:
        assert experiment_id == "greeting_line_v1"
        self.added_arm_id = request.arm
        return _admin_arm(self.added_arm_id or "warm")

    def patch_experiment_arm(
        self,
        experiment_id: str,
        arm_id: str,
        request: ExperimentArmPatchRequest,
    ) -> ExperimentArmAdminResponse:
        assert experiment_id == "greeting_line_v1"
        if request.enabled is False:
            self.disabled_arm_id = arm_id
            return _admin_arm(arm_id, enabled=False)
        self.updated_arm_id = arm_id
        return _admin_arm(arm_id, greeting_text=request.greeting_text)

    def delete_experiment_arm(self, experiment_id: str, arm_id: str) -> ExperimentArmDeleteResponse:
        assert experiment_id == "greeting_line_v1"
        self.deleted_arm_id = arm_id
        return ExperimentArmDeleteResponse(
            experiment_id=experiment_id,
            arm=arm_id,
            deleted=False,
            posterior_preserved=True,
            reason="posterior history preserved",
            arm_state=_admin_arm(arm_id, enabled=False),
        )


class ErrorClient(FakeClient):
    def get_overview(self) -> OverviewSnapshot:
        raise ApiError("connection refused", endpoint="/api/v1/operator/overview", retryable=True)


def _session() -> SessionSummary:
    return SessionSummary(
        session_id=_SESSION_ID,
        status="active",
        started_at_utc=_NOW,
        duration_s=65,
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        expected_greeting="hei rakas",
        latest_reward=0.42,
        latest_semantic_gate=1,
    )


def _experiment_summary() -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id="greeting_line_v1",
        label="Greeting line",
        active_arm_id="warm_welcome",
        arm_count=2,
        latest_reward=0.42,
        last_updated_utc=_NOW,
    )


def _experiment_detail() -> ExperimentDetail:
    return ExperimentDetail(
        experiment_id="greeting_line_v1",
        label="Greeting line",
        active_arm_id="warm_welcome",
        arms=[
            ArmSummary(
                arm_id="warm_welcome",
                greeting_text="hei rakas",
                posterior_alpha=3.0,
                posterior_beta=2.0,
                evaluation_variance=0.1,
                selection_count=5,
                recent_reward_mean=0.42,
                recent_semantic_pass_rate=0.8,
            )
        ],
        last_update_summary="last update preserved posterior state",
        last_updated_utc=_NOW,
    )


def _admin_arm(
    arm_id: str,
    *,
    greeting_text: str | None = None,
    enabled: bool = True,
) -> ExperimentArmAdminResponse:
    return ExperimentArmAdminResponse(
        experiment_id="greeting_line_v1",
        label="Greeting line",
        arm=arm_id,
        greeting_text=greeting_text or "hei rakas",
        alpha_param=1.0,
        beta_param=1.0,
        enabled=enabled,
        updated_at=_NOW,
    )


def _encounter() -> EncounterSummary:
    return EncounterSummary(
        encounter_id="enc-1",
        session_id=_SESSION_ID,
        segment_timestamp_utc=_NOW,
        state=EncounterState.COMPLETED,
        active_arm="warm_welcome",
        expected_greeting="hei rakas",
        stimulus_time_utc=_NOW,
        semantic_gate=1,
        semantic_confidence=0.91,
        transcription="hei rakas",
        p90_intensity=0.7,
        gated_reward=0.7,
        n_frames_in_window=42,
        au12_baseline_pre=0.1,
        observational_acoustic=ObservationalAcousticSummary(
            f0_valid_measure=True,
            f0_valid_baseline=True,
            perturbation_valid_measure=True,
            perturbation_valid_baseline=True,
            voiced_coverage_measure_s=2.0,
            f0_mean_measure_hz=220.0,
            f0_mean_baseline_hz=210.0,
            f0_delta_semitones=0.8,
        ),
        semantic_evaluation=SemanticEvaluationSummary(
            reasoning="expected_phrase_match",
            is_match=True,
            confidence_score=0.91,
            semantic_method="deterministic",
            semantic_method_version="v1",
        ),
        attribution=AttributionSummary(
            finality="online_provisional",
            soft_reward_candidate=0.4,
            au12_lift_p90=0.6,
            au12_peak_latency_ms=1200,
        ),
    )


def _latest_encounter() -> LatestEncounterSummary:
    encounter = _encounter()
    return LatestEncounterSummary(
        encounter_id=encounter.encounter_id,
        session_id=encounter.session_id,
        segment_timestamp_utc=encounter.segment_timestamp_utc,
        state=encounter.state,
        active_arm=encounter.active_arm,
        expected_greeting=encounter.expected_greeting,
        stimulus_time_utc=encounter.stimulus_time_utc,
        semantic_gate=encounter.semantic_gate,
        p90_intensity=encounter.p90_intensity,
        gated_reward=encounter.gated_reward,
        n_frames_in_window=encounter.n_frames_in_window,
        observational_acoustic=encounter.observational_acoustic,
        semantic_evaluation=encounter.semantic_evaluation,
        attribution=encounter.attribution,
    )


def _physiology() -> SessionPhysiologySnapshot:
    return SessionPhysiologySnapshot(
        session_id=_SESSION_ID,
        streamer=PhysiologyCurrentSnapshot(
            subject_role="streamer",
            rmssd_ms=52.0,
            heart_rate_bpm=68,
            provider="oura",
            source_timestamp_utc=_NOW,
            freshness_s=12.0,
            is_stale=False,
        ),
        operator=PhysiologyCurrentSnapshot(
            subject_role="operator",
            rmssd_ms=61.0,
            heart_rate_bpm=72,
            provider="oura",
            source_timestamp_utc=_NOW,
            freshness_s=15.0,
            is_stale=False,
        ),
        comodulation=CoModulationSummary(
            session_id=_SESSION_ID,
            co_modulation_index=0.82,
            n_paired_observations=6,
            coverage_ratio=0.75,
            streamer_rmssd_mean=52.0,
            operator_rmssd_mean=61.0,
            window_start_utc=_NOW,
            window_end_utc=_NOW,
        ),
        generated_at_utc=_NOW,
    )


def _health() -> HealthSnapshot:
    return HealthSnapshot(
        generated_at_utc=_NOW,
        overall_state=HealthState.DEGRADED,
        degraded_count=1,
        recovering_count=0,
        error_count=0,
        subsystems=[
            HealthSubsystemStatus(
                subsystem_key="capture_supervisor",
                label="Capture supervisor",
                state=HealthState.DEGRADED,
                detail="device reconnecting",
                recovery_mode="polling",
                operator_action_hint="check USB connection",
                last_success_utc=_NOW,
            )
        ],
        subsystem_probes={
            "ui_api_shell": HealthSubsystemProbe(
                subsystem_key="ui_api_shell",
                label="UI API shell",
                state=HealthProbeState.OK,
                latency_ms=8.0,
                detail="loopback ok",
                checked_at_utc=_NOW,
            )
        },
    )


def _alert() -> AlertEvent:
    return AlertEvent(
        alert_id="alert-1",
        severity=AlertSeverity.WARNING,
        kind=AlertKind.SUBSYSTEM_DEGRADED,
        message="capture supervisor reconnecting",
        session_id=_SESSION_ID,
        subsystem_key="capture_supervisor",
        emitted_at_utc=_NOW,
    )


def _overview() -> OverviewSnapshot:
    return OverviewSnapshot(
        generated_at_utc=_NOW,
        active_session=_session(),
        latest_encounter=_latest_encounter(),
        experiment_summary=_experiment_summary(),
        physiology=_physiology(),
        health=_health(),
        alerts=[_alert()],
    )


def test_global_api_url_option_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", _build_fake_client)
    result = runner.invoke(app, ["--api-url", "http://127.0.0.1:8000", "status"])
    assert result.exit_code == 0
    assert "Active session" in result.output


def test_root_help_lists_v4_desktop_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "overview" in result.output
    assert "sessions" in result.output
    assert "live-session" in result.output
    assert "experiments" in result.output
    assert "health" in result.output
    assert "alerts" in result.output
    assert "metrics" not in result.output
    assert "Docker" not in result.output
    assert "Postgres" not in result.output
    assert "Redis" not in result.output


def test_status_renders_desktop_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", _build_fake_client)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Active session" in result.output
    assert "stimulus strategy" in result.output
    assert "warm_welcome" in result.output
    assert "Health: degraded" in result.output
    assert "capture supervisor reconnecting" in result.output


def test_overview_json_outputs_dto_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", _build_fake_client)
    result = runner.invoke(app, ["overview", "--json"])
    assert result.exit_code == 0
    assert '"active_session"' in result.output
    assert '"warm_welcome"' in result.output


def test_sessions_list_and_show_use_typed_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("scripts.lsie_cli._build_client", lambda: client)
    list_result = runner.invoke(app, ["sessions", "list", "--limit", "10"])
    show_result = runner.invoke(app, ["sessions", "show", str(_SESSION_ID)])
    alias_result = runner.invoke(app, ["session", "status", str(_SESSION_ID)])
    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert alias_result.exit_code == 0
    assert client.list_session_limit == 10
    assert "Expected response" in show_result.output
    assert "hei rakas" in show_result.output


def test_sessions_start_and_end_send_lifecycle_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("scripts.lsie_cli._build_client", lambda: client)
    start_result = runner.invoke(
        app,
        ["sessions", "start", "test://device", "--experiment", "greeting_line_v1"],
    )
    end_result = runner.invoke(app, ["sessions", "end", str(_SESSION_ID)])
    assert start_result.exit_code == 0
    assert end_result.exit_code == 0
    assert client.started_stream_url == "test://device"
    assert client.started_experiment_id == "greeting_line_v1"
    assert client.ended_session_id == str(_SESSION_ID)
    assert "Session start accepted" in start_result.output
    assert "Session end accepted" in end_result.output


def test_live_session_encounters_and_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("scripts.lsie_cli._build_client", lambda: client)
    table_result = runner.invoke(
        app,
        ["live-session", "encounters", str(_SESSION_ID), "--limit", "3"],
    )
    readback_result = runner.invoke(app, ["live-session", "readback", str(_SESSION_ID)])
    assert table_result.exit_code == 0
    assert readback_result.exit_code == 0
    assert client.encounter_limit == 5
    assert "warm_welcome" in table_result.output
    assert "0.700" in table_result.output
    assert "Host response" in readback_result.output
    assert "Why it counted" in readback_result.output


def test_stimulus_submit_and_alias_send_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("scripts.lsie_cli._build_client", lambda: client)
    submit_result = runner.invoke(
        app,
        ["stimulus", "submit", str(_SESSION_ID), "--note", "operator ready"],
    )
    alias_result = runner.invoke(app, ["stimulus", "inject", str(_SESSION_ID)])
    assert submit_result.exit_code == 0
    assert alias_result.exit_code == 0
    assert client.stimulus_session_id == str(_SESSION_ID)
    assert "Authoritative stimulus time" in submit_result.output
    assert "from the service readback" in submit_result.output
    assert "accepted" in submit_result.output


def test_experiment_read_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", _build_fake_client)
    list_result = runner.invoke(app, ["experiments", "list"])
    show_result = runner.invoke(app, ["experiments", "show", "greeting_line_v1"])
    alias_result = runner.invoke(app, ["experiment", "show", "greeting_line_v1"])
    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert alias_result.exit_code == 0
    assert "Greeting line" in list_result.output
    assert "Expected response" in show_result.output
    assert "positive history" in show_result.output


def test_experiment_write_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("scripts.lsie_cli._build_client", lambda: client)
    create_result = runner.invoke(
        app,
        [
            "experiments",
            "create",
            "new_exp",
            "--label",
            "New experiment",
            "--arm",
            "warm=hei rakas",
        ],
    )
    add_result = runner.invoke(
        app,
        ["experiments", "add-arm", "greeting_line_v1", "direct", "--greeting-text", "moi"],
    )
    update_result = runner.invoke(
        app,
        [
            "experiments",
            "update-arm",
            "greeting_line_v1",
            "direct",
            "--greeting-text",
            "hei",
        ],
    )
    disable_result = runner.invoke(
        app,
        ["experiments", "disable-arm", "greeting_line_v1", "direct"],
    )
    delete_result = runner.invoke(app, ["experiments", "delete-arm", "greeting_line_v1", "direct"])
    assert create_result.exit_code == 0
    assert add_result.exit_code == 0
    assert update_result.exit_code == 0
    assert disable_result.exit_code == 0
    assert delete_result.exit_code == 0
    assert client.created_experiment_id == "new_exp"
    assert client.created_arm_ids == ["warm"]
    assert client.added_arm_id == "direct"
    assert client.updated_arm_id == "direct"
    assert client.disabled_arm_id == "direct"
    assert client.deleted_arm_id == "direct"
    assert "Posterior preserved: True" in delete_result.output


def test_health_alerts_physiology_and_comodulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", _build_fake_client)
    health_result = runner.invoke(app, ["health"])
    alerts_result = runner.invoke(app, ["alerts"])
    physiology_result = runner.invoke(app, ["physiology", "show", str(_SESSION_ID)])
    comodulation_result = runner.invoke(app, ["comodulation", "show", str(_SESSION_ID)])
    assert health_result.exit_code == 0
    assert alerts_result.exit_code == 0
    assert physiology_result.exit_code == 0
    assert comodulation_result.exit_code == 0
    assert "Capture supervisor" in health_result.output
    assert "capture supervisor reconnecting" in alerts_result.output
    assert "streamer" in physiology_result.output
    assert "Co-Modulation Index" in comodulation_result.output
    assert "+0.82" in comodulation_result.output


def test_api_error_uses_desktop_safe_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.lsie_cli._build_client", ErrorClient)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "operator API runtime" in result.output
    assert "--operator-api" in result.output
    assert "--api-url" in result.output
    assert "headless desktop app" not in result.output
    assert "Docker" not in result.output
    assert "Postgres" not in result.output
    assert "Redis" not in result.output
