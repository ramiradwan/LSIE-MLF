"""Operator CLI for the v4 desktop loopback API.

The CLI is a terminal companion to ``services.operator_console``: it uses the
same typed client, DTOs, and operator-language formatters as the PySide console.
It targets the local ``ui_api_shell`` FastAPI surface hosted by the desktop
ProcessGraph and never opens direct storage or infrastructure connections.

Entry point: ``python -m scripts <command> [options]``
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar, cast
from uuid import uuid4

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from packages.schemas.experiments import (
    ExperimentArmCreateRequest,
    ExperimentArmPatchRequest,
    ExperimentArmSeedRequest,
    ExperimentCreateRequest,
)
from packages.schemas.operator_console import (
    AlertEvent,
    CloudActionStatus,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOutboxSummary,
    CloudSignInResult,
    CoModulationSummary,
    EncounterSummary,
    ExperimentBundleRefreshResult,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    HealthSubsystemProbe,
    LatestEncounterSummary,
    PhysiologyCurrentSnapshot,
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.formatters import (
    build_acoustic_detail_display,
    build_cause_effect_display,
    build_co_modulation_display,
    build_health_detail,
    build_reward_explanation,
    build_semantic_attribution_diagnostics_display,
    build_strategy_evidence_display,
    format_active_session_readback,
    format_duration,
    format_freshness,
    format_health_probe_state,
    format_health_state,
    format_percentage,
    format_reward,
    format_semantic_gate,
    format_timestamp,
)

_DEFAULT_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_API_TIMEOUT_SECONDS = 10.0

API_BASE: str = os.environ.get("LSIE_API_URL", _DEFAULT_API_BASE).rstrip("/")

console = Console(highlight=False, width=140)
error_console = Console(stderr=True, highlight=False, width=140)

_ResultT = TypeVar("_ResultT")


class _ReconfigurableStream(Protocol):
    def reconfigure(self, *, encoding: str, errors: str) -> None: ...


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            cast(_ReconfigurableStream, stream).reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()

_ARMS_OPTION = typer.Option(
    ...,
    "--arm",
    help="Initial arm as ARM_ID=EXPECTED_RESPONSE; repeat for multiple arms",
)

app = typer.Typer(
    name="lsie",
    help="LSIE-MLF v4 desktop operator CLI for the local loopback API.",
    no_args_is_help=True,
    add_completion=False,
)
sessions_app = typer.Typer(help="Inspect and control desktop live sessions", no_args_is_help=True)
session_alias_app = typer.Typer(
    help="Compatibility alias for session readbacks",
    no_args_is_help=True,
)
live_session_app = typer.Typer(
    help="Inspect live-session encounters and readbacks",
    no_args_is_help=True,
)
experiments_app = typer.Typer(help="Inspect and manage stimulus strategies", no_args_is_help=True)
experiment_alias_app = typer.Typer(
    help="Compatibility alias for experiment readbacks",
    no_args_is_help=True,
)
stimulus_app = typer.Typer(help="Submit operator stimulus actions", no_args_is_help=True)
physiology_app = typer.Typer(help="Inspect physiology snapshots", no_args_is_help=True)
comodulation_app = typer.Typer(help="Inspect Co-Modulation Index readbacks", no_args_is_help=True)
cloud_app = typer.Typer(help="Inspect and control cloud sync readbacks", no_args_is_help=True)

app.add_typer(sessions_app, name="sessions")
app.add_typer(session_alias_app, name="session")
app.add_typer(live_session_app, name="live-session")
app.add_typer(experiments_app, name="experiments")
app.add_typer(experiment_alias_app, name="experiment")
app.add_typer(stimulus_app, name="stimulus")
app.add_typer(physiology_app, name="physiology")
app.add_typer(comodulation_app, name="comodulation")
app.add_typer(cloud_app, name="cloud")


@app.callback()
def _root(
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Loopback API base URL (default: $LSIE_API_URL or http://127.0.0.1:8000)",
    ),
) -> None:
    """Apply global options before running a command."""
    if api_url:
        global API_BASE
        API_BASE = api_url.rstrip("/")


def _consume_trailing_api_url() -> None:
    try:
        index = sys.argv.index("--api-url")
    except ValueError:
        return
    value_index = index + 1
    if value_index >= len(sys.argv):
        return
    global API_BASE
    API_BASE = sys.argv[value_index].rstrip("/")
    del sys.argv[index : value_index + 1]


_consume_trailing_api_url()


def _configured_timeout_seconds() -> float:
    raw = os.environ.get("LSIE_API_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_API_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_API_TIMEOUT_SECONDS


def _build_client() -> ApiClient:
    return ApiClient(API_BASE, timeout_seconds=_configured_timeout_seconds())


def _run_api_call(call: Callable[[], _ResultT]) -> _ResultT:
    try:
        return call()
    except ApiError as exc:
        error_console.print(f"API call failed against {API_BASE}: {exc}")
        error_console.print(
            "Start the operator API runtime with `python -m services.desktop_app "
            "--operator-api`, launch the Operator Console, or pass --api-url if the "
            "loopback API selected another port."
        )
        raise typer.Exit(1) from exc


def _print_json(payload: BaseModel | Sequence[BaseModel]) -> None:
    if isinstance(payload, BaseModel):
        console.print(payload.model_dump_json(indent=2))
        return
    console.print(json.dumps([item.model_dump(mode="json") for item in payload], indent=2))


def _render_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    title: str | None = None,
    empty_message: str = "(no data)",
) -> None:
    materialized = list(rows)
    if not materialized:
        console.print(empty_message)
        return
    table = Table(title=title, show_lines=False, pad_edge=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in materialized:
        table.add_row(*row)
    console.print(table)


def _text(value: object | None) -> str:
    if value is None:
        return "—"
    return str(value)


def _health_counts(snapshot: HealthSnapshot | None) -> str:
    if snapshot is None:
        return "health unknown"
    parts: list[str] = []
    if snapshot.degraded_count:
        parts.append(f"{snapshot.degraded_count} degraded")
    if snapshot.recovering_count:
        parts.append(f"{snapshot.recovering_count} recovering")
    if snapshot.error_count:
        parts.append(f"{snapshot.error_count} error")
    if not parts:
        parts.append("all subsystems ok")
    return " · ".join(parts)


def _latest_encounter_summary(encounter: LatestEncounterSummary | None) -> str:
    if encounter is None:
        return "No completed encounter yet."
    parts = [
        f"state {encounter.state.value}",
        f"stimulus confirmed {format_semantic_gate(encounter.semantic_gate)}",
        f"strongest response signal {format_reward(encounter.p90_intensity)}",
        f"reward used {format_reward(encounter.gated_reward)}",
    ]
    if encounter.semantic_evaluation is not None or encounter.attribution is not None:
        diagnostics = build_semantic_attribution_diagnostics_display(
            encounter.semantic_evaluation,
            encounter.attribution,
        )
        parts.append(diagnostics.compact_summary)
    return " · ".join(parts)


@app.command("status")
def status(json_output: bool = typer.Option(False, "--json", help="Render DTO JSON")) -> None:
    """Show a compact desktop operator status readback."""
    overview = _run_api_call(lambda: _build_client().get_overview())
    if json_output:
        _print_json(overview)
        return
    console.print(f"Generated: {format_timestamp(overview.generated_at_utc)}")
    if overview.active_session is None:
        console.print("Active session: none")
    else:
        console.print(f"Active session: {format_active_session_readback(overview.active_session)}")
    if overview.health is None:
        console.print("Health: unknown")
    else:
        health_state = format_health_state(overview.health.overall_state)
        console.print(f"Health: {health_state} · {_health_counts(overview.health)}")
    if not overview.alerts:
        console.print("Alerts: none")
    else:
        console.print(f"Alerts: {len(overview.alerts)} recent")
        for alert in overview.alerts[:3]:
            console.print(f"- {alert.severity.value}: {alert.message}")


@app.command("overview")
def overview(json_output: bool = typer.Option(False, "--json", help="Render DTO JSON")) -> None:
    """Show the terminal version of the Operator Console overview."""
    snapshot = _run_api_call(lambda: _build_client().get_overview())
    if json_output:
        _print_json(snapshot)
        return
    console.print(f"Overview generated {format_timestamp(snapshot.generated_at_utc)}")
    console.print("")
    _render_overview_session(snapshot.active_session)
    _render_overview_experiment(snapshot.experiment_summary)
    _render_overview_encounter(snapshot.latest_encounter)
    _render_overview_physiology(snapshot.physiology)
    _render_overview_health(snapshot.health)
    _render_overview_alerts(snapshot.alerts)


def _render_overview_session(session: SessionSummary | None) -> None:
    console.print("Active session")
    if session is None:
        console.print("  none")
        return
    console.print(f"  {format_active_session_readback(session)}")
    console.print(f"  status {session.status} · experiment {_text(session.experiment_id)}")


def _render_overview_experiment(experiment: ExperimentSummary | None) -> None:
    console.print("Experiment")
    if experiment is None:
        console.print("  none")
        return
    console.print(
        f"  {experiment.experiment_id} · {_text(experiment.label)} · "
        f"active strategy {_text(experiment.active_arm_id)}"
    )
    console.print(
        f"  {experiment.arm_count} arm(s) · latest reward {format_reward(experiment.latest_reward)}"
    )


def _render_overview_encounter(encounter: LatestEncounterSummary | None) -> None:
    console.print("Latest encounter")
    console.print(f"  {_latest_encounter_summary(encounter)}")


def _render_overview_physiology(snapshot: SessionPhysiologySnapshot | None) -> None:
    console.print("Physiology")
    if snapshot is None:
        console.print("  no physiology snapshot yet")
        return
    _render_physio_role("streamer", snapshot.streamer)
    _render_physio_role("operator", snapshot.operator)
    display = build_co_modulation_display(snapshot.comodulation)
    console.print(f"  {display.title}: {display.primary} · {display.secondary}")


def _render_overview_health(snapshot: HealthSnapshot | None) -> None:
    console.print("Health")
    if snapshot is None:
        console.print("  unknown")
        return
    console.print(
        f"  {format_health_state(snapshot.overall_state)} · {_health_counts(snapshot)} · "
        f"generated {format_timestamp(snapshot.generated_at_utc)}"
    )


def _render_overview_alerts(alerts: Sequence[AlertEvent]) -> None:
    console.print("Alerts")
    if not alerts:
        console.print("  none")
        return
    for alert in alerts[:5]:
        console.print(f"  {alert.severity.value}: {alert.message}")


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(50, "--limit", min=1, max=500),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """List recent desktop sessions."""
    rows = _run_api_call(lambda: _build_client().list_sessions(limit=limit))
    if json_output:
        _print_json(rows)
        return
    _render_sessions(rows)


@session_alias_app.command("list")
def session_list_alias(
    limit: int = typer.Option(50, "--limit", min=1, max=500),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Compatibility alias for ``sessions list``."""
    sessions_list(limit=limit, json_output=json_output)


def _render_sessions(rows: Sequence[SessionSummary]) -> None:
    _render_table(
        ["Session", "Status", "Stimulus strategy", "Expected response", "Duration", "Reward"],
        (
            [
                str(row.session_id),
                row.status,
                _text(row.active_arm),
                _text(row.expected_greeting),
                format_duration(row.duration_s),
                format_reward(row.latest_reward),
            ]
            for row in rows
        ),
        empty_message="No sessions found. Start the operator API runtime and begin a live session.",
    )


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show one desktop session readback."""
    session = _run_api_call(lambda: _build_client().get_session(session_id))
    if json_output:
        _print_json(session)
        return
    _render_session_detail(session)


@session_alias_app.command("status")
def session_status_alias(
    session_id: str = typer.Argument(..., help="Session identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Compatibility alias for ``sessions show``."""
    sessions_show(session_id=session_id, json_output=json_output)


def _render_session_detail(session: SessionSummary) -> None:
    console.print(f"Session: {session.session_id}")
    console.print(f"Status: {session.status}")
    console.print(f"Started: {format_timestamp(session.started_at_utc)}")
    console.print(f"Ended: {format_timestamp(session.ended_at_utc)}")
    console.print(f"Duration: {format_duration(session.duration_s)}")
    console.print(f"Experiment: {_text(session.experiment_id)}")
    console.print(f"Stimulus strategy: {_text(session.active_arm)}")
    console.print(f"Expected response: {_text(session.expected_greeting)}")
    console.print(f"Latest reward: {format_reward(session.latest_reward)}")
    console.print(f"Stimulus confirmed: {format_semantic_gate(session.latest_semantic_gate)}")


@sessions_app.command("start")
def sessions_start(
    stream_url: str = typer.Argument(..., help="Capture source or test stream URL"),
    experiment_id: str = typer.Option(..., "--experiment", help="Experiment identifier"),
) -> None:
    """Request a new desktop live session."""
    try:
        request = SessionCreateRequest(
            stream_url=stream_url,
            experiment_id=experiment_id,
            client_action_id=uuid4(),
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = _run_api_call(lambda: _build_client().post_session_start(request))
    _render_lifecycle_result(result)


@sessions_app.command("end")
def sessions_end(session_id: str = typer.Argument(..., help="Session identifier")) -> None:
    """Request desktop live-session shutdown."""
    request = SessionEndRequest(client_action_id=uuid4())
    result = _run_api_call(lambda: _build_client().post_session_end(session_id, request))
    _render_lifecycle_result(result)


def _render_lifecycle_result(result: SessionLifecycleAccepted) -> None:
    accepted = "accepted" if result.accepted else "not accepted"
    console.print(f"Session {result.action} {accepted}: {result.session_id}")
    console.print(f"Client action: {result.client_action_id}")
    console.print(f"Received: {format_timestamp(result.received_at_utc)}")
    if result.message:
        console.print(result.message)


@live_session_app.command("encounters")
def live_session_encounters(
    session_id: str = typer.Argument(..., help="Session identifier"),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """List session-scoped encounter rows."""
    rows = _run_api_call(lambda: _build_client().list_session_encounters(session_id, limit=limit))
    if json_output:
        _print_json(rows)
        return
    labels = [
        "Time",
        "State",
        "Stimulus strategy",
        "Stimulus confirmed?",
        "Strongest response signal",
        "Reward used",
        "Frames",
    ]
    _render_table(
        labels,
        (
            [
                format_timestamp(row.segment_timestamp_utc),
                row.state.value,
                _text(row.active_arm),
                format_semantic_gate(row.semantic_gate),
                format_reward(row.p90_intensity),
                format_reward(row.gated_reward),
                _text(row.n_frames_in_window),
            ]
            for row in rows
        ),
        empty_message="No live-session encounters yet.",
    )


@live_session_app.command("readback")
def live_session_readback(
    session_id: str = typer.Argument(..., help="Session identifier"),
    limit: int = typer.Option(5, "--limit", min=1, max=50),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show recent operator-language encounter readbacks."""
    rows = _run_api_call(lambda: _build_client().list_session_encounters(session_id, limit=limit))
    if json_output:
        _print_json(rows)
        return
    if not rows:
        console.print("No live-session readbacks yet.")
        return
    for encounter in rows:
        _render_encounter_readback(encounter)
        console.print("")


def _render_encounter_readback(encounter: EncounterSummary) -> None:
    cause = build_cause_effect_display(encounter)
    acoustic = build_acoustic_detail_display(encounter.observational_acoustic)
    diagnostics = build_semantic_attribution_diagnostics_display(
        encounter.semantic_evaluation,
        encounter.attribution,
    )
    encounter_time = format_timestamp(encounter.segment_timestamp_utc)
    console.print(f"Encounter {encounter.encounter_id} · {encounter_time}")
    console.print(cause.headline)
    console.print(cause.detail)
    console.print(f"Response: {cause.response_summary}")
    console.print(f"Voice: {cause.voice_summary}")
    console.print(f"Why it counted: {cause.technical_summary}")
    console.print(f"Acoustics: {acoustic.explanation}")
    console.print(f"Diagnostics: {diagnostics.compact_summary}")
    console.print(f"Reward: {build_reward_explanation(encounter)}")


@stimulus_app.command("submit")
def stimulus_submit(
    session_id: str = typer.Argument(..., help="Session identifier"),
    note: str | None = typer.Option(None, "--note", help="Operator note for the stimulus"),
) -> None:
    """Submit a stimulus intent for the active live session."""
    request = StimulusRequest(operator_note=note, client_action_id=uuid4())
    result = _run_api_call(lambda: _build_client().post_stimulus(session_id, request))
    _render_stimulus_result(result)


@stimulus_app.command("inject")
def stimulus_inject_alias(
    session_id: str = typer.Argument(..., help="Session identifier"),
    note: str | None = typer.Option(None, "--note", help="Operator note for the stimulus"),
) -> None:
    """Compatibility alias for ``stimulus submit``."""
    stimulus_submit(session_id=session_id, note=note)


def _render_stimulus_result(result: StimulusAccepted) -> None:
    accepted = "accepted" if result.accepted else "not accepted"
    console.print(f"Stimulus {accepted}: {result.session_id}")
    console.print(f"Client action: {result.client_action_id}")
    console.print(f"Received: {format_timestamp(result.received_at_utc)}")
    console.print(
        "Authoritative stimulus time: "
        f"{format_timestamp(result.stimulus_time_utc)} from the service readback"
    )
    if result.message:
        console.print(result.message)


@experiments_app.command("list")
def experiments_list(
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """List experiment summaries."""
    rows = _run_api_call(lambda: _build_client().list_experiments())
    if json_output:
        _print_json(rows)
        return
    _render_experiment_summaries(rows)


@experiment_alias_app.command("list")
def experiment_list_alias(
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Compatibility alias for ``experiments list``."""
    experiments_list(json_output=json_output)


def _render_experiment_summaries(rows: Sequence[ExperimentSummary]) -> None:
    _render_table(
        ["Experiment", "Label", "Active strategy", "Arms", "Latest reward"],
        (
            [
                row.experiment_id,
                _text(row.label),
                _text(row.active_arm_id),
                str(row.arm_count),
                format_reward(row.latest_reward),
            ]
            for row in rows
        ),
        empty_message="No experiments registered.",
    )


@experiments_app.command("show")
def experiments_show(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show experiment strategy readback."""
    detail = _run_api_call(lambda: _build_client().get_experiment_detail(experiment_id))
    if json_output:
        _print_json(detail)
        return
    _render_experiment_detail(detail)


@experiment_alias_app.command("show")
def experiment_show_alias(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Compatibility alias for ``experiments show``."""
    experiments_show(experiment_id=experiment_id, json_output=json_output)


def _render_experiment_detail(detail: ExperimentDetail) -> None:
    console.print(f"Experiment: {detail.experiment_id}")
    console.print(f"Label: {_text(detail.label)}")
    console.print(f"Active strategy: {_text(detail.active_arm_id)}")
    if detail.last_update_summary:
        console.print(detail.last_update_summary)
    displays = build_strategy_evidence_display(detail)
    _render_table(
        ["Strategy", "Expected response", "Status", "Evidence", "Outcome"],
        (
            [
                display.arm_id,
                display.greeting_text,
                display.label,
                display.evidence,
                display.outcome,
            ]
            for display in displays
        ),
        empty_message="No arms registered.",
    )


@experiments_app.command("create")
def experiments_create(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    label: str = typer.Option(..., "--label", help="Human-readable experiment label"),
    arms: list[str] = _ARMS_OPTION,
) -> None:
    """Create an experiment with initial stimulus strategies."""
    try:
        request = ExperimentCreateRequest(
            experiment_id=experiment_id,
            label=label,
            arms=[_parse_arm_seed(value) for value in arms],
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = _run_api_call(lambda: _build_client().create_experiment(request))
    console.print(f"Experiment created: {result.experiment_id} · {result.label}")
    console.print(f"Arms: {len(result.arms)}")


@experiments_app.command("add-arm")
def experiments_add_arm(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    arm_id: str = typer.Argument(..., help="Arm identifier"),
    greeting_text: str = typer.Option(..., "--greeting-text", help="Expected response text"),
) -> None:
    """Add one new experiment arm."""
    try:
        request = ExperimentArmCreateRequest(arm=arm_id, greeting_text=greeting_text)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = _run_api_call(lambda: _build_client().add_experiment_arm(experiment_id, request))
    console.print(f"Arm added: {result.experiment_id} · {result.arm}")
    console.print(f"Expected response: {result.greeting_text}")


@experiments_app.command("update-arm")
def experiments_update_arm(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    arm_id: str = typer.Argument(..., help="Arm identifier"),
    greeting_text: str = typer.Option(..., "--greeting-text", help="Expected response text"),
) -> None:
    """Update human-owned arm metadata."""
    try:
        request = ExperimentArmPatchRequest(greeting_text=greeting_text)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = _run_api_call(
        lambda: _build_client().patch_experiment_arm(experiment_id, arm_id, request)
    )
    console.print(f"Arm updated: {result.experiment_id} · {result.arm}")
    console.print(f"Expected response: {result.greeting_text}")


@experiments_app.command("disable-arm")
def experiments_disable_arm(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    arm_id: str = typer.Argument(..., help="Arm identifier"),
) -> None:
    """Disable one experiment arm without editing posterior state."""
    request = ExperimentArmPatchRequest(enabled=False)
    result = _run_api_call(
        lambda: _build_client().patch_experiment_arm(experiment_id, arm_id, request)
    )
    console.print(f"Arm disabled: {result.experiment_id} · {result.arm}")


@experiments_app.command("delete-arm")
def experiments_delete_arm(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    arm_id: str = typer.Argument(..., help="Arm identifier"),
) -> None:
    """Delete or end-date one arm using the service guard."""
    result = _run_api_call(lambda: _build_client().delete_experiment_arm(experiment_id, arm_id))
    outcome = "deleted" if result.deleted else "disabled/end-dated"
    console.print(f"Arm {outcome}: {result.experiment_id} · {result.arm}")
    console.print(f"Posterior preserved: {result.posterior_preserved}")
    if result.reason:
        console.print(result.reason)


def _parse_arm_seed(value: str) -> ExperimentArmSeedRequest:
    arm_id, separator, greeting_text = value.partition("=")
    if not separator or not arm_id.strip() or not greeting_text.strip():
        raise typer.BadParameter("--arm must use ARM_ID=EXPECTED_RESPONSE")
    try:
        return ExperimentArmSeedRequest(arm=arm_id.strip(), greeting_text=greeting_text.strip())
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("health")
def health(json_output: bool = typer.Option(False, "--json", help="Render DTO JSON")) -> None:
    """Show desktop subsystem health."""
    snapshot = _run_api_call(lambda: _build_client().get_health())
    if json_output:
        _print_json(snapshot)
        return
    _render_health(snapshot)


def _render_health(snapshot: HealthSnapshot) -> None:
    console.print(
        f"Overall: {format_health_state(snapshot.overall_state)} · {_health_counts(snapshot)} · "
        f"generated {format_timestamp(snapshot.generated_at_utc)}"
    )
    _render_table(
        ["Subsystem", "State", "Detail"],
        (
            [row.label, format_health_state(row.state), build_health_detail(row)]
            for row in snapshot.subsystems
        ),
        empty_message="No subsystem rows reported.",
    )
    if snapshot.subsystem_probes:
        _render_table(
            ["Probe", "State", "Latency", "Detail"],
            (_probe_row(probe) for probe in snapshot.subsystem_probes.values()),
            title="Connectivity probes",
        )


def _probe_row(probe: HealthSubsystemProbe) -> list[str]:
    latency = "—" if probe.latency_ms is None else f"{probe.latency_ms:.0f}ms"
    return [probe.label, format_health_probe_state(probe.state), latency, probe.detail or "—"]


@app.command("alerts")
def alerts(
    limit: int = typer.Option(50, "--limit", min=1, max=500),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show the operator attention queue."""
    rows = _run_api_call(lambda: _build_client().list_alerts(limit=limit))
    if json_output:
        _print_json(rows)
        return
    _render_table(
        ["Time", "Severity", "Kind", "Message", "Acknowledged"],
        (
            [
                format_timestamp(row.emitted_at_utc),
                row.severity.value,
                row.kind.value,
                row.message,
                "yes" if row.acknowledged else "no",
            ]
            for row in rows
        ),
        empty_message="No alerts.",
    )


@physiology_app.command("show")
def physiology_show(
    session_id: str = typer.Argument(..., help="Session identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show latest physiology readback for a session."""
    snapshot = _run_api_call(lambda: _build_client().get_session_physiology(session_id))
    if json_output:
        _print_json(snapshot)
        return
    console.print(f"Session: {snapshot.session_id}")
    console.print(f"Generated: {format_timestamp(snapshot.generated_at_utc)}")
    _render_physio_role("streamer", snapshot.streamer)
    _render_physio_role("operator", snapshot.operator)
    display = build_co_modulation_display(snapshot.comodulation)
    console.print(f"{display.title}: {display.primary} · {display.secondary}")
    console.print(display.detail)


def _render_physio_role(label: str, snapshot: PhysiologyCurrentSnapshot | None) -> None:
    if snapshot is None:
        console.print(f"  {label}: absent")
        return
    heart_rate = "—" if snapshot.heart_rate_bpm is None else f"{snapshot.heart_rate_bpm} bpm"
    console.print(
        f"  {label}: RMSSD {_text(snapshot.rmssd_ms)} ms · heart rate {heart_rate} · "
        f"{format_freshness(snapshot.freshness_s, is_stale=snapshot.is_stale)}"
    )


@comodulation_app.command("show")
def comodulation_show(
    session_id: str = typer.Argument(..., help="Session identifier"),
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Show the latest Co-Modulation Index readback."""
    snapshot = _run_api_call(lambda: _build_client().get_session_physiology(session_id))
    summary = snapshot.comodulation or CoModulationSummary(session_id=snapshot.session_id)
    if json_output:
        _print_json(summary)
        return
    _render_comodulation(summary)


def _render_comodulation(summary: CoModulationSummary) -> None:
    display = build_co_modulation_display(summary)
    console.print(f"{display.title}: {display.primary}")
    console.print(display.secondary)
    console.print(display.detail)
    window_start = format_timestamp(summary.window_start_utc)
    window_end = format_timestamp(summary.window_end_utc)
    console.print(f"Window: {window_start} to {window_end}")
    console.print(f"Paired observations: {summary.n_paired_observations}")
    console.print(f"Coverage: {format_percentage(summary.coverage_ratio, digits=0)}")


@cloud_app.command("status")
def cloud_status(json_output: bool = typer.Option(False, "--json", help="Render DTO JSON")) -> None:
    """Show cloud sign-in and outbox readbacks."""
    auth = _run_api_call(lambda: _build_client().get_cloud_auth_status())
    outbox = _run_api_call(lambda: _build_client().get_cloud_outbox_summary())
    if json_output:
        console.print(
            json.dumps(
                {
                    "auth": auth.model_dump(mode="json"),
                    "outbox": outbox.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return
    _render_cloud_auth(auth)
    _render_cloud_outbox(outbox)


@cloud_app.command("sign-in")
def cloud_sign_in(
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Run cloud browser sign-in through the loopback API."""
    result = _run_api_call(lambda: _build_client().post_cloud_sign_in())
    if json_output:
        _print_json(result)
    else:
        _render_cloud_sign_in_result(result)
    if result.status is CloudActionStatus.FAILED:
        raise typer.Exit(1)


@cloud_app.command("refresh-experiments")
def cloud_refresh_experiments(
    json_output: bool = typer.Option(False, "--json", help="Render DTO JSON"),
) -> None:
    """Refresh signed experiment bundle through the loopback API."""
    result = _run_api_call(lambda: _build_client().post_experiment_bundle_refresh())
    if json_output:
        _print_json(result)
    else:
        _render_cloud_refresh_result(result)
    if result.status is CloudExperimentRefreshStatus.FAILED:
        raise typer.Exit(1)


def _render_cloud_auth(status: CloudAuthStatus) -> None:
    console.print(f"Cloud sign-in: {status.state.value.replace('_', ' ')}")
    console.print(f"Checked: {format_timestamp(status.checked_at_utc)}")
    if status.message:
        console.print(status.message)
    if status.retryable:
        console.print("Retryable: yes")


def _render_cloud_outbox(summary: CloudOutboxSummary) -> None:
    console.print("Cloud outbox")
    console.print(f"Generated: {format_timestamp(summary.generated_at_utc)}")
    console.print(f"Pending: {summary.pending_count}")
    console.print(f"In flight: {summary.in_flight_count}")
    console.print(f"Retry scheduled: {summary.retry_scheduled_count}")
    console.print(f"Dead-letter: {summary.dead_letter_count}")
    console.print(f"Redacted: {summary.redacted_count}")
    if summary.earliest_next_attempt_utc is not None:
        console.print(f"Next attempt: {format_timestamp(summary.earliest_next_attempt_utc)}")
    if summary.last_error:
        console.print(f"Last error: {summary.last_error}")


def _render_cloud_sign_in_result(result: CloudSignInResult) -> None:
    auth_state = result.auth_state.value.replace("_", " ")
    console.print(f"Cloud sign-in {result.status.value}: {auth_state}")
    console.print(result.message)
    if result.error_code is not None:
        console.print(f"Error code: {result.error_code.value}")
    if result.retryable:
        console.print("Retryable: yes")


def _render_cloud_refresh_result(result: ExperimentBundleRefreshResult) -> None:
    console.print(f"Experiment refresh {result.status.value}")
    console.print(result.message)
    if result.bundle_id is not None:
        console.print(f"Bundle: {result.bundle_id}")
    console.print(f"Experiments: {result.experiment_count}")
    if result.error_code is not None:
        console.print(f"Error code: {result.error_code.value}")
    if result.retryable:
        console.print("Retryable: yes")


def main() -> None:
    """Run the CLI. Kept as a function so scripts/__main__.py can import it."""
    app()


if __name__ == "__main__":
    main()
