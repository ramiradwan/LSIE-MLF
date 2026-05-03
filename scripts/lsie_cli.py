"""Operator CLI — §4.E.1, §4.C, §2 step 7.

Typer-based command-line interface wrapping the LSIE-MLF REST API.
All data access goes through the API Server; the CLI never talks
to Postgres or Redis directly (§2 step 7).

Entry point: ``python -m scripts <group> <command> [options]``

Command groups:
    session       — list, status
    experiment    — list, show
    encounter     — list, summary
    metrics       — au12, acoustic
    physiology    — show
    comodulation  — show
    stimulus      — inject

§4.E.1    — Operator intervention bridge and experiment inspection.
§4.C      — Session lifecycle visibility.
§2 step 7 — All data access through parameterized API endpoints.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

_DEFAULT_API_BASE = "http://localhost:8000"
_DEFAULT_EXPERIMENT_ID = "greeting_line_v1"

API_BASE: str = os.environ.get("LSIE_API_URL", _DEFAULT_API_BASE).rstrip("/")

console = Console(highlight=False)

app = typer.Typer(
    name="lsie",
    help="LSIE-MLF Operator CLI — §4.E.1",
    no_args_is_help=True,
    add_completion=False,
)
session_app = typer.Typer(help="Session management", no_args_is_help=True)
experiment_app = typer.Typer(help="Thompson Sampling experiment inspection", no_args_is_help=True)
encounter_app = typer.Typer(help="Encounter log and reward summaries", no_args_is_help=True)
metrics_app = typer.Typer(help="Per-session metric time-series", no_args_is_help=True)
physiology_app = typer.Typer(help="Physiological snapshot readback (§4.E.2)", no_args_is_help=True)
comodulation_app = typer.Typer(help="Co-Modulation Index history (§7C)", no_args_is_help=True)
stimulus_app = typer.Typer(help="Operator stimulus trigger (§4.E.1)", no_args_is_help=True)

app.add_typer(session_app, name="session")
app.add_typer(experiment_app, name="experiment")
app.add_typer(encounter_app, name="encounter")
app.add_typer(metrics_app, name="metrics")
app.add_typer(physiology_app, name="physiology")
app.add_typer(comodulation_app, name="comodulation")
app.add_typer(stimulus_app, name="stimulus")


# ---------------------------------------------------------------------------
# HTTP helpers — stdlib only (no new runtime dependency beyond typer+rich)
# ---------------------------------------------------------------------------


def _api_get(path: str) -> Any:
    """Perform a GET request against the API server and return parsed JSON."""
    url = f"{API_BASE}{path}"
    request = Request(url, method="GET")
    request.add_header("Accept", "application/json")
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body).get("detail", body)
        except (json.JSONDecodeError, AttributeError):
            detail = body
        print(f"API error ({exc.code}): {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
    except URLError as exc:
        print(f"Cannot reach API at {API_BASE}: {exc.reason}", file=sys.stderr)
        print("Is the API Server running?", file=sys.stderr)
        raise SystemExit(1) from exc


def _api_post(path: str, payload: dict[str, Any] | None = None) -> Any:
    """Perform a POST request against the API server and return parsed JSON."""
    url = f"{API_BASE}{path}"
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, method="POST", data=body)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body).get("detail", body)
        except (json.JSONDecodeError, AttributeError):
            detail = body
        print(f"API error ({exc.code}): {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
    except URLError as exc:
        print(f"Cannot reach API at {API_BASE}: {exc.reason}", file=sys.stderr)
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: Any) -> str:
    """Format numeric values consistently for operator output."""
    if value is None:
        return "(n/a)"
    if not isinstance(value, (int, float, str)):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _render_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    title: str | None = None,
    empty_message: str = "(no data)",
) -> None:
    """Render a list of row-dicts as a rich table. Empty → plain message."""
    if not rows:
        console.print(empty_message)
        return

    table = Table(title=title, show_lines=False, pad_edge=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*[str(row.get(column, "")) for column in columns])
    console.print(table)


# ---------------------------------------------------------------------------
# Root callback — global --api-url option
# ---------------------------------------------------------------------------


@app.callback()
def _root(
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="API base URL (default: $LSIE_API_URL or http://localhost:8000)",
    ),
) -> None:
    """Global options applied before every command."""
    if api_url:
        global API_BASE
        API_BASE = api_url.rstrip("/")


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


@session_app.command("list")
def session_list() -> None:
    """GET /api/v1/sessions — list all sessions."""
    sessions = _api_get("/api/v1/sessions")
    if not sessions:
        console.print("No sessions found. Start a stream to generate data.")
        return
    _render_table(
        sessions,
        ["session_id", "stream_url", "started_at", "ended_at", "metric_count"],
    )


@session_app.command("status")
def session_status(session_id: str = typer.Argument(..., help="Session identifier")) -> None:
    """GET /api/v1/sessions/{id} — show session details and summary metrics."""
    data = _api_get(f"/api/v1/sessions/{session_id}")
    console.print(f"Session:  {data['session_id']}")
    console.print(f"Stream:   {data.get('stream_url', '(none)')}")
    console.print(f"Started:  {data.get('started_at', '(unknown)')}")
    console.print(f"Ended:    {data.get('ended_at', '(active)')}")

    summary = data.get("summary")
    if not summary:
        console.print("\n(no metrics yet)")
        return

    console.print("")
    console.print(f"Segments:    {summary.get('total_segments', 0)}")
    console.print(f"Avg AU12:    {_fmt_float(summary.get('avg_au12'))}")
    console.print(f"Avg F0 (measure): {_fmt_float(summary.get('avg_f0_mean_measure_hz'))} Hz")
    console.print(f"Avg Jitter (measure):  {_fmt_float(summary.get('avg_jitter_mean_measure'))}")
    console.print(f"Avg Shimmer (measure): {_fmt_float(summary.get('avg_shimmer_mean_measure'))}")
    console.print(
        f"Time range:  {summary.get('first_segment_at', '?')} → "
        f"{summary.get('last_segment_at', '?')}"
    )


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------


@experiment_app.command("list")
def experiment_list() -> None:
    """GET /api/v1/experiments — list experiment IDs."""
    rows = _api_get("/api/v1/experiments")
    if not rows:
        console.print("(no experiments registered)")
        return
    _render_table(rows, ["experiment_id"])


@experiment_app.command("show")
def experiment_show(
    experiment_id: str = typer.Argument(
        _DEFAULT_EXPERIMENT_ID,
        help=f"Experiment ID (default: {_DEFAULT_EXPERIMENT_ID})",
    ),
) -> None:
    """GET /api/v1/experiments/{id} — show Thompson Sampling arm state."""
    data = _api_get(f"/api/v1/experiments/{experiment_id}")
    console.print(f"Experiment: {data['experiment_id']}")
    console.print("")

    arms = data.get("arms", [])
    if not arms:
        console.print("(no arms registered)")
        return

    _render_table(arms, ["arm", "alpha_param", "beta_param", "updated_at"])


# ---------------------------------------------------------------------------
# encounter
# ---------------------------------------------------------------------------


@encounter_app.command("list")
def encounter_list(
    experiment: str | None = typer.Option(None, "--experiment", help="Filter by experiment_id"),
    arm: str | None = typer.Option(None, "--arm", help="Filter by arm"),
    valid_only: bool = typer.Option(
        False,
        "--valid-only",
        help="Only encounters with measurement-window AU12 frames",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
) -> None:
    """GET /api/v1/encounters — list encounter log entries."""
    params: list[str] = []
    if experiment is not None:
        params.append(f"experiment_id={experiment}")
    if arm is not None:
        params.append(f"arm={arm}")
    if valid_only:
        params.append("valid_only=true")
    params.append(f"limit={limit}")
    query = "&".join(params)
    rows = _api_get(f"/api/v1/encounters?{query}")
    _render_table(
        rows,
        [
            "id",
            "session_id",
            "arm",
            "timestamp_utc",
            "gated_reward",
            "p90_intensity",
            "semantic_gate",
            "n_frames_in_window",
            "au12_baseline_pre",
        ],
    )


@encounter_app.command("summary")
def encounter_summary(
    experiment_id: str = typer.Argument(
        _DEFAULT_EXPERIMENT_ID,
        help=f"Experiment ID (default: {_DEFAULT_EXPERIMENT_ID})",
    ),
) -> None:
    """GET /api/v1/encounters/{id}/summary — per-arm reward summary."""
    rows = _api_get(f"/api/v1/encounters/{experiment_id}/summary")
    console.print(f"Encounter summary for: {experiment_id}")
    console.print("")
    _render_table(
        rows,
        [
            "arm",
            "encounter_count",
            "valid_count",
            "avg_reward",
            "avg_valid_reward",
            "gate_rate",
            "avg_frames",
        ],
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@metrics_app.command("au12")
def metrics_au12(session_id: str = typer.Argument(..., help="Session identifier")) -> None:
    """GET /api/v1/metrics/{session_id}/au12 — AU12 intensity time-series."""
    rows = _api_get(f"/api/v1/metrics/{session_id}/au12")
    _render_table(rows, ["segment_id", "timestamp_utc", "au12_intensity"])


@metrics_app.command("acoustic")
def metrics_acoustic(session_id: str = typer.Argument(..., help="Session identifier")) -> None:
    """GET /api/v1/metrics/{session_id}/acoustic — canonical §7D acoustic series."""
    rows = _api_get(f"/api/v1/metrics/{session_id}/acoustic")
    _render_table(
        rows,
        [
            "segment_id",
            "timestamp_utc",
            "f0_mean_measure_hz",
            "jitter_mean_measure",
            "shimmer_mean_measure",
            "f0_delta_semitones",
            "jitter_delta",
            "shimmer_delta",
        ],
    )


# ---------------------------------------------------------------------------
# physiology
# ---------------------------------------------------------------------------


@physiology_app.command("show")
def physiology_show(
    session_id: str = typer.Argument(..., help="Session identifier"),
    series: bool = typer.Option(False, "--series", help="Return full time-series"),
    limit: int = typer.Option(500, "--limit", min=1, max=5000, help="Max rows in series mode"),
) -> None:
    """GET /api/v1/physiology/{session_id} — latest per-role snapshot or time-series."""
    query = f"?series=true&limit={limit}" if series else ""
    rows = _api_get(f"/api/v1/physiology/{session_id}{query}")
    _render_table(
        rows,
        [
            "subject_role",
            "segment_id",
            "rmssd_ms",
            "heart_rate_bpm",
            "freshness_s",
            "is_stale",
            "provider",
            "source_timestamp_utc",
        ],
    )


# ---------------------------------------------------------------------------
# comodulation
# ---------------------------------------------------------------------------


@comodulation_app.command("show")
def comodulation_show(
    session_id: str = typer.Argument(..., help="Session identifier"),
    limit: int = typer.Option(100, "--limit", min=1, max=2000),
) -> None:
    """GET /api/v1/comodulation/{session_id} — rolling Co-Modulation Index."""
    rows = _api_get(f"/api/v1/comodulation/{session_id}?limit={limit}")
    _render_table(
        rows,
        [
            "window_end_utc",
            "window_minutes",
            "co_modulation_index",
            "n_paired_observations",
            "coverage_ratio",
            "streamer_rmssd_mean",
            "operator_rmssd_mean",
        ],
    )


# ---------------------------------------------------------------------------
# stimulus
# ---------------------------------------------------------------------------


@stimulus_app.command("inject")
def stimulus_inject(
    session_id: str = typer.Argument(..., help="Session identifier"),
) -> None:
    """POST /api/v1/operator/sessions/{session_id}/stimulus — submit stimulus intent."""
    result = _api_post(
        f"/api/v1/operator/sessions/{session_id}/stimulus",
        {"client_action_id": str(uuid4())},
    )
    accepted = result.get("accepted", False)
    if accepted:
        console.print("Stimulus accepted. Calibration phase ended.")
        return

    console.print(f"Stimulus response: {result.get('message', 'not accepted')}")


def main() -> None:
    """Run the CLI. Kept as a function so scripts/__main__.py can import it."""
    app()


if __name__ == "__main__":
    main()
