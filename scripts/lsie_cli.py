"""Operator Session CLI — §4.E.1, §4.C, §2 step 7.

Thin command-line interface wrapping the LSIE-MLF REST API for
streamlined operator workflow during live stream experiments.

Entry point: python -m scripts.lsie_cli <command> [options]

Commands:
    session list            — List all sessions with metric counts
    session status <id>     — Session detail with summary metrics
    session inject          — Trigger stimulus injection (greeting delivered)
    experiment show <id>    — Arm state with alpha/beta posteriors
    experiment summary <id> — Encounter-level reward summary per arm

§4.E.1 — Operator intervention bridge and experiment inspection.
§4.C   — Session lifecycle visibility.
§2 step 7 — All data access through parameterized API endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE: str = os.environ.get("LSIE_API_URL", "http://localhost:8000").rstrip("/")
_DEFAULT_EXPERIMENT_ID = "greeting_line_v1"


def _api_get(path: str) -> Any:
    """Perform a GET request and return the parsed JSON body."""
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
        print("Is the API container running?", file=sys.stderr)
        raise SystemExit(1) from exc


def _api_post(path: str) -> Any:
    """Perform a POST request and return the parsed JSON body."""
    url = f"{API_BASE}{path}"
    request = Request(url, method="POST", data=b"")
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


def _format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Format rows as an aligned plain-text table."""
    if not rows:
        return "(no data)"

    widths = [len(column) for column in columns]
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered = [str(row.get(column, "")) for column in columns]
        rendered_rows.append(rendered)
        for index, value in enumerate(rendered):
            widths[index] = max(widths[index], len(value))

    lines = ["  ".join(column.ljust(widths[index]) for index, column in enumerate(columns))]
    lines.append("  ".join("-" * width for width in widths))
    for rendered in rendered_rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(rendered)))
    return "\n".join(lines)


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


def cmd_session_list(args: argparse.Namespace) -> None:
    """GET /api/v1/sessions — list sessions."""
    _ = args
    sessions = _api_get("/api/v1/sessions")
    if not sessions:
        print("No sessions found. Start a stream to generate data.")
        return
    columns = ("session_id", "stream_url", "started_at", "ended_at", "metric_count")
    print(_format_table(sessions, columns))


def cmd_session_status(args: argparse.Namespace) -> None:
    """GET /api/v1/sessions/{id} — show session details and summary metrics."""
    data = _api_get(f"/api/v1/sessions/{args.session_id}")
    print(f"Session:  {data['session_id']}")
    print(f"Stream:   {data.get('stream_url', '(none)')}")
    print(f"Started:  {data.get('started_at', '(unknown)')}")
    print(f"Ended:    {data.get('ended_at', '(active)')}")

    summary = data.get("summary")
    if not summary:
        print("\n(no metrics yet)")
        return

    print()
    print(f"Segments:    {summary.get('total_segments', 0)}")
    print(f"Avg AU12:    {_fmt_float(summary.get('avg_au12'))}")
    print(f"Avg Pitch:   {_fmt_float(summary.get('avg_pitch_f0'))} Hz")
    print(f"Avg Jitter:  {_fmt_float(summary.get('avg_jitter'))}")
    print(f"Avg Shimmer: {_fmt_float(summary.get('avg_shimmer'))}")
    print(f"Time range:  {summary.get('first_segment_at', '?')} → {summary.get('last_segment_at', '?')}")


def cmd_session_inject(args: argparse.Namespace) -> None:
    """POST /api/v1/stimulus — publish a stimulus trigger."""
    _ = args
    result = _api_post("/api/v1/stimulus")
    status = result.get("status", "unknown")
    if status == "triggered":
        print("Stimulus injected. Calibration phase ended.")
        return

    print(f"Stimulus response: {status}")
    warning = result.get("warning")
    if warning:
        print(f"Warning: {warning}")


def cmd_experiment_show(args: argparse.Namespace) -> None:
    """GET /api/v1/experiments/{id} — show experiment arm state."""
    data = _api_get(f"/api/v1/experiments/{args.experiment_id}")
    print(f"Experiment: {data['experiment_id']}")
    print()

    arms = data.get("arms", [])
    if not arms:
        print("(no arms registered)")
        return

    columns = ("arm", "alpha_param", "beta_param", "updated_at")
    print(_format_table(arms, columns))


def cmd_experiment_summary(args: argparse.Namespace) -> None:
    """GET /api/v1/encounters/{id}/summary — show reward summary per arm."""
    summary_rows = _api_get(f"/api/v1/encounters/{args.experiment_id}/summary")
    columns = (
        "arm",
        "encounter_count",
        "valid_count",
        "avg_reward",
        "avg_valid_reward",
        "gate_rate",
        "avg_frames",
    )
    print(f"Encounter summary for: {args.experiment_id}")
    print()
    print(_format_table(summary_rows, columns))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="lsie",
        description="LSIE-MLF Operator CLI — §4.E.1",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL (default: $LSIE_API_URL or http://localhost:8000)",
    )

    top_level = parser.add_subparsers(dest="command", required=True)

    session_parser = top_level.add_parser("session", help="Session management")
    session_subcommands = session_parser.add_subparsers(dest="subcommand", required=True)
    session_subcommands.add_parser("list", help="List all sessions")

    status_parser = session_subcommands.add_parser("status", help="Show session details")
    status_parser.add_argument("session_id", help="Session identifier")

    session_subcommands.add_parser("inject", help="Trigger stimulus injection")

    experiment_parser = top_level.add_parser("experiment", help="Experiment inspection")
    experiment_subcommands = experiment_parser.add_subparsers(dest="subcommand", required=True)

    show_parser = experiment_subcommands.add_parser("show", help="Show experiment state")
    show_parser.add_argument(
        "experiment_id",
        nargs="?",
        default=_DEFAULT_EXPERIMENT_ID,
        help=f"Experiment ID (default: {_DEFAULT_EXPERIMENT_ID})",
    )

    summary_parser = experiment_subcommands.add_parser("summary", help="Show encounter summary")
    summary_parser.add_argument(
        "experiment_id",
        nargs="?",
        default=_DEFAULT_EXPERIMENT_ID,
        help=f"Experiment ID (default: {_DEFAULT_EXPERIMENT_ID})",
    )

    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.api_url:
        global API_BASE
        API_BASE = args.api_url.rstrip("/")

    handlers = {
        ("session", "list"): cmd_session_list,
        ("session", "status"): cmd_session_status,
        ("session", "inject"): cmd_session_inject,
        ("experiment", "show"): cmd_experiment_show,
        ("experiment", "summary"): cmd_experiment_summary,
    }
    handler = handlers.get((args.command, args.subcommand))
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    handler(args)


if __name__ == "__main__":
    main()
