"""Tests for Operator Session CLI — §4.E.1, §4.C, §2 step 7.

Covers:
  - Argument parser construction and validation
  - Command dispatch for all subcommands
  - API response formatting (_format_table, _fmt_float)
  - Error handling for unreachable API and HTTP errors
"""

from __future__ import annotations

import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from scripts.lsie_cli import (
    _api_get,
    _api_post,
    _fmt_float,
    _format_table,
    build_parser,
    cmd_experiment_show,
    cmd_experiment_summary,
    cmd_session_inject,
    cmd_session_list,
    cmd_session_status,
)


def _make_http_error(code: int, detail: str) -> HTTPError:
    """Build an HTTPError with a JSON detail payload."""
    payload = io.BytesIO(json.dumps({"detail": detail}).encode("utf-8"))
    return HTTPError("http://api.test", code, "error", Message(), payload)


class TestBuildParser:
    """Argument parser accepts all expected command shapes."""

    def test_session_list(self) -> None:
        args = build_parser().parse_args(["session", "list"])
        assert args.command == "session"
        assert args.subcommand == "list"

    def test_session_status(self) -> None:
        args = build_parser().parse_args(["session", "status", "abc-123"])
        assert args.command == "session"
        assert args.subcommand == "status"
        assert args.session_id == "abc-123"

    def test_session_inject(self) -> None:
        args = build_parser().parse_args(["session", "inject"])
        assert args.command == "session"
        assert args.subcommand == "inject"

    def test_experiment_show_default(self) -> None:
        args = build_parser().parse_args(["experiment", "show"])
        assert args.command == "experiment"
        assert args.subcommand == "show"
        assert args.experiment_id == "greeting_line_v1"

    def test_experiment_show_custom(self) -> None:
        args = build_parser().parse_args(["experiment", "show", "my_exp"])
        assert args.experiment_id == "my_exp"

    def test_experiment_summary(self) -> None:
        args = build_parser().parse_args(["experiment", "summary", "exp_1"])
        assert args.command == "experiment"
        assert args.subcommand == "summary"
        assert args.experiment_id == "exp_1"

    def test_missing_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestFormatTable:
    """Table formatter produces aligned output."""

    def test_basic_table(self) -> None:
        rows = [{"a": "hello", "b": 42}, {"a": "x", "b": 1000}]
        result = _format_table(rows, ["a", "b"])
        lines = result.splitlines()
        assert len(lines) == 4
        assert lines[0].startswith("a")
        assert "hello" in lines[2]
        assert "1000" in lines[3]

    def test_empty_rows(self) -> None:
        assert _format_table([], ["a"]) == "(no data)"

    def test_missing_column_shows_empty(self) -> None:
        rows = [{"a": "value"}]
        result = _format_table(rows, ["a", "missing_col"])
        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[2].startswith("value")


class TestFmtFloat:
    """Float formatter handles None and conversion fallbacks."""

    def test_none(self) -> None:
        assert _fmt_float(None) == "(n/a)"

    def test_float(self) -> None:
        assert _fmt_float(0.85) == "0.8500"

    def test_string_passthrough(self) -> None:
        assert _fmt_float("bad") == "bad"


class TestApiHelpers:
    """API helpers normalize transport and HTTP failures."""

    @patch("scripts.lsie_cli.urlopen")
    def test_api_get_http_error(self, mock_urlopen: MagicMock, capsys: Any) -> None:
        mock_urlopen.side_effect = _make_http_error(503, "service unavailable")

        with pytest.raises(SystemExit):
            _api_get("/api/v1/sessions")

        captured = capsys.readouterr()
        assert captured.err.strip() == "API error (503): service unavailable"

    @patch("scripts.lsie_cli.urlopen")
    def test_api_get_url_error(self, mock_urlopen: MagicMock, capsys: Any) -> None:
        mock_urlopen.side_effect = URLError("connection refused")

        with patch("scripts.lsie_cli.API_BASE", "http://api.test"), pytest.raises(SystemExit):
            _api_get("/api/v1/sessions")

        captured = capsys.readouterr()
        assert captured.err.splitlines() == [
            "Cannot reach API at http://api.test: connection refused",
            "Is the API container running?",
        ]

    @patch("scripts.lsie_cli.urlopen")
    def test_api_post_http_error(self, mock_urlopen: MagicMock, capsys: Any) -> None:
        mock_urlopen.side_effect = _make_http_error(500, "stimulus failed")

        with pytest.raises(SystemExit):
            _api_post("/api/v1/stimulus")

        captured = capsys.readouterr()
        assert captured.err.strip() == "API error (500): stimulus failed"

    @patch("scripts.lsie_cli.urlopen")
    def test_api_post_url_error(self, mock_urlopen: MagicMock, capsys: Any) -> None:
        mock_urlopen.side_effect = URLError("connection refused")

        with patch("scripts.lsie_cli.API_BASE", "http://api.test"), pytest.raises(SystemExit):
            _api_post("/api/v1/stimulus")

        captured = capsys.readouterr()
        assert captured.err.splitlines() == [
            "Cannot reach API at http://api.test: connection refused",
        ]


class TestSessionList:
    @patch("scripts.lsie_cli._api_get")
    def test_prints_table(self, mock_get: MagicMock, capsys: Any) -> None:
        mock_get.return_value = [
            {
                "session_id": "s1",
                "stream_url": "rtmp://example/live",
                "started_at": "2025-04-01T00:00:00",
                "ended_at": None,
                "metric_count": 5,
            }
        ]
        args = build_parser().parse_args(["session", "list"])
        cmd_session_list(args)
        out = capsys.readouterr().out
        assert "s1" in out
        assert "metric_count" in out
        mock_get.assert_called_once_with("/api/v1/sessions")

    @patch("scripts.lsie_cli._api_get")
    def test_empty_sessions(self, mock_get: MagicMock, capsys: Any) -> None:
        mock_get.return_value = []
        args = build_parser().parse_args(["session", "list"])
        cmd_session_list(args)
        out = capsys.readouterr().out
        assert "No sessions" in out


class TestSessionStatus:
    @patch("scripts.lsie_cli._api_get")
    def test_prints_detail(self, mock_get: MagicMock, capsys: Any) -> None:
        mock_get.return_value = {
            "session_id": "s1",
            "stream_url": "rtmp://example/live",
            "started_at": "2025-04-01T00:00:00",
            "ended_at": None,
            "summary": {
                "total_segments": 10,
                "avg_au12": 0.45,
                "avg_pitch_f0": 220.0,
                "avg_jitter": 0.02,
                "avg_shimmer": 0.03,
                "first_segment_at": "2025-04-01T00:01:00",
                "last_segment_at": "2025-04-01T00:05:00",
            },
        }
        args = build_parser().parse_args(["session", "status", "s1"])
        cmd_session_status(args)
        out = capsys.readouterr().out
        assert "Session:  s1" in out
        assert "Segments:" in out
        assert "0.4500" in out
        mock_get.assert_called_once_with("/api/v1/sessions/s1")


class TestSessionInject:
    @patch("scripts.lsie_cli._api_post")
    def test_triggered(self, mock_post: MagicMock, capsys: Any) -> None:
        mock_post.return_value = {"status": "triggered"}
        args = build_parser().parse_args(["session", "inject"])
        cmd_session_inject(args)
        out = capsys.readouterr().out
        assert out.strip() == "Stimulus injected. Calibration phase ended."
        mock_post.assert_called_once_with("/api/v1/stimulus")

    @patch("scripts.lsie_cli._api_post")
    def test_published_with_warning(self, mock_post: MagicMock, capsys: Any) -> None:
        mock_post.return_value = {
            "status": "published",
            "receivers": 0,
            "warning": "No orchestrator instance is currently listening.",
        }
        args = build_parser().parse_args(["session", "inject"])
        cmd_session_inject(args)
        out = capsys.readouterr().out
        assert "published" in out
        assert "Warning:" in out


class TestExperimentShow:
    @patch("scripts.lsie_cli._api_get")
    def test_prints_arms(self, mock_get: MagicMock, capsys: Any) -> None:
        mock_get.return_value = {
            "experiment_id": "greeting_line_v1",
            "arms": [
                {"arm": "warm", "alpha_param": 3.5, "beta_param": 2.1, "updated_at": "2025-04-01"},
                {"arm": "direct", "alpha_param": 1.0, "beta_param": 1.0, "updated_at": None},
            ],
        }
        args = build_parser().parse_args(["experiment", "show", "greeting_line_v1"])
        cmd_experiment_show(args)
        out = capsys.readouterr().out
        assert "Experiment: greeting_line_v1" in out
        assert "warm" in out
        assert "3.5" in out
        mock_get.assert_called_once_with("/api/v1/experiments/greeting_line_v1")


class TestExperimentSummary:
    @patch("scripts.lsie_cli._api_get")
    def test_prints_summary(self, mock_get: MagicMock, capsys: Any) -> None:
        mock_get.return_value = [
            {
                "arm": "warm",
                "encounter_count": 12,
                "valid_count": 10,
                "avg_reward": 0.72,
                "avg_valid_reward": 0.80,
                "gate_rate": 0.83,
                "avg_frames": 45,
            }
        ]
        args = build_parser().parse_args(["experiment", "summary", "greeting_line_v1"])
        cmd_experiment_summary(args)
        out = capsys.readouterr().out
        assert "Encounter summary for: greeting_line_v1" in out
        assert "warm" in out
        assert "12" in out
        mock_get.assert_called_once_with("/api/v1/encounters/greeting_line_v1/summary")
