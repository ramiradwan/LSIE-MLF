"""Tests for the Typer-based Operator CLI — §4.E.1, §4.C, §2 step 7.

Covers every command group (session, experiment, encounter, metrics,
physiology, comodulation, stimulus) plus API transport helpers.

The CLI is exercised via ``typer.testing.CliRunner``. API calls are
mocked at module level on ``_api_get`` / ``_api_post`` so the tests
run offline.
"""

from __future__ import annotations

import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest
from typer.testing import CliRunner

from scripts.lsie_cli import _api_get, _api_post, _fmt_float, app

runner = CliRunner()


def _make_http_error(code: int, detail: str) -> HTTPError:
    payload = io.BytesIO(json.dumps({"detail": detail}).encode("utf-8"))
    return HTTPError("http://api.test", code, "error", Message(), payload)


class TestFmtFloat:
    def test_none(self) -> None:
        assert _fmt_float(None) == "(n/a)"

    def test_float(self) -> None:
        assert _fmt_float(0.85) == "0.8500"

    def test_string_passthrough(self) -> None:
        assert _fmt_float("bad") == "bad"


class TestApiHelpers:
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


class TestRootHelp:
    def test_help_lists_all_groups(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "session" in result.output
        assert "experiment" in result.output
        assert "encounter" in result.output
        assert "metrics" in result.output
        assert "physiology" in result.output
        assert "comodulation" in result.output
        assert "stimulus" in result.output


class TestSessionCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_session_list(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [
            {
                "session_id": "s1",
                "stream_url": "rtmp://example/live",
                "started_at": "2025-04-01T00:00:00",
                "ended_at": None,
                "metric_count": 5,
            }
        ]
        result = runner.invoke(app, ["session", "list"])
        assert result.exit_code == 0
        assert "s1" in result.output
        mock_get.assert_called_once_with("/api/v1/sessions")

    @patch("scripts.lsie_cli._api_get")
    def test_session_list_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = runner.invoke(app, ["session", "list"])
        assert result.exit_code == 0
        assert "No sessions" in result.output

    @patch("scripts.lsie_cli._api_get")
    def test_session_status(self, mock_get: MagicMock) -> None:
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
        result = runner.invoke(app, ["session", "status", "s1"])
        assert result.exit_code == 0
        assert "Session:  s1" in result.output
        assert "0.4500" in result.output
        mock_get.assert_called_once_with("/api/v1/sessions/s1")


class TestExperimentCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_experiment_list(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [{"experiment_id": "greeting_line_v1"}]
        result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0
        assert "greeting_line_v1" in result.output
        mock_get.assert_called_once_with("/api/v1/experiments")

    @patch("scripts.lsie_cli._api_get")
    def test_experiment_show_default(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "experiment_id": "greeting_line_v1",
            "arms": [
                {
                    "arm": "warm",
                    "alpha_param": 3.5,
                    "beta_param": 2.1,
                    "updated_at": "2025-04-01",
                },
            ],
        }
        result = runner.invoke(app, ["experiment", "show"])
        assert result.exit_code == 0
        assert "Experiment: greeting_line_v1" in result.output
        assert "warm" in result.output
        mock_get.assert_called_once_with("/api/v1/experiments/greeting_line_v1")

    @patch("scripts.lsie_cli._api_get")
    def test_experiment_show_custom(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"experiment_id": "my_exp", "arms": []}
        result = runner.invoke(app, ["experiment", "show", "my_exp"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("/api/v1/experiments/my_exp")


class TestEncounterCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_encounter_list_no_filters(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = runner.invoke(app, ["encounter", "list"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("/api/v1/encounters?limit=100")

    @patch("scripts.lsie_cli._api_get")
    def test_encounter_list_all_filters(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = runner.invoke(
            app,
            [
                "encounter",
                "list",
                "--experiment",
                "greeting_line_v1",
                "--arm",
                "warm",
                "--valid-only",
                "--limit",
                "50",
            ],
        )
        assert result.exit_code == 0
        call_args = mock_get.call_args[0][0]
        assert "experiment_id=greeting_line_v1" in call_args
        assert "arm=warm" in call_args
        assert "valid_only=true" in call_args
        assert "limit=50" in call_args

    @patch("scripts.lsie_cli._api_get")
    def test_encounter_summary_default(self, mock_get: MagicMock) -> None:
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
        result = runner.invoke(app, ["encounter", "summary"])
        assert result.exit_code == 0
        assert "greeting_line_v1" in result.output
        assert "warm" in result.output
        mock_get.assert_called_once_with("/api/v1/encounters/greeting_line_v1/summary")


class TestMetricsCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_metrics_au12(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [
            {"segment_id": "seg1", "timestamp_utc": "2025-04-01T00:00:00", "au12_intensity": 0.3},
        ]
        result = runner.invoke(app, ["metrics", "au12", "s1"])
        assert result.exit_code == 0
        assert "seg1" in result.output
        mock_get.assert_called_once_with("/api/v1/metrics/s1/au12")

    @patch("scripts.lsie_cli._api_get")
    def test_metrics_acoustic(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = runner.invoke(app, ["metrics", "acoustic", "s1"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("/api/v1/metrics/s1/acoustic")


class TestPhysiologyCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_physiology_show_latest(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [
            {
                "subject_role": "streamer",
                "segment_id": "seg1",
                "rmssd_ms": 52.1,
                "heart_rate_bpm": 68,
                "freshness_s": 42.0,
                "is_stale": False,
                "provider": "oura",
                "source_timestamp_utc": "2025-04-01T00:00:00",
            }
        ]
        result = runner.invoke(app, ["physiology", "show", "s1"])
        assert result.exit_code == 0
        assert "streamer" in result.output
        assert "oura" in result.output
        mock_get.assert_called_once_with("/api/v1/physiology/s1")

    @patch("scripts.lsie_cli._api_get")
    def test_physiology_show_series(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = runner.invoke(app, ["physiology", "show", "s1", "--series", "--limit", "200"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("/api/v1/physiology/s1?series=true&limit=200")


class TestComodulationCommands:
    @patch("scripts.lsie_cli._api_get")
    def test_comodulation_show(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [
            {
                "window_end_utc": "2025-04-01T00:05:00",
                "window_minutes": 5,
                "co_modulation_index": 0.61,
                "n_paired_observations": 60,
                "coverage_ratio": 0.95,
                "streamer_rmssd_mean": 52.1,
                "operator_rmssd_mean": 60.8,
            }
        ]
        result = runner.invoke(app, ["comodulation", "show", "s1"])
        assert result.exit_code == 0
        assert "0.61" in result.output
        mock_get.assert_called_once_with("/api/v1/comodulation/s1?limit=100")


class TestStimulusCommands:
    @patch("scripts.lsie_cli._api_post")
    def test_stimulus_inject_triggered(self, mock_post: MagicMock) -> None:
        mock_post.return_value = {"status": "triggered"}
        result = runner.invoke(app, ["stimulus", "inject"])
        assert result.exit_code == 0
        assert "Stimulus injected" in result.output
        mock_post.assert_called_once_with("/api/v1/stimulus")

    @patch("scripts.lsie_cli._api_post")
    def test_stimulus_inject_with_warning(self, mock_post: MagicMock) -> None:
        mock_post.return_value = {
            "status": "published",
            "receivers": 0,
            "warning": "No orchestrator instance is currently listening.",
        }
        result = runner.invoke(app, ["stimulus", "inject"])
        assert result.exit_code == 0
        assert "published" in result.output
        assert "Warning:" in result.output
