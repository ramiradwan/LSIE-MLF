"""Tests for the Operator Console REST client.

All network calls are mocked at ``urlopen`` so the suite runs offline.
The focus here is on type narrowing and error surfacing — the console
UI depends on the client never raising ``SystemExit`` and always
yielding the shapes the view layer expects.
"""

from __future__ import annotations

import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from services.operator_console.api_client import ApiClient, ApiError


def _http_error(code: int, detail: str) -> HTTPError:
    payload = io.BytesIO(json.dumps({"detail": detail}).encode("utf-8"))
    return HTTPError("http://api.test", code, "error", Message(), payload)


def _mock_response(body: Any) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=None)
    return response


class TestListSessions:
    @patch("services.operator_console.api_client.urlopen")
    def test_returns_list_of_dicts(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response([{"session_id": "s1", "metric_count": 3}])
        client = ApiClient("http://api.test")
        rows = client.list_sessions()
        assert rows == [{"session_id": "s1", "metric_count": 3}]

    @patch("services.operator_console.api_client.urlopen")
    def test_filters_non_dict_entries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            [{"session_id": "s1"}, "garbage", 42, {"session_id": "s2"}]
        )
        client = ApiClient("http://api.test")
        rows = client.list_sessions()
        assert [row["session_id"] for row in rows] == ["s1", "s2"]

    @patch("services.operator_console.api_client.urlopen")
    def test_raises_on_non_list_shape(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"unexpected": "shape"})
        client = ApiClient("http://api.test")
        with pytest.raises(ApiError) as info:
            client.list_sessions()
        assert "expected list" in str(info.value)


class TestErrorHandling:
    @patch("services.operator_console.api_client.urlopen")
    def test_http_error_surfaces_status_code(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(503, "unavailable")
        client = ApiClient("http://api.test")
        with pytest.raises(ApiError) as info:
            client.list_sessions()
        assert info.value.status_code == 503
        assert "unavailable" in str(info.value)

    @patch("services.operator_console.api_client.urlopen")
    def test_url_error_becomes_api_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = URLError("connection refused")
        client = ApiClient("http://api.test")
        with pytest.raises(ApiError) as info:
            client.list_sessions()
        # long-running UI must never see SystemExit from the client
        assert "cannot reach http://api.test" in str(info.value)
        assert info.value.status_code is None

    @patch("services.operator_console.api_client.urlopen")
    def test_malformed_json_body_is_reported(self, mock_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = b"not json"
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=None)
        mock_urlopen.return_value = response
        client = ApiClient("http://api.test")
        with pytest.raises(ApiError):
            client.list_sessions()


class TestOtherEndpoints:
    @patch("services.operator_console.api_client.urlopen")
    def test_physiology_series_flag_builds_querystring(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response([])
        client = ApiClient("http://api.test")
        client.physiology_snapshot("s1", series=True, limit=50)
        called = mock_urlopen.call_args[0][0]
        assert "/api/v1/physiology/s1?series=true&limit=50" in called.full_url

    @patch("services.operator_console.api_client.urlopen")
    def test_inject_stimulus_posts_empty_body(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"status": "triggered"})
        client = ApiClient("http://api.test")
        result = client.inject_stimulus()
        assert result == {"status": "triggered"}
        called = mock_urlopen.call_args[0][0]
        assert called.get_method() == "POST"
