"""
Tests for services/worker/tasks/enrichment.py — Phase 4.3 validation.

Verifies scrape_context against:
  §4.F.1 — Patchright browser scraping
  §4.F contract — Failure modes: CAPTCHA, HTTP block, crash, timeout
  §12 Worker crash F — Retry 2x
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch


def _make_mock_pw(
    status: int = 200,
    title: str = "Test Page",
    body_text: str = "Hello world",
    goto_returns_none: bool = False,
) -> MagicMock:
    """Create mock sync_playwright context manager with full chain."""
    mock_page = MagicMock()

    if goto_returns_none:
        mock_page.goto.return_value = None
    else:
        mock_response = MagicMock()
        mock_response.status = status
        mock_page.goto.return_value = mock_response

    mock_page.title.return_value = title
    mock_page.inner_text.return_value = body_text
    mock_page.evaluate.return_value = [{"name": "description", "property": None, "content": "test"}]

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_pw = MagicMock()
    mock_pw.chromium.launch.return_value = mock_browser

    mock_sync_pw = MagicMock()
    mock_sync_pw.return_value.__enter__ = MagicMock(return_value=mock_pw)
    mock_sync_pw.return_value.__exit__ = MagicMock(return_value=False)

    return mock_sync_pw


def _get_scrape_context() -> Any:
    """Import scrape_context with mocked celery decorator."""
    mock_app = MagicMock()
    mock_app.task.return_value = lambda f: f
    celery_mod = ModuleType("celery")
    celery_mod.Task = object  # type: ignore[attr-defined]
    celery_app_mod = ModuleType("services.worker.celery_app")
    celery_app_mod.celery_app = mock_app  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {
            "celery": celery_mod,
            "services.worker.celery_app": celery_app_mod,
        },
    ):
        mod_name = "services.worker.tasks.enrichment"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        return mod.scrape_context


def _patch_pw(mock_sync_pw: MagicMock) -> Any:
    """Patch patchright.sync_api.sync_playwright in sys.modules."""
    mock_patchright_sync = MagicMock()
    mock_patchright_sync.sync_playwright = mock_sync_pw
    return patch.dict(
        sys.modules,
        {
            "patchright": MagicMock(),
            "patchright.sync_api": mock_patchright_sync,
        },
    )


class TestScrapeContext:
    """§4.F — Context enrichment via patchright."""

    def test_successful_scrape(self) -> None:
        """§4.F.1 — Successful scrape returns data with success=True."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw()

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert result["success"] is True
        assert result["source_url"] == "https://example.com"
        assert result["data"]["title"] == "Test Page"
        assert result["data"]["status_code"] == 200

    def test_http_403_returns_empty(self) -> None:
        """§4.F contract — CAPTCHA/HTTP blocking returns empty result."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw(status=403)

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert result["success"] is False
        assert result["data"] == {}

    def test_http_429_returns_empty(self) -> None:
        """§4.F contract — Rate limiting returns empty result."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw(status=429)

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert result["success"] is False

    def test_http_500_returns_empty(self) -> None:
        """§4.F contract — Server error returns empty result."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw(status=500)

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert result["success"] is False

    def test_null_response_returns_empty(self) -> None:
        """§4.F contract — No response returns empty result."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw(goto_returns_none=True)

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert result["success"] is False

    def test_browser_crash_retries(self) -> None:
        """§12 Worker crash F — Browser crash triggers retry."""
        fn = _get_scrape_context()
        mock_task = MagicMock()
        mock_task.MaxRetriesExceededError = Exception

        mock_sync_pw = MagicMock()
        mock_sync_pw.return_value.__enter__ = MagicMock(side_effect=RuntimeError("browser crash"))
        mock_sync_pw.return_value.__exit__ = MagicMock(return_value=False)

        with _patch_pw(mock_sync_pw):
            result = fn(mock_task, "https://example.com", "metadata")

        mock_task.retry.assert_called_once()
        assert result["success"] is False

    def test_result_structure(self) -> None:
        """Result dict has required keys."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw()

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert "source_url" in result
        assert "scraped_at_utc" in result
        assert "scrape_type" in result
        assert "data" in result
        assert "success" in result

    def test_text_content_capped(self) -> None:
        """Extracted text capped at 10000 chars."""
        fn = _get_scrape_context()
        long_text = "x" * 20000
        mock_sync_pw = _make_mock_pw(body_text=long_text)

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "metadata")

        assert len(result["data"]["text_content"]) == 10000

    def test_scrape_type_in_result(self) -> None:
        """scrape_type passed through to result."""
        fn = _get_scrape_context()
        mock_sync_pw = _make_mock_pw()

        with _patch_pw(mock_sync_pw):
            result = fn(MagicMock(), "https://example.com", "profile")

        assert result["scrape_type"] == "profile"
