"""
Tests for packages/ml_core/semantic.py — Phase 1 validation.

Verifies SemanticEvaluator against §8.1–8.3:
deterministic params, structured output, retry logic.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from packages.ml_core.semantic import (
    LLM_PARAMS,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    SemanticEvaluator,
)


class TestSemanticConstants:
    """§8.1–8.3 — Configuration constants."""

    def test_deterministic_params(self) -> None:
        """§8.1 — temperature=0, top_p=1.0, seed=42."""
        assert LLM_PARAMS["temperature"] == 0.0
        assert LLM_PARAMS["top_p"] == 1.0
        assert LLM_PARAMS["seed"] == 42
        assert LLM_PARAMS["max_tokens"] == 500

    def test_output_schema_required_fields(self) -> None:
        """§8.2 — Schema requires reasoning, is_match, confidence_score."""
        assert set(OUTPUT_SCHEMA["required"]) == {
            "reasoning",
            "is_match",
            "confidence_score",
        }
        assert OUTPUT_SCHEMA["additionalProperties"] is False

    def test_system_prompt_defined(self) -> None:
        """§8.3 — Canonical system prompt is non-empty."""
        assert len(SYSTEM_PROMPT) > 0
        assert "linguistic evaluation" in SYSTEM_PROMPT


@pytest.fixture()
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


@pytest.fixture()
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock openai into sys.modules."""
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", mock)
    return mock


def _make_response(content: str) -> MagicMock:
    """Build a mock ChatCompletion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestSemanticEvaluator:
    """§8 — Deterministic semantic evaluation via Azure OpenAI."""

    def test_evaluate_returns_validated_result(
        self, env_vars: None, mock_openai: MagicMock
    ) -> None:
        """§8.1–8.3 — Returns validated dict on success."""
        response_content = json.dumps(
            {
                "reasoning": "Semantically equivalent greeting.",
                "is_match": True,
                "confidence_score": 0.95,
            }
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(
            response_content
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator()
        result = evaluator.evaluate("Hello!", "Hi there!")

        assert result is not None
        assert result["is_match"] is True
        assert result["confidence_score"] == 0.95
        assert "reasoning" in result

    def test_evaluate_uses_deterministic_params(
        self, env_vars: None, mock_openai: MagicMock
    ) -> None:
        """§8.1 — temperature=0, top_p=1.0, seed=42 passed to API."""
        response_content = json.dumps(
            {"reasoning": "Match.", "is_match": True, "confidence_score": 0.9}
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(
            response_content
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator()
        evaluator.evaluate("Hello!", "Hi!")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["top_p"] == 1.0
        assert call_kwargs["seed"] == 42

    def test_evaluate_retries_once_then_returns_none(
        self, env_vars: None, mock_openai: MagicMock
    ) -> None:
        """§4.D contract — LLM timeout retries once before returning None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator()
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is None
        assert mock_client.chat.completions.create.call_count == 2

    def test_evaluate_invalid_json_retries(
        self, env_vars: None, mock_openai: MagicMock
    ) -> None:
        """§8.2 — Invalid JSON triggers retry then returns None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response("not json")
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator()
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is None

    def test_init_client_called_once(
        self, env_vars: None, mock_openai: MagicMock
    ) -> None:
        """§8.1 — Client initialized lazily on first call."""
        response_content = json.dumps(
            {"reasoning": "Match.", "is_match": True, "confidence_score": 0.9}
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(
            response_content
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator()
        evaluator.evaluate("Hello!", "Hi!")
        evaluator.evaluate("Bye!", "Goodbye!")

        mock_openai.AzureOpenAI.assert_called_once()
