"""
Tests for packages/ml_core/semantic.py — v3.4 deterministic semantic scorer.

Verifies the local Cross-Encoder primary path, exact gray-band routing,
Azure fallback gating, bounded reason codes, failure codes, and method metadata.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.ml_core.semantic import (
    CROSS_ENCODER_METHOD_VERSION,
    CROSS_ENCODER_MODEL_ID,
    CROSS_ENCODER_MODEL_VERSION,
    GRAY_BAND_FALLBACK_SCHEMA,
    GRAY_BAND_LOWER_THRESHOLD,
    GRAY_BAND_RESPONSE_FORMAT,
    LLM_PARAMS,
    MATCH_THRESHOLD,
    OUTPUT_SCHEMA,
    SEMANTIC_CALIBRATION_VERSION,
    SYSTEM_PROMPT,
    LocalCrossEncoderScorer,
    SemanticEvaluator,
    calibrate_cross_encoder_score,
)
from packages.schemas.evaluation import SEMANTIC_REASON_CODES


class TestSemanticConstants:
    """§8.1–8.3 — Configuration constants."""

    def test_pinned_cross_encoder_artifact_and_calibration(self) -> None:
        """§8.1 / §8.2.1 — Pinned local primary scorer metadata."""
        assert CROSS_ENCODER_MODEL_ID == "local://models/semantic/lsie-greeting-cross-encoder-v1"
        assert CROSS_ENCODER_MODEL_VERSION == "lsie-greeting-cross-encoder-v1.0.0"
        assert SEMANTIC_CALIBRATION_VERSION == "semantic-greeting-calibration-v1.0.0"
        assert CROSS_ENCODER_METHOD_VERSION == (
            "lsie-greeting-cross-encoder-v1.0.0+semantic-greeting-calibration-v1.0.0"
        )

    def test_exact_threshold_constants(self) -> None:
        """§8.2.1 — Gray band is 0.58 <= score < 0.72."""
        assert GRAY_BAND_LOWER_THRESHOLD == 0.58
        assert MATCH_THRESHOLD == 0.72

    def test_deterministic_fallback_params(self) -> None:
        """§8.1 — temperature=0, top_p=1.0, max_tokens=500, seed=42."""
        assert LLM_PARAMS["temperature"] == 0.0
        assert LLM_PARAMS["top_p"] == 1.0
        assert LLM_PARAMS["seed"] == 42
        assert LLM_PARAMS["max_tokens"] == 500

    def test_output_schema_required_fields_and_bounded_enums(self) -> None:
        """§8.3 — Canonical scorer schema excludes downstream metadata."""
        assert set(OUTPUT_SCHEMA["required"]) == {
            "reasoning",
            "is_match",
            "confidence_score",
        }
        assert OUTPUT_SCHEMA["properties"]["reasoning"]["enum"] == list(SEMANTIC_REASON_CODES)
        assert "semantic_method" not in OUTPUT_SCHEMA["properties"]
        assert "semantic_method_version" not in OUTPUT_SCHEMA["properties"]
        assert OUTPUT_SCHEMA["additionalProperties"] is False

        assert GRAY_BAND_FALLBACK_SCHEMA["properties"]["reasoning"]["enum"] == [
            "gray_band_llm_match",
            "gray_band_llm_nonmatch",
        ]
        assert GRAY_BAND_FALLBACK_SCHEMA["additionalProperties"] is False

    def test_system_prompt_requires_bounded_reason_codes(self) -> None:
        """§8.4 — Fallback prompt is the canonical bounded-code prompt."""
        assert len(SYSTEM_PROMPT) > 0
        assert "You are a deterministic semantic evaluation fallback." in SYSTEM_PROMPT
        assert "Return only the bounded reason code requested below." in SYSTEM_PROMPT
        assert "semantic_method" not in SYSTEM_PROMPT
        assert "gray_band_llm_match" in SYSTEM_PROMPT


@pytest.fixture()
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")


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


def _score(value: float) -> Any:
    """Return a deterministic injected primary scorer."""

    def scorer(_expected: str, _actual: str) -> float:
        return value

    return scorer


def _llm_response(*, is_match: bool, confidence_score: float) -> MagicMock:
    """Azure gray-band response matching the §8.4 provider schema."""
    return _make_response(
        json.dumps(
            {
                "reasoning": ("gray_band_llm_match" if is_match else "gray_band_llm_nonmatch"),
                "is_match": is_match,
                "confidence_score": confidence_score,
            }
        )
    )


class TestCrossEncoderPrimaryRouting:
    """§8.1 / §8.2.1 — Local Cross-Encoder live path and boundaries."""

    @pytest.mark.audit_item("13.27")
    @pytest.mark.parametrize(
        ("score", "expected_reason", "expected_match"),
        [
            (0.57, "cross_encoder_high_nonmatch", False),
            (0.72, "cross_encoder_high_match", True),
        ],
    )
    def test_direct_boundaries_do_not_invoke_azure(
        self,
        score: float,
        expected_reason: str,
        expected_match: bool,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """0.57 is direct non-match and 0.72 is direct match, not gray band."""
        evaluator = SemanticEvaluator(
            primary_scorer=_score(score),
            gray_band_fallback_enabled=True,
        )

        result = evaluator.evaluate("Hello!", "Hi there!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == expected_reason
        assert result["is_match"] is expected_match
        assert result["confidence_score"] == score
        mock_openai.AzureOpenAI.assert_not_called()

    @pytest.mark.audit_item("13.27")
    @pytest.mark.parametrize(
        ("score", "llm_match", "llm_confidence", "expected_reason"),
        [
            (0.58, True, 0.91, "gray_band_llm_match"),
            (0.7199, False, 0.41, "gray_band_llm_nonmatch"),
        ],
    )
    def test_enabled_gray_band_boundaries_invoke_azure_once(
        self,
        score: float,
        llm_match: bool,
        llm_confidence: float,
        expected_reason: str,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """0.58 and 0.7199 are gray-band routed only when fallback is enabled."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            is_match=llm_match,
            confidence_score=llm_confidence,
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator(
            primary_scorer=_score(score),
            gray_band_fallback_enabled=True,
        )
        result = evaluator.evaluate("Hello!", "Hi there!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == expected_reason
        assert result["is_match"] is llm_match
        assert result["confidence_score"] == llm_confidence
        assert result["reasoning"] in SEMANTIC_REASON_CODES
        assert mock_client.chat.completions.create.call_count == 1

    def test_gray_band_fallback_disabled_does_not_invoke_azure(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """A gray-band score emits bounded false/zero output when fallback is disabled."""
        evaluator = SemanticEvaluator(
            primary_scorer=_score(0.58),
            gray_band_fallback_enabled=False,
        )

        result = evaluator.evaluate("Hello!", "Hi there!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == "semantic_error"
        assert result["is_match"] is False
        assert result["confidence_score"] == 0.0
        mock_openai.AzureOpenAI.assert_not_called()

    def test_confidence_score_is_bounded_after_calibration(self) -> None:
        """Primary scorer logits are calibrated and bounded to [0, 1]."""
        result = SemanticEvaluator(
            primary_scorer=_score(999.0),
            gray_band_fallback_enabled=False,
        ).evaluate("Hello!", "Hello!")

        assert result is not None
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["confidence_score"] == 1.0
        assert calibrate_cross_encoder_score(-999.0) == 0.0


class TestGrayBandFallbackFailureHandling:
    """§12.4.1 — Deterministic bounded failure codes for fallback paths."""

    def test_fallback_uses_deterministic_params(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """§8.1 — Azure fallback uses deterministic Structured Outputs params."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            is_match=True,
            confidence_score=0.9,
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator(
            primary_scorer=_score(0.58),
            gray_band_fallback_enabled=True,
        )
        evaluator.evaluate("Hello!", "Hi!")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["top_p"] == 1.0
        assert call_kwargs["seed"] == 42
        assert call_kwargs["max_tokens"] == 500
        assert "max_completion_tokens" not in call_kwargs
        assert call_kwargs["response_format"] == GRAY_BAND_RESPONSE_FORMAT
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True
        assert call_kwargs["response_format"]["json_schema"]["schema"] == GRAY_BAND_FALLBACK_SCHEMA

    def test_fallback_timeout_retries_once_then_returns_timeout_code(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """§12.4.1 — Timeout retries once then emits semantic_timeout."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator(
            primary_scorer=_score(0.58),
            gray_band_fallback_enabled=True,
        )
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == "semantic_timeout"
        assert result["is_match"] is False
        assert result["confidence_score"] == 0.0
        assert mock_client.chat.completions.create.call_count == 2

    def test_fallback_parse_error_retries_once_then_returns_error_code(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """Invalid fallback output emits the bounded semantic_error code."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response("not json")
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator(
            primary_scorer=_score(0.7199),
            gray_band_fallback_enabled=True,
        )
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is not None
        assert result["reasoning"] == "semantic_error"
        assert result["is_match"] is False
        assert result["confidence_score"] == 0.0
        assert mock_client.chat.completions.create.call_count == 2

    def test_local_primary_failure_does_not_bypass_to_azure(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """Local scorer failures remain bounded locally even when fallback is enabled."""

        def failing_scorer(_expected: str, _actual: str) -> float:
            raise RuntimeError("model unavailable")

        evaluator = SemanticEvaluator(
            primary_scorer=failing_scorer,
            gray_band_fallback_enabled=True,
        )
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == "semantic_local_failure_fallback"
        assert result["is_match"] is False
        assert result["confidence_score"] == 0.0
        assert result["reasoning"] in SEMANTIC_REASON_CODES
        mock_openai.AzureOpenAI.assert_not_called()

    def test_client_initialized_once_for_repeated_gray_band_calls(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """§8.1 — Fallback client is initialized lazily and reused."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            is_match=True,
            confidence_score=0.9,
        )
        mock_openai.AzureOpenAI.return_value = mock_client

        evaluator = SemanticEvaluator(
            primary_scorer=_score(0.58),
            gray_band_fallback_enabled=True,
        )
        evaluator.evaluate("Hello!", "Hi!")
        evaluator.evaluate("Hello!", "Hi again!")

        mock_openai.AzureOpenAI.assert_called_once()
        assert mock_client.chat.completions.create.call_count == 2

    def test_local_primary_failure_without_fallback_is_bounded_locally(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        """Local scorer failures remain bounded locally when fallback is disabled."""

        def failing_scorer(_expected: str, _actual: str) -> float:
            raise RuntimeError("model unavailable")

        evaluator = SemanticEvaluator(
            primary_scorer=failing_scorer,
            gray_band_fallback_enabled=False,
        )
        result = evaluator.evaluate("Hello!", "Hi!")

        assert result is not None
        assert set(result) == {"reasoning", "is_match", "confidence_score"}
        assert result["reasoning"] == "semantic_local_failure_fallback"
        assert result["is_match"] is False
        assert result["confidence_score"] == 0.0
        mock_openai.AzureOpenAI.assert_not_called()

    def test_missing_local_model_uses_deterministic_lexical_fallback(
        self,
        env_vars: None,
        mock_openai: MagicMock,
    ) -> None:
        evaluator = SemanticEvaluator(
            primary_scorer=LocalCrossEncoderScorer(model_id="local://missing/semantic-model"),
            gray_band_fallback_enabled=True,
        )

        result = evaluator.evaluate("Say hello to the creator", "hello creator")

        assert result is not None
        assert result["reasoning"] == "cross_encoder_high_match"
        assert result["is_match"] is True
        assert result["confidence_score"] >= MATCH_THRESHOLD
        assert evaluator.last_semantic_method == "cross_encoder"
        assert evaluator.last_semantic_method_version == (
            f"{CROSS_ENCODER_METHOD_VERSION}+lexical-unavailable-fallback"
        )
        mock_openai.AzureOpenAI.assert_not_called()
