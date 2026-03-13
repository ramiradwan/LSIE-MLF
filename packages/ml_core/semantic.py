"""
Semantic Evaluation — §8 Deterministic LLM Prompt Specification

Azure OpenAI Chat Completion configured in deterministic mode for
robust semantic matching of spoken utterances against greeting rules.
"""

from __future__ import annotations

import os
from typing import Any

# §8.3 — Canonical system prompt
SYSTEM_PROMPT: str = (
    "You are a highly precise linguistic evaluation assistant.\n"
    "Your objective is to determine whether a transcribed 'Actual Utterance' "
    "is semantically equivalent to an 'Expected Greeting Rule'.\n\n"
    "Inputs:\n"
    "1. Expected Greeting Rule\n"
    "2. Actual Utterance\n\n"
    "Evaluation constraints:\n"
    "- Minor transcription errors and filler words are acceptable if the "
    "semantic intent remains unchanged.\n"
    "- Major deviations in subject matter must be rejected.\n\n"
    "Steps:\n"
    "1. Produce a reasoning explanation evaluating semantic similarity.\n"
    "2. Output boolean is_match.\n"
    "3. Output confidence_score between 0.00 and 1.00.\n\n"
    "Return the result strictly as valid JSON without markdown formatting."
)

# §8.1 — Inference parameter constraints
LLM_PARAMS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 500,
    "seed": 42,
}

# §8.2 — Expected output schema for structured outputs
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["reasoning", "is_match", "confidence_score"],
    "properties": {
        "reasoning": {"type": "string"},
        "is_match": {"type": "boolean"},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}


class SemanticEvaluator:
    """
    §8 — Deterministic semantic evaluation via Azure OpenAI.

    Configured with temperature=0, top_p=1.0, seed=42, and
    JSON Schema structured outputs to guarantee reproducible results.

    Failure mode (§4.D contract): LLM timeout retries once before
    recording null.
    """

    def __init__(self) -> None:
        self.endpoint: str = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.api_key: str = os.environ["AZURE_OPENAI_API_KEY"]
        self.deployment: str = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.api_version: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._client: Any = None  # Lazy-loaded AzureOpenAI

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client (§8.1)."""
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any] | None:
        """
        Evaluate semantic equivalence between expected and actual utterance.

        §8.1–8.3 — Deterministic Azure OpenAI Chat Completion with
        structured JSON output.

        Args:
            expected_greeting: The greeting rule to match against.
            actual_utterance: The transcribed spoken utterance.

        Returns:
            Dict with reasoning, is_match, confidence_score — or None on failure.
        """
        import json
        import logging

        logger: logging.Logger = logging.getLogger(__name__)

        if self._client is None:
            self._init_client()

        # §8.3 — User prompt with expected and actual utterance
        user_content: str = (
            f"Expected Greeting Rule: {expected_greeting}\n"
            f"Actual Utterance: {actual_utterance}"
        )

        # §4.D contract — LLM timeout retries once before recording null
        for attempt in range(2):
            try:
                response: Any = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    # §8.1 — Deterministic inference parameters
                    temperature=LLM_PARAMS["temperature"],
                    top_p=LLM_PARAMS["top_p"],
                    max_tokens=LLM_PARAMS["max_tokens"],
                    seed=LLM_PARAMS["seed"],
                    # §8.2 — Structured JSON output
                    response_format={"type": "json_object"},
                )

                raw: str = response.choices[0].message.content
                result: dict[str, Any] = json.loads(raw)

                # §8.2 — Validate against expected output schema
                from packages.schemas.evaluation import SemanticEvaluationResult

                validated = SemanticEvaluationResult.model_validate(result)
                return validated.model_dump()

            except Exception:
                logger.warning(
                    "Semantic evaluation attempt %d failed", attempt + 1, exc_info=True
                )
                if attempt == 1:
                    # §4.D contract — record null after retry exhaustion
                    return None

        return None
