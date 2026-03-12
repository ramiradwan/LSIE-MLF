"""
Semantic Evaluation Schema — §8.2 Expected Output Schema

Pydantic model for the deterministic LLM response from Azure OpenAI
semantic matching evaluation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SemanticEvaluationResult(BaseModel):
    """
    §8.2 — Structured output enforced on Azure OpenAI Chat Completion.
    additionalProperties: false
    """

    reasoning: str = Field(..., description="Reasoning explanation evaluating semantic similarity.")
    is_match: bool = Field(
        ..., description="Whether the utterance matches the expected greeting rule."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.00 and 1.00.",
    )

    model_config = {"extra": "forbid"}
