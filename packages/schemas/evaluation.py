"""
Canonical semantic-evaluation payload contract (§8.3).

The module defines the bounded reason-code and scorer-method vocabularies plus
the strict Pydantic model for semantic evaluation output. It accepts only the
three canonical scorer fields and leaves method/version transport metadata to
downstream enrichment; it does not persist unbounded semantic rationales or
alter the §7B reward path (§8.6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SemanticReasonCode = Literal[
    "cross_encoder_high_match",
    "cross_encoder_high_nonmatch",
    "gray_band_llm_match",
    "gray_band_llm_nonmatch",
    "semantic_local_failure_fallback",
    "semantic_timeout",
    "semantic_error",
]

SEMANTIC_REASON_CODES: tuple[str, ...] = (
    "cross_encoder_high_match",
    "cross_encoder_high_nonmatch",
    "gray_band_llm_match",
    "gray_band_llm_nonmatch",
    "semantic_local_failure_fallback",
    "semantic_timeout",
    "semantic_error",
)

SemanticMethod = Literal["cross_encoder", "llm_gray_band"]

SEMANTIC_METHODS: tuple[str, ...] = (
    "cross_encoder",
    "llm_gray_band",
)


class SemanticEvaluationResult(BaseModel):
    """
    Validate the canonical §8.3 semantic scorer response.

    Accepts a bounded reason code, binary match gate, and [0, 1] confidence
    score and produces a strict Pydantic payload for Module D. It does not
    include transport metadata, free-form reasoning text, or reward-modulating
    probability fields.
    """

    reasoning: SemanticReasonCode = Field(
        ...,
        description="Bounded semantic reason code.",
    )
    is_match: bool = Field(
        ..., description="Whether the utterance matches the expected greeting rule."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic match probability/confidence between 0.00 and 1.00.",
    )

    model_config = ConfigDict(extra="forbid")
