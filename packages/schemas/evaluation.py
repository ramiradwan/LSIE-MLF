"""
Semantic Evaluation Schema — §8.3 Expected Output Schema

Pydantic model for the canonical deterministic semantic scorer payload. v3.4
keeps this scorer contract to the three §8.3 fields; D→E transport metadata
(``semantic_method`` and ``semantic_method_version``) is attached only after
canonical scorer validation.
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

SemanticMethod = Literal["cross_encoder", "llm_gray_band", "azure_llm_legacy"]

SEMANTIC_METHODS: tuple[str, ...] = (
    "cross_encoder",
    "llm_gray_band",
    "azure_llm_legacy",
)


class SemanticEvaluationResult(BaseModel):
    """
    §8.3 — Expected semantic evaluation JSON output contract.

    The ``reasoning`` field is a bounded reason code, not free-form rationale.
    ``extra='forbid'`` keeps the canonical scorer payload limited to the exact
    §8.3 fields. Downstream D→E metadata is appended after this validation step.
    """

    reasoning: SemanticReasonCode = Field(
        ...,
        description="Bounded semantic reason code; not a free-form rationale.",
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
