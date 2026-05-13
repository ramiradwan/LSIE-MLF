"""
Canonical semantic-evaluation and stimulus-contract payloads (§8.3).

The module defines the bounded reason-code and scorer-method vocabularies,
plus the shared typed stimulus definition used across admin, cloud, runtime,
and attribution surfaces. It accepts only canonical stimulus/response fields
and leaves method/version transport metadata to downstream enrichment; it does
not persist unbounded semantic rationales or alter the §7B reward path (§8.6).
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

ResponseRegistrationStatus = Literal[
    "observable_response",
    "no_observable_response",
    "ambiguous_response",
    "invalid_trigger",
    "pending",
]

ResponseReasonCode = Literal[
    "response_read_aloud",
    "response_direct_answer",
    "response_username_ack",
    "response_semantic_ack",
    "response_chat_reply",
    "response_positive_affect_only",
    "response_timeout",
    "no_observable_response",
    "ambiguous_response",
    "invalid_trigger",
]

StimulusModality = Literal[
    "spoken_greeting",
    "written_comment",
    "gift",
    "follow",
    "question",
    "cta",
    "product_pitch",
    "discount_offer",
    "profile_interaction",
    "dm_follow_up",
    "other",
]


class StimulusPayload(BaseModel):
    """Validate the typed payload content for one operator stimulus."""

    content_type: Literal["text"] = "text"
    text: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StimulusDefinition(BaseModel):
    """Validate the canonical stimulus definition attached to an experiment arm."""

    stimulus_modality: StimulusModality
    stimulus_payload: StimulusPayload
    expected_stimulus_rule: str = Field(..., min_length=1)
    expected_response_rule: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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
        ...,
        description="Whether the observed response matches the expected response rule.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic match probability/confidence between the observed response and "
            "the expected response rule, from 0.00 to 1.00."
        ),
    )

    model_config = ConfigDict(extra="forbid")
