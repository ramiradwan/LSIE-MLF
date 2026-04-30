"""
Deterministic semantic scoring for greeting-rule evaluation (§8).

The module routes an expected greeting and transcribed utterance through a
local Cross-Encoder primary scorer, optionally invokes the configured Azure
OpenAI fallback only for scores in the gray band [0.58, 0.72), and validates the
canonical §8.3 payload of bounded reason code, binary match gate, and confidence
score. Method/version sidecar fields are recorded for downstream analytics;
semantic probability does not modulate the §7B reward path per §8.6.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from typing import Any, Final

from packages.schemas.evaluation import (
    SEMANTIC_REASON_CODES,
    SemanticEvaluationResult,
)

logger = logging.getLogger(__name__)

# §8.1 / §8.2.1 — Pinned deterministic semantic scorer registry entries.
CROSS_ENCODER_MODEL_ID: Final[str] = "local://models/semantic/lsie-greeting-cross-encoder-v1"
CROSS_ENCODER_MODEL_VERSION: Final[str] = "lsie-greeting-cross-encoder-v1.0.0"
SEMANTIC_CALIBRATION_VERSION: Final[str] = "semantic-greeting-calibration-v1.0.0"
CROSS_ENCODER_METHOD_VERSION: Final[str] = (
    f"{CROSS_ENCODER_MODEL_VERSION}+{SEMANTIC_CALIBRATION_VERSION}"
)

# §8.2.1 — Exact deterministic routing thresholds.
GRAY_BAND_LOWER_THRESHOLD: Final[float] = 0.58
MATCH_THRESHOLD: Final[float] = 0.72

# §8.4 — Canonical system prompt for the optional gray-band fallback.
SYSTEM_PROMPT: str = (
    "You are a deterministic semantic evaluation fallback.\n"
    "Your only task is to resolve ambiguous utterances that were routed into "
    "the gray band by the primary local semantic scorer.\n\n"
    "Inputs:\n"
    "1. Expected Greeting Rule\n"
    "2. Actual Utterance\n\n"
    "Evaluation constraints:\n"
    "- Minor transcription errors and filler words are acceptable if the "
    "semantic intent remains unchanged.\n"
    "- Major deviations in subject matter must be rejected.\n"
    "- Return only the bounded reason code requested below.\n\n"
    "Return strict JSON with exactly these fields:\n"
    '1. reasoning: bounded reason code, one of ["gray_band_llm_match", '
    '"gray_band_llm_nonmatch"]\n'
    "2. is_match: boolean\n"
    "3. confidence_score: number between 0.00 and 1.00\n\n"
    "Return the result strictly as valid JSON without markdown formatting."
)

# §8.1 — Inference parameter constraints for deterministic Azure fallback.
LLM_PARAMS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 500,
    "seed": 42,
}

# §8.3 — Canonical scorer output, no downstream metadata or unbounded prose fields.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["reasoning", "is_match", "confidence_score"],
    "properties": {
        "reasoning": {
            "type": "string",
            "enum": list(SEMANTIC_REASON_CODES),
            "description": "Bounded semantic reason code.",
        },
        "is_match": {"type": "boolean"},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

GRAY_BAND_PROVIDER_REASON_CODES: Final[tuple[str, str]] = (
    "gray_band_llm_match",
    "gray_band_llm_nonmatch",
)

# §8.4 — Provider schema narrows the gray-band fallback reason-code surface.
GRAY_BAND_FALLBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["reasoning", "is_match", "confidence_score"],
    "properties": {
        "reasoning": {
            "type": "string",
            "enum": list(GRAY_BAND_PROVIDER_REASON_CODES),
        },
        "is_match": {"type": "boolean"},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

GRAY_BAND_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "gray_band_semantic_evaluation",
        "strict": True,
        "schema": GRAY_BAND_FALLBACK_SCHEMA,
    },
}

PrimaryScorer = Callable[[str, str], float]


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Read a deterministic boolean feature flag from the environment."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_local_model_path(model_id: str) -> str:
    """Translate the spec's local:// model URI into a filesystem path."""

    if model_id.startswith("local://"):
        return model_id.removeprefix("local://")
    return model_id


def _extract_scalar(value: Any) -> float:
    """Coerce common model-return containers into a single float score."""

    if isinstance(value, list | tuple):
        if not value:
            raise ValueError("empty semantic score sequence")
        return _extract_scalar(value[0])

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _extract_scalar(tolist())

    item = getattr(value, "item", None)
    if callable(item):
        return _extract_scalar(item())

    return float(value)


def _clamp_probability(value: Any) -> float:
    """Clamp a finite numeric value to the closed probability interval [0, 1]."""

    score = _extract_scalar(value)
    if not math.isfinite(score):
        raise ValueError("semantic score must be finite")
    return min(1.0, max(0.0, score))


def calibrate_cross_encoder_score(raw_score: Any) -> float:
    """
    Apply the pinned v3.4 calibration mapping and bound the probability.

    The pinned ``semantic-greeting-calibration-v1.0.0`` mapping treats scores
    already in [0, 1] as calibrated probabilities and applies a deterministic
    logistic transform to out-of-range logits before the final [0, 1] clamp.
    """

    score = _extract_scalar(raw_score)
    if not math.isfinite(score):
        raise ValueError("semantic score must be finite")

    if 0.0 <= score <= 1.0:
        return score

    try:
        calibrated = 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        calibrated = 0.0 if score < 0.0 else 1.0
    return min(1.0, max(0.0, calibrated))


def _coerce_bool(value: Any) -> bool:
    """Coerce strict JSON-compatible boolean values without bool('false') bugs."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    raise ValueError("semantic is_match must be boolean")


def _is_timeout_error(exc: Exception) -> bool:
    """Classify timeout exceptions without depending on a specific SDK class."""

    if isinstance(exc, TimeoutError):
        return True
    exc_name = type(exc).__name__.lower()
    return "timeout" in exc_name or "timedout" in exc_name


class LocalCrossEncoderScorer:
    """
    Lazy wrapper around the local Cross-Encoder primary scorer.

    Accepts an expected greeting and actual utterance pair, loads the configured
    sentence-transformers model on first use, and produces a raw scalar score for
    SemanticEvaluator calibration. It does not apply thresholds, call the
    gray-band fallback, or emit the §8.3 result payload.
    """

    def __init__(
        self,
        *,
        model_id: str = CROSS_ENCODER_MODEL_ID,
        device_mode: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_mode = device_mode or os.environ.get("SEMANTIC_DEVICE_MODE", "cuda:0")
        self._model: Any = None

    def _init_model(self) -> None:
        """Load the pinned sentence-transformers CrossEncoder lazily."""

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            _resolve_local_model_path(self.model_id),
            device=self.device_mode,
        )

    def score(self, expected_greeting: str, actual_utterance: str) -> float:
        """Return the raw Cross-Encoder score for one greeting/utterance pair."""

        if self._model is None:
            self._init_model()

        raw_score = self._model.predict(  # type: ignore[union-attr]
            [(expected_greeting, actual_utterance)],
            show_progress_bar=False,
        )
        return _extract_scalar(raw_score)


class SemanticEvaluator:
    """
    Evaluate greeting semantics and emit the canonical §8.3 scorer payload.

    Accepts expected-greeting text and an actual utterance plus an optional primary
    scorer and gray-band fallback flag. Produces a dict containing only
    ``reasoning``, ``is_match``, and ``confidence_score`` while retaining method
    metadata on sidecar attributes. It does not persist rationales, expose
    free-form LLM text, or let confidence scores alter the §7B reward gate;
    fallback is only eligible for [0.58, 0.72) scores when enabled (§8.6).
    """

    def __init__(
        self,
        *,
        model_id: str = CROSS_ENCODER_MODEL_ID,
        device_mode: str | None = None,
        primary_scorer: PrimaryScorer | LocalCrossEncoderScorer | None = None,
        gray_band_fallback_enabled: bool | None = None,
        shadow_mode_enabled: bool | None = None,
        shadow_scorer: PrimaryScorer | LocalCrossEncoderScorer | None = None,
    ) -> None:
        self.gray_band_fallback_enabled = (
            _env_flag("SEMANTIC_GRAY_BAND_FALLBACK_ENABLED")
            if gray_band_fallback_enabled is None
            else gray_band_fallback_enabled
        )
        self.shadow_mode_enabled = (
            _env_flag("SEMANTIC_SHADOW_MODE_ENABLED")
            if shadow_mode_enabled is None
            else shadow_mode_enabled
        )
        self._primary_scorer = primary_scorer or LocalCrossEncoderScorer(
            model_id=model_id,
            device_mode=device_mode,
        )
        self._shadow_scorer = shadow_scorer
        self.last_semantic_method: str | None = None
        self.last_semantic_method_version: str | None = None
        self.last_shadow_semantic: dict[str, Any] | None = None
        self.last_shadow_semantic_method: str | None = None
        self.last_shadow_semantic_method_version: str | None = None
        self.endpoint: str | None = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key: str | None = os.environ.get("AZURE_OPENAI_API_KEY")
        self.deployment: str | None = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_version: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._client: Any = None  # Lazy-loaded AzureOpenAI fallback client.

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client for the optional fallback path."""

        if not self.endpoint or not self.api_key or not self.deployment:
            raise RuntimeError("Azure OpenAI gray-band fallback is not configured")

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _fallback_method_version(self) -> str:
        deployment = self.deployment or "unconfigured"
        return f"azure-openai:{deployment}:{self.api_version}"

    def _score_primary(self, expected_greeting: str, actual_utterance: str) -> float:
        scorer = self._primary_scorer
        if isinstance(scorer, LocalCrossEncoderScorer):
            return scorer.score(expected_greeting, actual_utterance)
        return _extract_scalar(scorer(expected_greeting, actual_utterance))

    def _validated_result(
        self,
        *,
        reasoning: str,
        is_match: bool,
        confidence_score: Any,
    ) -> dict[str, Any]:
        """Validate and return only the canonical scorer payload."""

        return SemanticEvaluationResult.model_validate(
            {
                "reasoning": reasoning,
                "is_match": bool(is_match),
                "confidence_score": _clamp_probability(confidence_score),
            }
        ).model_dump()

    def _primary_result(self, score: float) -> dict[str, Any]:
        """Route a non-gray primary score to its deterministic direct result."""

        if score >= MATCH_THRESHOLD:
            return self._validated_result(
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=score,
            )

        return self._validated_result(
            reasoning="cross_encoder_high_nonmatch",
            is_match=False,
            confidence_score=score,
        )

    def _gray_band_without_fallback(self, score: float) -> dict[str, Any]:
        """Emit deterministic false/zero output when gray-band fallback is disabled."""

        self._record_attribution(
            semantic_method="cross_encoder",
            semantic_method_version=CROSS_ENCODER_METHOD_VERSION,
        )
        return self._validated_result(
            reasoning="semantic_error",
            is_match=False,
            confidence_score=0.0,
        )

    def _fallback_failure_result(self, reason_code: str) -> dict[str, Any]:
        """Emit deterministic false/zero outputs for fallback timeout/error paths."""

        self._record_attribution(
            semantic_method="llm_gray_band",
            semantic_method_version=self._fallback_method_version(),
        )
        return self._validated_result(
            reasoning=reason_code,
            is_match=False,
            confidence_score=0.0,
        )

    def _evaluate_gray_band_fallback(
        self,
        expected_greeting: str,
        actual_utterance: str,
        _primary_score: float,
    ) -> dict[str, Any]:
        """Invoke Azure OpenAI for an enabled primary Cross-Encoder gray-band score."""

        user_content: str = (
            f"Expected Greeting Rule: {expected_greeting}\nActual Utterance: {actual_utterance}"
        )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if self._client is None:
                    self._init_client()

                response: Any = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=LLM_PARAMS["temperature"],
                    top_p=LLM_PARAMS["top_p"],
                    max_tokens=LLM_PARAMS["max_tokens"],
                    seed=LLM_PARAMS["seed"],
                    response_format=GRAY_BAND_RESPONSE_FORMAT,
                )

                raw: str = response.choices[0].message.content
                parsed: dict[str, Any] = json.loads(raw)
                if set(parsed) != {"reasoning", "is_match", "confidence_score"}:
                    raise ValueError("gray-band fallback payload must match the §8.4 schema")

                is_match = _coerce_bool(parsed["is_match"])
                expected_reason = "gray_band_llm_match" if is_match else "gray_band_llm_nonmatch"
                provider_reason = parsed["reasoning"]
                if provider_reason not in GRAY_BAND_PROVIDER_REASON_CODES:
                    raise ValueError("gray-band fallback reasoning must be a provider reason code")
                if provider_reason != expected_reason:
                    raise ValueError("gray-band fallback reasoning must agree with is_match")

                self._record_attribution(
                    semantic_method="llm_gray_band",
                    semantic_method_version=self._fallback_method_version(),
                )
                return self._validated_result(
                    reasoning=provider_reason,
                    is_match=is_match,
                    confidence_score=parsed["confidence_score"],
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Semantic gray-band fallback attempt %d failed",
                    attempt + 1,
                    exc_info=True,
                )

        if last_error is not None and _is_timeout_error(last_error):
            return self._fallback_failure_result("semantic_timeout")
        return self._fallback_failure_result("semantic_error")

    def evaluate(self, expected_greeting: str, actual_utterance: str) -> dict[str, Any] | None:
        """
        Evaluate semantic equivalence between expected and actual utterance.

        Returns only the canonical §8 scorer payload: ``reasoning``,
        ``is_match``, and ``confidence_score``. Route-specific method/version
        attribution is retained separately on ``last_semantic_method`` and
        ``last_semantic_method_version`` for the D→E transport boundary.
        The optional shadow scorer records candidate outputs on shadow-only
        sidecar attributes and never changes the returned live payload.
        The only normal path returning ``None`` is an unexpected validation failure
        after all deterministic degradation paths have been exhausted.
        """

        self.last_semantic_method = None
        self.last_semantic_method_version = None
        self.last_shadow_semantic = None
        self.last_shadow_semantic_method = None
        self.last_shadow_semantic_method_version = None

        try:
            primary_score = calibrate_cross_encoder_score(
                self._score_primary(expected_greeting, actual_utterance)
            )
        except Exception:
            logger.warning("Local semantic Cross-Encoder scoring failed", exc_info=True)
            self._record_attribution(
                semantic_method="cross_encoder",
                semantic_method_version=CROSS_ENCODER_METHOD_VERSION,
            )
            result = self._validated_result(
                reasoning="semantic_local_failure_fallback",
                is_match=False,
                confidence_score=0.0,
            )
            self._evaluate_shadow(expected_greeting, actual_utterance)
            return result

        # §8.2.1 — score = 0.72 is direct match and is NOT gray-band routed.
        if primary_score >= MATCH_THRESHOLD or primary_score < GRAY_BAND_LOWER_THRESHOLD:
            self._record_attribution(
                semantic_method="cross_encoder",
                semantic_method_version=CROSS_ENCODER_METHOD_VERSION,
            )
            result = self._primary_result(primary_score)
            self._evaluate_shadow(expected_greeting, actual_utterance)
            return result

        # §8.1 — Azure is reachable only for explicitly enabled true gray band.
        if self.gray_band_fallback_enabled:
            result = self._evaluate_gray_band_fallback(
                expected_greeting,
                actual_utterance,
                primary_score,
            )
        else:
            result = self._gray_band_without_fallback(primary_score)
        self._evaluate_shadow(expected_greeting, actual_utterance)
        return result

    def _record_attribution(
        self,
        *,
        semantic_method: str,
        semantic_method_version: str,
    ) -> None:
        """Record method/version sidecar without altering the canonical scorer JSON."""

        self.last_semantic_method = semantic_method
        self.last_semantic_method_version = semantic_method_version

    def _evaluate_shadow(self, expected_greeting: str, actual_utterance: str) -> None:
        """Run an optional candidate scorer without changing live semantic outputs."""
        if not self.shadow_mode_enabled or self._shadow_scorer is None:
            return

        try:
            scorer = self._shadow_scorer
            if isinstance(scorer, LocalCrossEncoderScorer):
                raw_score = scorer.score(expected_greeting, actual_utterance)
            else:
                raw_score = scorer(expected_greeting, actual_utterance)
            score = calibrate_cross_encoder_score(raw_score)
            is_match = score >= MATCH_THRESHOLD
            self.last_shadow_semantic = {
                "reasoning": "shadow_candidate_match" if is_match else "shadow_candidate_nonmatch",
                "is_match": is_match,
                "confidence_score": score,
            }
            self.last_shadow_semantic_method = "semantic_shadow_candidate"
            self.last_shadow_semantic_method_version = "candidate-unpromoted"
        except Exception:
            logger.warning("Semantic shadow scoring failed", exc_info=True)
            self.last_shadow_semantic = {
                "reasoning": "shadow_candidate_error",
                "is_match": False,
                "confidence_score": 0.0,
            }
            self.last_shadow_semantic_method = "semantic_shadow_candidate"
            self.last_shadow_semantic_method_version = "candidate-unpromoted"
