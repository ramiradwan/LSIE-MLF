from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.responses import JSONResponse

from packages.schemas.data_tiers import DataTier, mark_data_tier

logger = logging.getLogger(__name__)

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH"})
FORBIDDEN_DETAIL = "raw media payloads are not accepted by cloud API"

_EXACT_KEY_RULES: dict[str, str] = {
    "_audio_data": "raw_audio_field",
    "_frame_data": "raw_frame_field",
    "audio_b64": "raw_audio_field",
    "frame_b64": "raw_frame_field",
    "video_b64": "raw_frame_field",
    "image_b64": "raw_frame_field",
    "audio_base64": "raw_audio_field",
    "audioBase64": "raw_audio_field",
    "raw_audio": "raw_audio_field",
    "raw_frame": "raw_frame_field",
    "decoded_frame": "decoded_frame_field",
    "decoded_frames": "decoded_frame_field",
    "voiceprint": "biometric_embedding_field",
    "voice_embedding": "biometric_embedding_field",
    "speaker_embedding": "biometric_embedding_field",
    "raw_provider_response": "raw_provider_body_field",
    "provider_raw_blob": "raw_provider_body_field",
    "raw_webhook_body": "raw_provider_body_field",
    "raw_notification": "raw_provider_body_field",
    "hydration_body": "raw_provider_body_field",
    "physiologicalchunkevent": "physiological_chunk_payload",
    "rationale": "free_form_semantic_rationale",
    "semantic_rationale": "free_form_semantic_rationale",
    "reason_text": "free_form_semantic_rationale",
    "free_form_reasoning": "free_form_semantic_rationale",
    "llm_reasoning": "free_form_semantic_rationale",
    "explanation": "free_form_semantic_rationale",
}
_PCM_KEY_RE = re.compile(r"^pcm(?:_|$)", re.IGNORECASE)
_MEDIA_BASE64_KEY_RE = re.compile(r"^(audio|frame|video|image)_(b64|base64)$")
_BIOMETRIC_KEY_RE = re.compile(r"^(voice|speaker)_(print|embedding)$")
_JSON_CONTENT_RE = re.compile(r"(^|[/+])json($|;)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RawPayloadViolation:
    rule_id: str


def contains_forbidden_raw_payload(value: object) -> RawPayloadViolation | None:
    return _inspect_json(value)


async def forbid_raw_payload_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.method not in WRITE_METHODS or not _is_json_content(request):
        return await call_next(request)

    body = mark_data_tier(
        await request.body(),
        DataTier.TRANSIENT,
        spec_ref="§5.2.1",
    )
    if not body:
        return await call_next(request)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return await call_next(request)

    violation = contains_forbidden_raw_payload(payload)
    if violation is None:
        return await call_next(request)

    logger.warning(
        "cloud privacy perimeter rejected inbound JSON",
        extra={
            "rule_id": violation.rule_id,
            "method": request.method,
            "path": request.url.path,
            "content_length": request.headers.get("content-length", "0"),
        },
    )
    return JSONResponse(status_code=422, content={"detail": FORBIDDEN_DETAIL})


def _is_json_content(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return bool(_JSON_CONTENT_RE.search(content_type))


def _inspect_json(value: object) -> RawPayloadViolation | None:
    if isinstance(value, dict):
        event_type = value.get("event_type")
        if isinstance(event_type, str) and event_type == "physiological_chunk":
            return RawPayloadViolation("physiological_chunk_payload")
        for key, child in value.items():
            violation = _inspect_key(key)
            if violation is not None:
                return violation
            violation = _inspect_json(child)
            if violation is not None:
                return violation
        return None
    if isinstance(value, list):
        for child in value:
            violation = _inspect_json(child)
            if violation is not None:
                return violation
    return None


def _inspect_key(key: object) -> RawPayloadViolation | None:
    if not isinstance(key, str):
        return None
    normalized = _normalize_key(key)
    if _PCM_KEY_RE.match(normalized):
        return RawPayloadViolation("raw_pcm_field")
    if _MEDIA_BASE64_KEY_RE.match(normalized):
        return RawPayloadViolation("raw_media_base64_field")
    if _BIOMETRIC_KEY_RE.match(normalized):
        return RawPayloadViolation("biometric_embedding_field")
    rule_id = _EXACT_KEY_RULES.get(key) or _EXACT_KEY_RULES.get(normalized)
    if rule_id is None:
        return None
    return RawPayloadViolation(rule_id)


def _normalize_key(key: str) -> str:
    chars: list[str] = []
    previous_was_lower_or_digit = False
    for char in key:
        if char.isupper() and previous_was_lower_or_digit:
            chars.append("_")
        if char in {"-", " ", "."}:
            chars.append("_")
            previous_was_lower_or_digit = False
            continue
        chars.append(char.lower())
        previous_was_lower_or_digit = char.islower() or char.isdigit()
    return "".join(chars)
