"""
Serialization Helpers — Celery JSON Transport

Celery is configured with task_serializer="json" (§9.1, celery_app.py).
JSON cannot natively serialize Python bytes objects. The orchestrator's
assemble_segment() attaches raw audio (PCM bytes) and video frame data
(numpy tobytes()) to the InferenceHandoffPayload dict.

These helpers provide base64 encoding/decoding for bytes fields so the
payload survives Celery's JSON serialization round-trip without changing
the serializer to pickle (which introduces security concerns).

Usage:
    # In orchestrator assemble_segment():
    from services.worker.pipeline.serialization import encode_bytes_fields
    result = encode_bytes_fields(result, ["_audio_data", "_frame_data"])

    # In process_segment():
    from services.worker.pipeline.serialization import decode_bytes_fields
    payload = decode_bytes_fields(payload, ["_audio_data", "_frame_data"])
"""

from __future__ import annotations

import base64
import math
from typing import Any


def encode_bytes_fields(
    payload: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    """
    Base64-encode specified bytes fields for JSON serialization.

    Fields that are None are left as None. Fields that are already
    strings are left unchanged (idempotent if called twice).

    Args:
        payload: The dict to modify (mutated in place and returned).
        fields: List of keys whose values should be base64-encoded.

    Returns:
        The same dict with bytes fields replaced by base64 ASCII strings.
    """
    for key in fields:
        value = payload.get(key)
        if isinstance(value, (bytes, bytearray)):
            payload[key] = base64.b64encode(value).decode("ascii")
    return payload


def decode_bytes_fields(
    payload: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    """
    Base64-decode specified fields back to bytes.

    Fields that are None are left as None. Fields that are already
    bytes are left unchanged (idempotent if called twice).

    Args:
        payload: The dict to modify (mutated in place and returned).
        fields: List of keys whose values should be base64-decoded.

    Returns:
        The same dict with base64 string fields replaced by bytes.
    """
    for key in fields:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = base64.b64decode(value)
    return payload


def sanitize_json_payload(payload: Any) -> Any:
    """Recursively coerce JSON-unsafe non-finite floats to ``None``.

    This keeps Python ``None`` intact so downstream JSON serializers emit
    canonical ``null`` values for required nullable fields, and prevents
    ``NaN`` / ``Infinity`` tokens from leaking into Celery JSON payloads or
    operator-facing responses.

    Handoff-specific optional objects are normalized while traversing dicts:
    an ineligible ``_physiological_context`` is removed instead of serialized
    as ``{}`` or as a null-only role wrapper, and absent bandit snapshot
    optionals are omitted rather than emitted as explicit ``null`` members.

    Dicts and lists are mutated in place and returned for convenience.
    Tuples are rebuilt because they are immutable.
    """
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, dict):
        if "_physiological_context" in payload:
            context = payload.get("_physiological_context")
            if not isinstance(context, dict) or not context:
                payload.pop("_physiological_context", None)
            else:
                streamer = context.get("streamer")
                operator = context.get("operator")
                if isinstance(streamer, dict) and not streamer:
                    streamer = None
                if isinstance(operator, dict) and not operator:
                    operator = None
                if streamer is None and operator is None:
                    payload.pop("_physiological_context", None)
                else:
                    context["streamer"] = streamer
                    context["operator"] = operator

        snapshot = payload.get("_bandit_decision_snapshot")
        if isinstance(snapshot, dict):
            for optional_key in (
                "sampled_theta_by_arm",
                "decision_context_hash",
                "random_seed",
            ):
                if snapshot.get(optional_key) is None:
                    snapshot.pop(optional_key, None)

        for key, value in list(payload.items()):
            payload[key] = sanitize_json_payload(value)
        return payload
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            payload[idx] = sanitize_json_payload(value)
        return payload
    if isinstance(payload, tuple):
        return tuple(sanitize_json_payload(value) for value in payload)
    return payload
