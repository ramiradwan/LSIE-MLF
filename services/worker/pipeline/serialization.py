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
