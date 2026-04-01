"""
Tests for services/worker/pipeline/serialization.py — Phase 3.1 gap-fix coverage.

Verifies base64 encode/decode helpers that enable bytes fields to survive
Celery's JSON serialization round-trip (§9.1 celery_app task_serializer="json").
"""

from __future__ import annotations

import base64
from typing import Any

from services.worker.pipeline.serialization import decode_bytes_fields, encode_bytes_fields


class TestEncodeByteFields:
    """encode_bytes_fields: bytes/bytearray → base64 string for JSON transport."""

    def test_bytes_to_base64_string(self) -> None:
        """bytes values are converted to base64 ASCII strings."""
        payload: dict[str, Any] = {"_audio_data": b"\x00\x01\x02\x03"}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert isinstance(result["_audio_data"], str)
        assert base64.b64decode(result["_audio_data"]) == b"\x00\x01\x02\x03"

    def test_bytearray_to_base64_string(self) -> None:
        """bytearray values are also converted."""
        payload: dict[str, Any] = {"_audio_data": bytearray(b"\xff\xfe")}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert isinstance(result["_audio_data"], str)
        assert base64.b64decode(result["_audio_data"]) == b"\xff\xfe"

    def test_none_value_unchanged(self) -> None:
        """None values are left as None (not encoded)."""
        payload: dict[str, Any] = {"_audio_data": None}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert result["_audio_data"] is None

    def test_already_string_unchanged(self) -> None:
        """String values are left unchanged (idempotent)."""
        b64 = base64.b64encode(b"hello").decode("ascii")
        payload: dict[str, Any] = {"_audio_data": b64}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert result["_audio_data"] == b64

    def test_non_target_fields_untouched(self) -> None:
        """Fields not in the fields list are not modified."""
        payload: dict[str, Any] = {
            "_audio_data": b"\x00\x01",
            "session_id": "abc-123",
            "count": 42,
        }
        encode_bytes_fields(payload, ["_audio_data"])
        assert payload["session_id"] == "abc-123"
        assert payload["count"] == 42

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict without error."""
        result = encode_bytes_fields({}, ["_audio_data"])
        assert result == {}

    def test_missing_field_key_no_error(self) -> None:
        """Requesting a field not present in the dict does not raise."""
        payload: dict[str, Any] = {"other": "value"}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert "other" in result
        assert "_audio_data" not in result

    def test_mutates_in_place(self) -> None:
        """The function mutates the original dict and returns it."""
        payload: dict[str, Any] = {"_audio_data": b"\x00"}
        result = encode_bytes_fields(payload, ["_audio_data"])
        assert result is payload


class TestDecodeByteFields:
    """decode_bytes_fields: base64 string → bytes for ML pipeline consumption."""

    def test_base64_string_to_bytes(self) -> None:
        """base64 string values are decoded back to bytes."""
        original = b"\x00\x01\x02\x03"
        b64 = base64.b64encode(original).decode("ascii")
        payload: dict[str, Any] = {"_audio_data": b64}
        result = decode_bytes_fields(payload, ["_audio_data"])
        assert isinstance(result["_audio_data"], bytes)
        assert result["_audio_data"] == original

    def test_none_value_unchanged(self) -> None:
        """None values are left as None (not decoded)."""
        payload: dict[str, Any] = {"_audio_data": None}
        result = decode_bytes_fields(payload, ["_audio_data"])
        assert result["_audio_data"] is None

    def test_already_bytes_unchanged(self) -> None:
        """bytes values are left unchanged (idempotent)."""
        payload: dict[str, Any] = {"_audio_data": b"\x00\x01"}
        result = decode_bytes_fields(payload, ["_audio_data"])
        assert result["_audio_data"] == b"\x00\x01"

    def test_non_target_fields_untouched(self) -> None:
        """Fields not in the fields list are not modified."""
        b64 = base64.b64encode(b"data").decode("ascii")
        payload: dict[str, Any] = {
            "_audio_data": b64,
            "session_id": "abc-123",
        }
        decode_bytes_fields(payload, ["_audio_data"])
        assert payload["session_id"] == "abc-123"

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict without error."""
        result = decode_bytes_fields({}, ["_audio_data"])
        assert result == {}

    def test_mutates_in_place(self) -> None:
        """The function mutates the original dict and returns it."""
        b64 = base64.b64encode(b"\x00").decode("ascii")
        payload: dict[str, Any] = {"_audio_data": b64}
        result = decode_bytes_fields(payload, ["_audio_data"])
        assert result is payload


class TestRoundTrip:
    """Encode then decode produces original bytes."""

    def test_round_trip_single_field(self) -> None:
        """Single field survives encode → decode round-trip."""
        original = b"\x00\x01\x02\xff" * 100
        payload: dict[str, Any] = {"_audio_data": original, "session_id": "s1"}
        fields = ["_audio_data"]
        encode_bytes_fields(payload, fields)
        decode_bytes_fields(payload, fields)
        assert payload["_audio_data"] == original
        assert payload["session_id"] == "s1"

    def test_round_trip_multiple_fields(self) -> None:
        """Multiple binary fields survive encode → decode."""
        audio = b"\x00\x01" * 50
        frame = b"\xff\xfe" * 100
        payload: dict[str, Any] = {
            "_audio_data": audio,
            "_frame_data": frame,
            "session_id": "s2",
        }
        fields = ["_audio_data", "_frame_data"]

        encode_bytes_fields(payload, fields)
        assert isinstance(payload["_audio_data"], str)
        assert isinstance(payload["_frame_data"], str)

        decode_bytes_fields(payload, fields)

        # Mypy narrowed the type to str above; ignore the overlap check after mutation
        assert payload["_audio_data"] == audio  # type: ignore[comparison-overlap]
        assert payload["_frame_data"] == frame  # type: ignore[comparison-overlap]

    def test_round_trip_with_none(self) -> None:
        """None fields pass through encode → decode unchanged."""
        payload: dict[str, Any] = {"_audio_data": None, "_frame_data": b"\x01"}
        fields = ["_audio_data", "_frame_data"]
        encode_bytes_fields(payload, fields)
        decode_bytes_fields(payload, fields)
        assert payload["_audio_data"] is None
        assert payload["_frame_data"] == b"\x01"
