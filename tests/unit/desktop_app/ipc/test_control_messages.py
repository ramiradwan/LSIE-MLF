"""WS3 P2 — InferenceControlMessage Pydantic validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.desktop_app.ipc.control_messages import (
    AudioBlockRef,
    InferenceControlMessage,
)
from services.desktop_app.ipc.shared_buffers import PcmBlockMetadata


def _valid_audio_ref() -> AudioBlockRef:
    return AudioBlockRef(
        name="lsie_ipc_pcm_abcdef0123456789",
        byte_length=960_000,
        sha256="0" * 64,
    )


def test_audio_ref_metadata_round_trip() -> None:
    metadata = PcmBlockMetadata(
        name="lsie_ipc_pcm_abc",
        byte_length=42,
        sha256="a" * 64,
    )
    ref = AudioBlockRef.from_metadata(metadata)
    assert ref.to_metadata() == metadata


def test_audio_ref_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(name="ok", byte_length=1, sha256="not-a-hash")


def test_audio_ref_rejects_zero_byte_length() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(name="ok", byte_length=0, sha256="0" * 64)


def test_audio_ref_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(
            name="ok",
            byte_length=1,
            sha256="0" * 64,
            unexpected="should fail",  # type: ignore[call-arg]
        )


def test_control_message_validates_minimal_handoff() -> None:
    msg = InferenceControlMessage(
        handoff={"segment_id": "abc"},
        audio=_valid_audio_ref(),
    )
    assert msg.forward_fields == {}
    assert msg.audio.byte_length == 960_000


def test_control_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        InferenceControlMessage(
            handoff={},
            audio=_valid_audio_ref(),
            stowaway="forbidden",  # type: ignore[call-arg]
        )


def test_control_message_round_trips_through_dump_validate() -> None:
    """Mirrors the IPC path: producer dumps, queue ships, consumer validates."""
    msg = InferenceControlMessage(
        handoff={"segment_id": "deadbeef"},
        audio=_valid_audio_ref(),
        forward_fields={"_experiment_code": "greeting_line_v1"},
    )
    dumped = msg.model_dump(mode="json")
    re_validated = InferenceControlMessage.model_validate(dumped)
    assert re_validated == msg
