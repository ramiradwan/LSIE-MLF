"""Pydantic envelope for the cross-process inference dispatch (WS3 P2).

Replaces the v3.4 Celery task argument (a single base64-bloated dict)
with a typed control message that travels over ``IpcChannels.ml_inbox``.
The 30 s PCM window does NOT travel here — its SharedMemory metadata
does, and the consumer attaches via ``shared_buffers.read_pcm_block``.

Field rationale:
    handoff           ``InferenceHandoffPayload.model_dump(by_alias=True)``
                      after sanitisation. Carries every ``_``-aliased
                      field that was previously inside the Celery task
                      payload, minus ``_audio_data``.
    audio             SharedMemory block locator + integrity metadata.
    forward_fields    The non-schema transport tail (``_experiment_code``,
                      and the ``_FORWARD_FIELDS`` /
                      ``_ATTRIBUTION_FORWARD_FIELDS`` keys that
                      ``services.worker.tasks.inference`` re-emits to
                      Module E without re-validating).

``model_config = ConfigDict(extra="forbid")`` matches the §6.1 stance
in the existing schemas (e.g.
``packages.schemas.attribution.AttributionBaseModel``) and rejects
unknown keys at IPC ingress, mirroring the cloud perimeter's
``forbid_raw`` middleware (WS5 P5).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from services.desktop_app.ipc.shared_buffers import PcmBlockMetadata


class AudioBlockRef(BaseModel):
    """SharedMemory locator + integrity metadata for one 30 s PCM block."""

    name: str = Field(..., min_length=1, max_length=200)
    byte_length: int = Field(..., gt=0)
    sha256: str = Field(..., pattern="^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_metadata(cls, metadata: PcmBlockMetadata) -> AudioBlockRef:
        return cls(
            name=metadata.name,
            byte_length=metadata.byte_length,
            sha256=metadata.sha256,
        )

    def to_metadata(self) -> PcmBlockMetadata:
        return PcmBlockMetadata(
            name=self.name,
            byte_length=self.byte_length,
            sha256=self.sha256,
        )


class InferenceControlMessage(BaseModel):
    """The IPC dispatch unit pushed onto ``IpcChannels.ml_inbox``."""

    handoff: dict[str, Any] = Field(...)
    audio: AudioBlockRef
    forward_fields: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
