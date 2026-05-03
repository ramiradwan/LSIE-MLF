"""Pydantic envelope for the cross-process inference dispatch.

Implements the desktop Queue control envelope described by §9.2 using a
 typed control message that travels over ``IpcChannels.ml_inbox``.
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
``forbid_raw`` middleware.
"""

from __future__ import annotations

from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from packages.schemas.evaluation import SemanticMethod, SemanticReasonCode
from packages.schemas.inference_handoff import InferenceHandoffPayload
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


class AnalyticsRewardInputs(BaseModel):
    gated_reward: float = Field(..., ge=0.0, le=1.0)
    p90_intensity: float = Field(..., ge=0.0, le=1.0)
    semantic_gate: Literal[0, 1]
    n_frames_in_window: int = Field(..., ge=0)
    au12_baseline_pre: float | None = Field(default=None, ge=0.0, le=1.0)
    stimulus_time: float | None = None

    model_config = ConfigDict(extra="forbid")


class AnalyticsSemanticResult(BaseModel):
    reasoning: SemanticReasonCode
    is_match: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    semantic_method: SemanticMethod
    semantic_method_version: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class AnalyticsAcousticMetrics(BaseModel):
    f0_valid_measure: bool
    f0_valid_baseline: bool
    perturbation_valid_measure: bool
    perturbation_valid_baseline: bool
    voiced_coverage_measure_s: float = Field(..., ge=0.0)
    voiced_coverage_baseline_s: float = Field(..., ge=0.0)
    f0_mean_measure_hz: float | None = Field(default=None, ge=0.0)
    f0_mean_baseline_hz: float | None = Field(default=None, ge=0.0)
    f0_delta_semitones: float | None = None
    jitter_mean_measure: float | None = Field(default=None, ge=0.0)
    jitter_mean_baseline: float | None = Field(default=None, ge=0.0)
    jitter_delta: float | None = None
    shimmer_mean_measure: float | None = Field(default=None, ge=0.0)
    shimmer_mean_baseline: float | None = Field(default=None, ge=0.0)
    shimmer_delta: float | None = None

    model_config = ConfigDict(extra="forbid")


class AnalyticsAttributionInputs(BaseModel):
    outcome_event: dict[str, Any] | None = Field(default=None, alias="_outcome_event")
    outcome_events: list[dict[str, Any]] | None = Field(default=None, alias="_outcome_events")
    creator_follow: bool | None = Field(default=None, alias="_creator_follow")

    model_config = ConfigDict(
        extra="forbid",
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class AnalyticsResultMessage(BaseModel):
    message_id: UUID
    schema_version: Literal["ws5.p4.analytics_result.v1"] = "ws5.p4.analytics_result.v1"
    handoff: InferenceHandoffPayload
    semantic: AnalyticsSemanticResult
    transcription: str = ""
    reward: AnalyticsRewardInputs | None = None
    acoustic: AnalyticsAcousticMetrics | None = None
    attribution: AnalyticsAttributionInputs | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _window_bounds_are_ordered(self) -> AnalyticsResultMessage:
        if self.handoff.segment_window_end_utc < self.handoff.segment_window_start_utc:
            raise ValueError("segment_window_end_utc must be after segment_window_start_utc")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_absent_optionals(self, handler: Any) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        handoff = data.get("handoff")
        if isinstance(handoff, dict) and handoff.get("_physiological_context") is None:
            handoff.pop("_physiological_context", None)
        for key in ("reward", "acoustic", "attribution"):
            if data.get(key) is None:
                data.pop(key, None)
        return data
