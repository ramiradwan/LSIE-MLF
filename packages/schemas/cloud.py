from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from packages.schemas.attribution import AttributionEvent
from packages.schemas.inference_handoff import InferenceHandoffPayload

SHA256_HEX_PATTERN = "^[0-9a-f]{64}$"


class CloudSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PosteriorDelta(CloudSchemaModel):
    experiment_id: int
    arm_id: str = Field(..., min_length=1)
    delta_alpha: float = Field(..., ge=0.0, le=1.0)
    delta_beta: float = Field(..., ge=0.0, le=1.0)
    segment_id: str = Field(..., pattern=SHA256_HEX_PATTERN)
    client_id: str = Field(..., min_length=1)
    event_id: UUID
    applied_at_utc: datetime
    decision_context_hash: str = Field(..., pattern=SHA256_HEX_PATTERN)


class TelemetrySegmentBatch(CloudSchemaModel):
    segments: list[InferenceHandoffPayload] = Field(default_factory=list)
    attribution_events: list[AttributionEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _requires_segment_or_event(self) -> TelemetrySegmentBatch:
        if not self.segments and not self.attribution_events:
            raise ValueError(
                "telemetry segment batch must include at least one segment or attribution event"
            )
        return self


class TelemetryPosteriorDeltaBatch(CloudSchemaModel):
    deltas: list[PosteriorDelta] = Field(..., min_length=1)


class CloudIngestResponse(CloudSchemaModel):
    status: Literal["accepted"] = "accepted"
    accepted_count: int = Field(..., ge=0)
    inserted_count: int = Field(..., ge=0)


class ExperimentBundleArm(CloudSchemaModel):
    arm_id: str = Field(..., min_length=1)
    greeting_text: str = Field(..., min_length=1)
    posterior_alpha: float = Field(..., gt=0.0)
    posterior_beta: float = Field(..., gt=0.0)
    selection_count: int = Field(default=0, ge=0)
    enabled: bool = True


class ExperimentBundleExperiment(CloudSchemaModel):
    experiment_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    arms: list[ExperimentBundleArm] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_arm_ids(self) -> ExperimentBundleExperiment:
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm identifiers must be unique within an experiment")
        return self


class ExperimentBundlePayload(CloudSchemaModel):
    bundle_id: str = Field(..., min_length=1)
    issued_at_utc: datetime
    expires_at_utc: datetime
    policy_version: str = Field(..., min_length=1)
    experiments: list[ExperimentBundleExperiment] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _valid_time_window(self) -> ExperimentBundlePayload:
        if self.expires_at_utc <= self.issued_at_utc:
            raise ValueError("expires_at_utc must be after issued_at_utc")
        return self

    @model_validator(mode="after")
    def _unique_experiment_ids(self) -> ExperimentBundlePayload:
        experiment_ids = [experiment.experiment_id for experiment in self.experiments]
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("experiment identifiers must be unique within a bundle")
        return self


class ExperimentBundle(ExperimentBundlePayload):
    signature: str = Field(..., min_length=1)


class CloudSessionCreateRequest(CloudSchemaModel):
    client_id: str = Field(..., min_length=1)
    started_at_utc: datetime
    policy_version: str = Field(..., min_length=1)


class CloudSessionCreateResponse(CloudSchemaModel):
    session_id: UUID4
    client_id: str = Field(..., min_length=1)
    created_at_utc: datetime


class CloudSessionEndRequest(CloudSchemaModel):
    ended_at_utc: datetime
    reason: str | None = Field(default=None, min_length=1)


class CloudSessionEndResponse(CloudSchemaModel):
    session_id: UUID4
    ended_at_utc: datetime
    status: Literal["ended"] = "ended"


class OAuthTokenRequest(CloudSchemaModel):
    grant_type: Literal["authorization_code", "refresh_token"]
    client_id: str = Field(..., min_length=1)
    code: str | None = Field(default=None, min_length=1)
    code_verifier: str | None = Field(default=None, min_length=1)
    redirect_uri: str | None = Field(default=None, min_length=1)
    refresh_token: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _required_grant_fields(self) -> OAuthTokenRequest:
        if self.grant_type == "authorization_code" and (
            self.code is None or self.code_verifier is None or self.redirect_uri is None
        ):
            raise ValueError(
                "authorization_code grant requires code, code_verifier, and redirect_uri"
            )
        if self.grant_type == "refresh_token" and self.refresh_token is None:
            raise ValueError("refresh_token grant requires refresh_token")
        return self


class OAuthTokenResponse(CloudSchemaModel):
    access_token: str = Field(..., min_length=1)
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: int = Field(..., gt=0)
    refresh_token: str | None = Field(default=None, min_length=1)
    scope: str | None = Field(default=None, min_length=1)
