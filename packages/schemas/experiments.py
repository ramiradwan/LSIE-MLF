"""Experiment admin DTOs for `/api/v1/experiments/*` write surfaces.

These models describe the additive experiment/arm management contract.
Posterior-owned numeric state remains read-only: callers may create arms
(which always start at Beta(1,1)) and may edit only human-owned arm
metadata (`greeting_text`) plus one-way disable semantics
(`enabled=false`).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentAdminModel(BaseModel):
    """Shared base config for experiment admin request/response models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExperimentArmSeedRequest(ExperimentAdminModel):
    """One arm to create under a new or existing experiment."""

    arm: str = Field(..., min_length=1)
    greeting_text: str = Field(..., min_length=1)


class ExperimentCreateRequest(ExperimentAdminModel):
    """Create a new experiment with its initial arm set."""

    experiment_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    arms: list[ExperimentArmSeedRequest] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_arm_ids(self) -> ExperimentCreateRequest:
        arm_ids = [arm.arm for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm identifiers must be unique within the experiment request")
        return self


class ExperimentArmCreateRequest(ExperimentArmSeedRequest):
    """Add one new arm to an existing experiment."""


class ExperimentArmPatchRequest(ExperimentAdminModel):
    """Mutable subset of arm fields.

    Only `greeting_text` and one-way disable (`enabled=false`) are
    accepted. Posterior-owned fields are intentionally absent and, via
    `extra='forbid'`, rejected at validation time.
    """

    greeting_text: str | None = Field(default=None, min_length=1)
    enabled: bool | None = None

    @model_validator(mode="after")
    def _supported_mutations_only(self) -> ExperimentArmPatchRequest:
        if self.greeting_text is None and self.enabled is None:
            raise ValueError("at least one supported arm field must be provided")
        if self.enabled is True:
            raise ValueError("enabled=true is not supported; use enabled=false to disable an arm")
        return self


class ExperimentArmAdminResponse(ExperimentAdminModel):
    """Write-surface readback for one experiment arm."""

    experiment_id: str
    label: str
    arm: str
    greeting_text: str
    alpha_param: float = Field(..., gt=0.0)
    beta_param: float = Field(..., gt=0.0)
    enabled: bool = True
    end_dated_at: datetime | None = None
    updated_at: datetime | None = None


class ExperimentArmDeleteResponse(ExperimentAdminModel):
    """DELETE readback for one arm.

    `deleted=True` means an unused Beta(1,1) arm was hard-deleted.
    `deleted=False` with `posterior_preserved=True` means the service
    guard found posterior or selection history and disabled/end-dated the
    arm instead of removing its row. Any included `arm_state` is read-only
    readback; no posterior-owned fields are writable through this model.
    """

    experiment_id: str
    arm: str
    deleted: bool
    posterior_preserved: bool
    reason: str | None = None
    arm_state: ExperimentArmAdminResponse | None = None


class ExperimentAdminResponse(ExperimentAdminModel):
    """Write-surface readback for a full experiment."""

    experiment_id: str
    label: str
    arms: list[ExperimentArmAdminResponse] = Field(default_factory=list)
