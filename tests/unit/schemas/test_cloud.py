from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from packages.schemas.cloud import (
    CloudSessionCreateRequest,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
    OAuthTokenRequest,
    PosteriorDelta,
    TelemetryPosteriorDeltaBatch,
)
from packages.schemas.evaluation import StimulusDefinition, StimulusPayload

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64


def _posterior_delta_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "experiment_id": 101,
        "arm_id": "arm_a",
        "delta_alpha": 1.0,
        "delta_beta": 0.0,
        "segment_id": SEGMENT_ID,
        "client_id": "desktop-a",
        "event_id": uuid.uuid4(),
        "applied_at_utc": datetime.now(UTC),
        "decision_context_hash": DECISION_CONTEXT_HASH,
    }
    data.update(overrides)
    return data


def _stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(content_type="text", text="hello"),
        expected_stimulus_rule="Deliver the spoken greeting",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


class TestCloudSchemas:
    def test_cloud_models_reject_extra_fields(self, sample_timestamp: datetime) -> None:
        with pytest.raises(ValidationError):
            CloudSessionCreateRequest.model_validate(
                {
                    "client_id": "desktop-a",
                    "started_at_utc": sample_timestamp,
                    "policy_version": "cloud-v1",
                    "unexpected": "rejected",
                }
            )

    def test_posterior_delta_accepts_closed_unit_interval_bounds(self) -> None:
        lower = PosteriorDelta.model_validate(
            _posterior_delta_data(delta_alpha=0.0, delta_beta=0.0)
        )
        upper = PosteriorDelta.model_validate(
            _posterior_delta_data(delta_alpha=1.0, delta_beta=1.0)
        )

        assert lower.delta_alpha == 0.0
        assert upper.delta_beta == 1.0

    def test_posterior_delta_accepts_uuid5_event_id(self) -> None:
        posterior_delta = PosteriorDelta.model_validate(
            _posterior_delta_data(event_id=uuid.uuid5(uuid.NAMESPACE_URL, "posterior-delta"))
        )

        assert posterior_delta.event_id.version == 5

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("delta_alpha", -0.01),
            ("delta_alpha", 1.01),
            ("delta_beta", -0.01),
            ("delta_beta", 1.01),
        ],
    )
    def test_posterior_delta_rejects_values_outside_unit_interval(
        self,
        field: str,
        value: float,
    ) -> None:
        with pytest.raises(ValidationError):
            PosteriorDelta.model_validate(_posterior_delta_data(**{field: value}))

    def test_posterior_delta_batch_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryPosteriorDeltaBatch.model_validate(
                {"deltas": [_posterior_delta_data()], "unexpected": "rejected"}
            )

    def test_oauth_token_request_requires_grant_specific_fields(self) -> None:
        with pytest.raises(ValidationError):
            OAuthTokenRequest.model_validate(
                {"grant_type": "authorization_code", "client_id": "desktop-a", "code": "abc"}
            )

        refresh = OAuthTokenRequest.model_validate(
            {
                "grant_type": "refresh_token",
                "client_id": "desktop-a",
                "refresh_token": "refresh-a",
            }
        )

        assert refresh.refresh_token == "refresh-a"

    def test_experiment_bundle_rejects_duplicate_experiment_and_arm_ids(
        self,
        sample_timestamp: datetime,
    ) -> None:
        arm = ExperimentBundleArm(
            arm_id="arm-a",
            stimulus_definition=_stimulus_definition(),
            posterior_alpha=1.0,
            posterior_beta=1.0,
        )
        with pytest.raises(ValidationError):
            ExperimentBundleExperiment(
                experiment_id="experiment-a",
                label="Experiment A",
                arms=[arm, arm],
            )

        experiment = ExperimentBundleExperiment(
            experiment_id="experiment-a",
            label="Experiment A",
            arms=[arm],
        )
        with pytest.raises(ValidationError):
            ExperimentBundlePayload(
                bundle_id="bundle-a",
                issued_at_utc=sample_timestamp,
                expires_at_utc=sample_timestamp,
                policy_version="v4.0",
                experiments=[experiment],
            )
        with pytest.raises(ValidationError):
            ExperimentBundlePayload(
                bundle_id="bundle-a",
                issued_at_utc=sample_timestamp,
                expires_at_utc=sample_timestamp.replace(year=sample_timestamp.year + 1),
                policy_version="v4.0",
                experiments=[experiment, experiment],
            )
