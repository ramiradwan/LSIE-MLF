"""Unit tests for `services/api/services/experiment_admin_service.py`."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.experiments import (
    ExperimentArmCreateRequest,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from services.api.services.experiment_admin_service import (
    ExperimentAdminService,
    ExperimentAlreadyExistsError,
    ExperimentArmAlreadyExistsError,
    ExperimentArmNotFoundError,
    ExperimentNotFoundError,
)


def _service(
    now: datetime | None = None,
) -> tuple[ExperimentAdminService, MagicMock, MagicMock, MagicMock]:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    get_conn = MagicMock(return_value=conn)
    put_conn = MagicMock()
    service = ExperimentAdminService(
        get_conn=get_conn,
        put_conn=put_conn,
        clock=lambda: now or datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )
    return service, conn, cursor, put_conn


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(content_type="text", text=text),
        expected_stimulus_rule=text,
        expected_response_rule=text,
    )


class TestCreateExperiment:
    def test_initial_arms_seed_beta_one_one(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentCreateRequest.model_validate(
            {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
                "arms": [
                    {
                        "arm": "a",
                        "stimulus_definition": StimulusDefinition.model_validate(
                            _stimulus_definition("Hello")
                        ),
                    },
                    {
                        "arm": "b",
                        "stimulus_definition": StimulusDefinition.model_validate(
                            _stimulus_definition("Hei")
                        ),
                    },
                ],
            }
        )
        rows = [
            {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
                "arm": "a",
                "stimulus_definition": StimulusDefinition.model_validate(
                    _stimulus_definition("Hello")
                ),
                "alpha_param": 1.0,
                "beta_param": 1.0,
                "enabled": True,
                "end_dated_at": None,
                "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
            },
            {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
                "arm": "b",
                "stimulus_definition": StimulusDefinition.model_validate(
                    _stimulus_definition("Hei")
                ),
                "alpha_param": 1.0,
                "beta_param": 1.0,
                "enabled": True,
                "end_dated_at": None,
                "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
            },
        ]

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_identity.return_value = None
            q.fetch_experiment_admin_rows.return_value = rows

            result = service.create_experiment(request)

        assert q.insert_experiment_arm.call_args_list == [
            call(
                cursor,
                experiment_id="exp-1",
                label="Experiment 1",
                arm="a",
                stimulus_definition=request.arms[0].stimulus_definition,
                alpha_param=1.0,
                beta_param=1.0,
                enabled=True,
                end_dated_at=None,
            ),
            call(
                cursor,
                experiment_id="exp-1",
                label="Experiment 1",
                arm="b",
                stimulus_definition=request.arms[1].stimulus_definition,
                alpha_param=1.0,
                beta_param=1.0,
                enabled=True,
                end_dated_at=None,
            ),
        ]
        assert result.arms[0].alpha_param == 1.0
        assert result.arms[0].beta_param == 1.0
        assert result.arms[0].stimulus_definition.expected_response_rule == "Hello"
        assert result.arms[1].alpha_param == 1.0
        assert result.arms[1].beta_param == 1.0
        assert result.arms[1].stimulus_definition.expected_response_rule == "Hei"
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_existing_experiment_raises_conflict(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentCreateRequest.model_validate(
            {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
                "arms": [
                    {
                        "arm": "a",
                        "stimulus_definition": StimulusDefinition.model_validate(
                            _stimulus_definition("Hello")
                        ),
                    }
                ],
            }
        )

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_identity.return_value = {"experiment_id": "exp-1", "label": "exp-1"}

            with pytest.raises(ExperimentAlreadyExistsError):
                service.create_experiment(request)

        q.insert_experiment_arm.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once_with()
        put_conn.assert_called_once_with(conn)


class TestAddArm:
    def test_adds_new_arm_without_touching_existing_rows(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentArmCreateRequest(
            arm="c",
            stimulus_definition=_stimulus_definition("Moi"),
        )
        inserted_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "c",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Moi")),
            "alpha_param": 1.0,
            "beta_param": 1.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 5, tzinfo=UTC),
        }

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_identity.return_value = {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
            }
            q.fetch_experiment_arm_row.side_effect = [None, inserted_row]

            result = service.add_arm("exp-1", request)

        q.insert_experiment_arm.assert_called_once_with(
            cursor,
            experiment_id="exp-1",
            label="Experiment 1",
            arm="c",
            stimulus_definition=request.stimulus_definition,
            alpha_param=1.0,
            beta_param=1.0,
            enabled=True,
            end_dated_at=None,
        )
        assert result.arm == "c"
        assert result.alpha_param == 1.0
        assert result.beta_param == 1.0
        assert result.stimulus_definition.expected_response_rule == "Moi"
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_duplicate_arm_raises_conflict(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentArmCreateRequest(
            arm="c",
            stimulus_definition=_stimulus_definition("Moi"),
        )

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_identity.return_value = {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
            }
            q.fetch_experiment_arm_row.return_value = {"arm": "c"}

            with pytest.raises(ExperimentArmAlreadyExistsError):
                service.add_arm("exp-1", request)

        q.insert_experiment_arm.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once_with()
        put_conn.assert_called_once_with(conn)

    def test_missing_experiment_raises_not_found(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentArmCreateRequest(
            arm="c",
            stimulus_definition=_stimulus_definition("Moi"),
        )

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_identity.return_value = None

            with pytest.raises(ExperimentNotFoundError):
                service.add_arm("missing", request)

        q.insert_experiment_arm.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once_with()
        put_conn.assert_called_once_with(conn)


class TestPatchArm:
    def test_patch_updates_stimulus_definition_only(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentArmPatchRequest(stimulus_definition=_stimulus_definition("Hei ystävä"))
        existing_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "a",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            "alpha_param": 5.0,
            "beta_param": 3.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        }
        updated_row = dict(existing_row)
        updated_row["stimulus_definition"] = StimulusDefinition.model_validate(
            _stimulus_definition("Hei ystävä")
        )

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.side_effect = [existing_row, updated_row]

            result = service.patch_arm("exp-1", "a", request)

        q.update_experiment_arm_metadata.assert_called_once_with(
            cursor,
            experiment_id="exp-1",
            arm="a",
            stimulus_definition=request.stimulus_definition,
            enabled=True,
            end_dated_at=None,
        )
        assert result.alpha_param == 5.0
        assert result.beta_param == 3.0
        assert result.stimulus_definition.expected_response_rule == "Hei ystävä"
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_patch_disable_sets_end_dated_at_without_rewriting_posterior(self) -> None:
        now = datetime(2026, 4, 17, 12, 10, tzinfo=UTC)
        service, conn, cursor, put_conn = _service(now)
        request = ExperimentArmPatchRequest(enabled=False)
        existing_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "a",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            "alpha_param": 7.0,
            "beta_param": 4.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        }
        updated_row = dict(existing_row)
        updated_row["enabled"] = False
        updated_row["end_dated_at"] = now

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.side_effect = [existing_row, updated_row]

            result = service.patch_arm("exp-1", "a", request)

        q.update_experiment_arm_metadata.assert_called_once_with(
            cursor,
            experiment_id="exp-1",
            arm="a",
            stimulus_definition=StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            enabled=False,
            end_dated_at=now,
        )
        update_kwargs = q.update_experiment_arm_metadata.call_args.kwargs
        assert "alpha_param" not in update_kwargs
        assert "beta_param" not in update_kwargs
        assert result.alpha_param == 7.0
        assert result.beta_param == 4.0
        assert result.enabled is False
        assert result.end_dated_at == now
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_missing_arm_raises_not_found(self) -> None:
        service, conn, cursor, put_conn = _service()
        request = ExperimentArmPatchRequest(stimulus_definition=_stimulus_definition("Hei"))

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.return_value = None
            q.fetch_experiment_identity.return_value = {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
            }

            with pytest.raises(ExperimentArmNotFoundError):
                service.patch_arm("exp-1", "missing", request)

        conn.commit.assert_not_called()
        conn.rollback.assert_called_once_with()
        put_conn.assert_called_once_with(conn)


class TestDeleteArm:
    def test_unused_prior_arm_is_hard_deleted(self) -> None:
        service, conn, cursor, put_conn = _service()
        existing_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "unused",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            "alpha_param": 1.0,
            "beta_param": 1.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        }

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.return_value = existing_row
            q.fetch_experiment_arm_selection_count.return_value = 0

            result = service.delete_arm("exp-1", "unused")

        assert result.deleted is True
        assert result.posterior_preserved is False
        assert result.arm_state is None
        q.delete_experiment_arm.assert_called_once_with(
            cursor,
            experiment_id="exp-1",
            arm="unused",
        )
        q.update_experiment_arm_metadata.assert_not_called()
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_selected_arm_is_disabled_instead_of_deleted(self) -> None:
        now = datetime(2026, 4, 17, 12, 10, tzinfo=UTC)
        service, conn, cursor, put_conn = _service(now)
        existing_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "historical",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            "alpha_param": 7.0,
            "beta_param": 4.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        }
        updated_row = dict(existing_row)
        updated_row["enabled"] = False
        updated_row["end_dated_at"] = now

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.side_effect = [existing_row, updated_row]
            q.fetch_experiment_arm_selection_count.return_value = 3

            result = service.delete_arm("exp-1", "historical")

        q.delete_experiment_arm.assert_not_called()
        q.update_experiment_arm_metadata.assert_called_once_with(
            cursor,
            experiment_id="exp-1",
            arm="historical",
            stimulus_definition=StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            enabled=False,
            end_dated_at=now,
        )
        update_kwargs = q.update_experiment_arm_metadata.call_args.kwargs
        assert "alpha_param" not in update_kwargs
        assert "beta_param" not in update_kwargs
        assert result.deleted is False
        assert result.posterior_preserved is True
        assert result.arm_state is not None
        assert result.arm_state.alpha_param == 7.0
        assert result.arm_state.beta_param == 4.0
        assert result.arm_state.enabled is False
        assert result.arm_state.stimulus_definition.expected_response_rule == "Hello"
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_changed_posterior_without_rollup_is_still_preserved(self) -> None:
        now = datetime(2026, 4, 17, 12, 10, tzinfo=UTC)
        service, conn, cursor, put_conn = _service(now)
        existing_row = {
            "experiment_id": "exp-1",
            "label": "Experiment 1",
            "arm": "updated",
            "stimulus_definition": StimulusDefinition.model_validate(_stimulus_definition("Hello")),
            "alpha_param": 1.5,
            "beta_param": 1.0,
            "enabled": True,
            "end_dated_at": None,
            "updated_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
        }
        updated_row = dict(existing_row, enabled=False, end_dated_at=now)

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.side_effect = [existing_row, updated_row]
            q.fetch_experiment_arm_selection_count.return_value = 0

            result = service.delete_arm("exp-1", "updated")

        q.delete_experiment_arm.assert_not_called()
        q.update_experiment_arm_metadata.assert_called_once()
        assert result.deleted is False
        assert result.posterior_preserved is True
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        put_conn.assert_called_once_with(conn)

    def test_missing_arm_raises_not_found(self) -> None:
        service, conn, cursor, put_conn = _service()

        with patch("services.api.services.experiment_admin_service.q") as q:
            q.fetch_experiment_arm_row.return_value = None
            q.fetch_experiment_identity.return_value = {
                "experiment_id": "exp-1",
                "label": "Experiment 1",
            }

            with pytest.raises(ExperimentArmNotFoundError):
                service.delete_arm("exp-1", "missing")

        q.delete_experiment_arm.assert_not_called()
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once_with()
        put_conn.assert_called_once_with(conn)
