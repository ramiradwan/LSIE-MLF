"""Unit tests for `services/api/repos/experiments_queries.py`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from services.api.repos import experiments_queries as q


def _cursor(
    *,
    columns: list[str] | None = None,
    rows: list[tuple[object, ...]] | None = None,
) -> MagicMock:
    cursor = MagicMock()
    if columns is None:
        cursor.description = None
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = None
        return cursor
    cursor.description = [(column,) for column in columns]
    payload_rows = rows or []
    cursor.fetchall.return_value = payload_rows
    cursor.fetchone.return_value = payload_rows[0] if payload_rows else None
    return cursor


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(content_type="text", text=text),
        expected_stimulus_rule=text,
        expected_response_rule=text,
    )


class TestFetchExperimentAdminRows:
    def test_decodes_stimulus_definition_to_model(self) -> None:
        stimulus_definition = _stimulus_definition("Hei")
        cursor = _cursor(
            columns=[
                "experiment_id",
                "label",
                "arm",
                "stimulus_definition",
                "alpha_param",
                "beta_param",
                "enabled",
                "end_dated_at",
                "updated_at",
            ],
            rows=[
                (
                    "exp-1",
                    "Experiment 1",
                    "warm",
                    stimulus_definition.model_dump_json(),
                    1.0,
                    1.0,
                    True,
                    None,
                    datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                )
            ],
        )

        rows = q.fetch_experiment_admin_rows(cursor, "exp-1")

        cursor.execute.assert_called_once_with(
            q._SELECT_EXPERIMENT_ADMIN_ROWS_SQL,
            {"experiment_id": "exp-1"},
        )
        assert isinstance(rows[0]["stimulus_definition"], StimulusDefinition)
        assert rows[0]["stimulus_definition"].expected_response_rule == "Hei"

    def test_rejects_legacy_greeting_text_only_rows(self) -> None:
        cursor = _cursor(
            columns=[
                "experiment_id",
                "label",
                "arm",
                "stimulus_definition",
                "alpha_param",
                "beta_param",
                "enabled",
                "end_dated_at",
                "updated_at",
            ],
            rows=[
                (
                    "exp-1",
                    "Experiment 1",
                    "warm",
                    json.dumps({"greeting_text": "Hei"}),
                    1.0,
                    1.0,
                    True,
                    None,
                    datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                )
            ],
        )

        with pytest.raises(ValidationError):
            q.fetch_experiment_admin_rows(cursor, "exp-1")


class TestFetchExperimentArmRow:
    def test_decodes_single_arm_row_to_model(self) -> None:
        stimulus_definition = _stimulus_definition("Moi")
        cursor = _cursor(
            columns=[
                "experiment_id",
                "label",
                "arm",
                "stimulus_definition",
                "alpha_param",
                "beta_param",
                "enabled",
                "end_dated_at",
                "updated_at",
            ],
            rows=[
                (
                    "exp-1",
                    "Experiment 1",
                    "direct",
                    stimulus_definition.model_dump(mode="json"),
                    1.0,
                    1.0,
                    True,
                    None,
                    datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                )
            ],
        )

        row = q.fetch_experiment_arm_row(cursor, experiment_id="exp-1", arm="direct")

        assert row is not None
        assert isinstance(row["stimulus_definition"], StimulusDefinition)
        assert row["stimulus_definition"].stimulus_payload.text == "Moi"


class TestPersistenceBoundaries:
    def test_insert_serializes_stimulus_definition_only_at_storage_boundary(self) -> None:
        cursor = MagicMock()
        stimulus_definition = _stimulus_definition("Hei")

        q.insert_experiment_arm(
            cursor,
            experiment_id="exp-1",
            label="Experiment 1",
            arm="warm",
            stimulus_definition=stimulus_definition,
            alpha_param=1.0,
            beta_param=1.0,
            enabled=True,
            end_dated_at=None,
        )

        execute_args = cursor.execute.call_args.args
        assert execute_args[0] == q._INSERT_EXPERIMENT_ARM_SQL
        params = execute_args[1]
        assert (
            StimulusDefinition.model_validate_json(params["stimulus_definition"])
            == stimulus_definition
        )
        assert "greeting_text" not in params

    def test_update_serializes_stimulus_definition_only_at_storage_boundary(self) -> None:
        cursor = MagicMock()
        stimulus_definition = _stimulus_definition("Hei ystävä")

        q.update_experiment_arm_metadata(
            cursor,
            experiment_id="exp-1",
            arm="warm",
            stimulus_definition=stimulus_definition,
            enabled=False,
            end_dated_at=None,
        )

        execute_args = cursor.execute.call_args.args
        assert execute_args[0] == q._UPDATE_EXPERIMENT_ARM_METADATA_SQL
        params = execute_args[1]
        assert (
            StimulusDefinition.model_validate_json(params["stimulus_definition"])
            == stimulus_definition
        )
        assert "greeting_text" not in params
