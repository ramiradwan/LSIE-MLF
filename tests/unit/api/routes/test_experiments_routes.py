"""Tests for `services/api/routes/experiments.py`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from packages.schemas.experiments import (
    ExperimentAdminResponse,
    ExperimentArmAdminResponse,
    ExperimentArmCreateRequest,
    ExperimentArmDeleteResponse,
    ExperimentArmPatchRequest,
    ExperimentCreateRequest,
)
from services.api.routes.experiments import (
    _rows_to_dicts,
    _serialize,
    add_experiment_arm,
    create_experiment,
    delete_experiment_arm,
    get_experiment,
    list_experiments,
    patch_experiment_arm,
)
from services.api.services.experiment_admin_service import (
    ExperimentAlreadyExistsError,
    ExperimentArmAlreadyExistsError,
    ExperimentArmNotFoundError,
    ExperimentMutationValidationError,
    ExperimentNotFoundError,
)


def _make_mock_cursor(
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> MagicMock:
    """Create a mock cursor with description and fetchall."""
    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.fetchall.return_value = rows
    cursor.fetchone.return_value = rows[0] if rows else None
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


class TestSerialize:
    """Helper serialization tests."""

    def test_datetime_serialized(self) -> None:
        dt = datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC)
        assert _serialize(dt) == "2025-04-01T12:00:00+00:00"

    def test_plain_value_passthrough(self) -> None:
        assert _serialize(42) == 42
        assert _serialize("hello") == "hello"
        assert _serialize(None) is None


class TestRowsToDicts:
    """Cursor result conversion."""

    def test_converts_rows(self) -> None:
        cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param"],
            [("greeting_line_v1", "warm_welcome", 3.5)],
        )

        result = _rows_to_dicts(cursor)

        assert result == [
            {
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "alpha_param": 3.5,
            }
        ]

    def test_empty_result(self) -> None:
        cursor = _make_mock_cursor(["experiment_id"], [])
        assert _rows_to_dicts(cursor) == []

    def test_none_description(self) -> None:
        cursor = MagicMock()
        cursor.description = None
        assert _rows_to_dicts(cursor) == []


class TestReadRoutes:
    """Legacy read shape is preserved."""

    def test_list_returns_experiment_ids(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id"],
            [("greeting_line_v1",), ("greeting_line_v2",)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(list_experiments())

        assert result == [
            {"experiment_id": "greeting_line_v1"},
            {"experiment_id": "greeting_line_v2"},
        ]

    def test_list_returns_empty_list(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(list_experiments())

        assert result == []

    def test_get_returns_arm_state(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [
                (
                    "greeting_line_v1",
                    "direct_ask",
                    1.8,
                    4.2,
                    datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC),
                ),
                (
                    "greeting_line_v1",
                    "warm_welcome",
                    3.5,
                    2.1,
                    datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC),
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(get_experiment("greeting_line_v1"))

        assert result["experiment_id"] == "greeting_line_v1"
        assert len(result["arms"]) == 2
        assert result["arms"][0]["arm"] == "direct_ask"
        assert result["arms"][0]["alpha_param"] == 1.8
        assert result["arms"][0]["beta_param"] == 4.2
        assert result["arms"][0]["updated_at"] == "2025-04-01T12:00:00+00:00"

    def test_get_404_unknown_experiment(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_experiment("missing"))

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert "No experiment found" in exc.detail

    def test_get_uses_parameterized_query(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [("greeting_line_v1", "arm_a", 1.0, 1.0, None)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
        ):
            asyncio.run(get_experiment("greeting_line_v1"))

        call_args = mock_cursor.execute.call_args
        assert "%(experiment_id)s" in call_args[0][0]
        assert call_args[0][1]["experiment_id"] == "greeting_line_v1"

    def test_runtime_error_becomes_503(self) -> None:
        with (
            patch(
                "services.api.routes.experiments.get_connection",
                side_effect=RuntimeError("Connection pool not initialized"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(list_experiments())

        exc: Any = exc_info.value
        assert exc.status_code == 503
        assert "Connection pool not initialized" in exc.detail

    def test_unexpected_read_error_becomes_500(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_cursor.execute.side_effect = Exception("boom")
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(list_experiments())

        exc: Any = exc_info.value
        assert exc.status_code == 500
        assert exc.detail == "Internal server error"

    def test_connection_returned_to_pool(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection") as mock_put,
        ):
            asyncio.run(list_experiments())

        mock_put.assert_called_once_with(mock_conn)


class TestWriteRouteRequestModels:
    """Validation of the additive write contract."""

    def test_patch_request_rejects_posterior_numeric_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentArmPatchRequest.model_validate({"posterior_alpha": 9.0})

    def test_create_request_rejects_posterior_numeric_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentCreateRequest.model_validate(
                {
                    "experiment_id": "exp-a",
                    "label": "Experiment A",
                    "arms": [
                        {
                            "arm": "hello",
                            "greeting_text": "Hi",
                            "posterior_alpha": 9.0,
                        }
                    ],
                }
            )

    def test_patch_request_rejects_rollup_numeric_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentArmPatchRequest.model_validate({"selection_count": 99})
        with pytest.raises(ValidationError):
            ExperimentArmPatchRequest.model_validate({"recent_reward_mean": 0.5})

    def test_patch_request_blocks_enabled_true(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentArmPatchRequest(enabled=True)

    def test_create_request_rejects_duplicate_arm_ids(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentCreateRequest.model_validate(
                {
                    "experiment_id": "exp-a",
                    "label": "Experiment A",
                    "arms": [
                        {"arm": "hello", "greeting_text": "Hi"},
                        {"arm": "hello", "greeting_text": "Hei"},
                    ],
                }
            )


class TestCreateExperimentRoute:
    def test_returns_serialized_admin_payload(self) -> None:
        svc = MagicMock()
        svc.create_experiment.return_value = ExperimentAdminResponse(
            experiment_id="greeting_line_v2",
            label="Greeting Line V2",
            arms=[
                ExperimentArmAdminResponse(
                    experiment_id="greeting_line_v2",
                    label="Greeting Line V2",
                    arm="warm",
                    greeting_text="Hi there!",
                    alpha_param=1.0,
                    beta_param=1.0,
                    enabled=True,
                    end_dated_at=None,
                    updated_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                )
            ],
        )
        request = ExperimentCreateRequest.model_validate(
            {
                "experiment_id": "greeting_line_v2",
                "label": "Greeting Line V2",
                "arms": [{"arm": "warm", "greeting_text": "Hi there!"}],
            }
        )

        result = asyncio.run(create_experiment(request=request, service=svc))

        assert result == {
            "experiment_id": "greeting_line_v2",
            "label": "Greeting Line V2",
            "arms": [
                {
                    "experiment_id": "greeting_line_v2",
                    "label": "Greeting Line V2",
                    "arm": "warm",
                    "greeting_text": "Hi there!",
                    "alpha_param": 1.0,
                    "beta_param": 1.0,
                    "enabled": True,
                    "end_dated_at": None,
                    "updated_at": "2026-04-17T12:00:00+00:00",
                }
            ],
        }
        svc.create_experiment.assert_called_once_with(request)

    def test_existing_experiment_becomes_409(self) -> None:
        svc = MagicMock()
        svc.create_experiment.side_effect = ExperimentAlreadyExistsError("exists")
        request = ExperimentCreateRequest.model_validate(
            {
                "experiment_id": "greeting_line_v2",
                "label": "Greeting Line V2",
                "arms": [{"arm": "warm", "greeting_text": "Hi there!"}],
            }
        )

        with pytest.raises(Exception) as exc_info:
            asyncio.run(create_experiment(request=request, service=svc))

        exc: Any = exc_info.value
        assert exc.status_code == 409
        assert exc.detail == "exists"


class TestAddExperimentArmRoute:
    def test_returns_serialized_arm_payload(self) -> None:
        svc = MagicMock()
        svc.add_arm.return_value = ExperimentArmAdminResponse(
            experiment_id="greeting_line_v1",
            label="Greeting Line V1",
            arm="new_arm",
            greeting_text="Hello there",
            alpha_param=1.0,
            beta_param=1.0,
            enabled=True,
            end_dated_at=None,
            updated_at=datetime(2026, 4, 17, 12, 5, tzinfo=UTC),
        )
        request = ExperimentArmCreateRequest(arm="new_arm", greeting_text="Hello there")

        result = asyncio.run(add_experiment_arm("greeting_line_v1", request=request, service=svc))

        assert result["alpha_param"] == 1.0
        assert result["beta_param"] == 1.0
        assert result["enabled"] is True
        assert result["updated_at"] == "2026-04-17T12:05:00+00:00"
        svc.add_arm.assert_called_once_with("greeting_line_v1", request)

    def test_duplicate_arm_becomes_409(self) -> None:
        svc = MagicMock()
        svc.add_arm.side_effect = ExperimentArmAlreadyExistsError("duplicate")
        request = ExperimentArmCreateRequest(arm="new_arm", greeting_text="Hello there")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(add_experiment_arm("greeting_line_v1", request=request, service=svc))

        exc: Any = exc_info.value
        assert exc.status_code == 409
        assert exc.detail == "duplicate"

    def test_missing_experiment_becomes_404(self) -> None:
        svc = MagicMock()
        svc.add_arm.side_effect = ExperimentNotFoundError("missing")
        request = ExperimentArmCreateRequest(arm="new_arm", greeting_text="Hello there")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(add_experiment_arm("missing", request=request, service=svc))

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert exc.detail == "missing"


class TestPatchExperimentArmRoute:
    def test_returns_serialized_patch_payload(self) -> None:
        svc = MagicMock()
        svc.patch_arm.return_value = ExperimentArmAdminResponse(
            experiment_id="greeting_line_v1",
            label="Greeting Line V1",
            arm="warm_welcome",
            greeting_text="Hei ystävä",
            alpha_param=5.0,
            beta_param=3.0,
            enabled=False,
            end_dated_at=datetime(2026, 4, 17, 12, 10, tzinfo=UTC),
            updated_at=datetime(2026, 4, 17, 12, 10, tzinfo=UTC),
        )
        request = ExperimentArmPatchRequest(greeting_text="Hei ystävä", enabled=False)

        result = asyncio.run(
            patch_experiment_arm(
                "greeting_line_v1",
                "warm_welcome",
                request=request,
                service=svc,
            )
        )

        assert result["greeting_text"] == "Hei ystävä"
        assert result["enabled"] is False
        assert result["alpha_param"] == 5.0
        assert result["beta_param"] == 3.0
        assert result["end_dated_at"] == "2026-04-17T12:10:00+00:00"
        svc.patch_arm.assert_called_once_with("greeting_line_v1", "warm_welcome", request)

    def test_missing_arm_becomes_404(self) -> None:
        svc = MagicMock()
        svc.patch_arm.side_effect = ExperimentArmNotFoundError("missing arm")
        request = ExperimentArmPatchRequest(greeting_text="Hei ystävä")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(
                patch_experiment_arm(
                    "greeting_line_v1",
                    "missing",
                    request=request,
                    service=svc,
                )
            )

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert exc.detail == "missing arm"

    def test_unsupported_mutation_becomes_422(self) -> None:
        svc = MagicMock()
        svc.patch_arm.side_effect = ExperimentMutationValidationError("unsupported")
        request = ExperimentArmPatchRequest(greeting_text="Hei ystävä")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(
                patch_experiment_arm(
                    "greeting_line_v1",
                    "warm_welcome",
                    request=request,
                    service=svc,
                )
            )

        exc: Any = exc_info.value
        assert exc.status_code == 422
        assert exc.detail == "unsupported"


class TestDeleteExperimentArmRoute:
    def test_returns_serialized_delete_guard_payload(self) -> None:
        svc = MagicMock()
        svc.delete_arm.return_value = ExperimentArmDeleteResponse(
            experiment_id="greeting_line_v1",
            arm="warm_welcome",
            deleted=False,
            posterior_preserved=True,
            reason="arm has posterior history; disabled instead of hard-deleting",
            arm_state=ExperimentArmAdminResponse(
                experiment_id="greeting_line_v1",
                label="Greeting Line V1",
                arm="warm_welcome",
                greeting_text="Hei ystävä",
                alpha_param=5.0,
                beta_param=3.0,
                enabled=False,
                end_dated_at=datetime(2026, 4, 17, 12, 10, tzinfo=UTC),
                updated_at=datetime(2026, 4, 17, 12, 10, tzinfo=UTC),
            ),
        )

        result = asyncio.run(delete_experiment_arm("greeting_line_v1", "warm_welcome", service=svc))

        assert result["deleted"] is False
        assert result["posterior_preserved"] is True
        assert result["arm_state"]["alpha_param"] == 5.0
        assert result["arm_state"]["beta_param"] == 3.0
        assert result["arm_state"]["enabled"] is False
        assert result["arm_state"]["end_dated_at"] == "2026-04-17T12:10:00+00:00"
        svc.delete_arm.assert_called_once_with("greeting_line_v1", "warm_welcome")

    def test_missing_arm_becomes_404(self) -> None:
        svc = MagicMock()
        svc.delete_arm.side_effect = ExperimentArmNotFoundError("missing arm")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(delete_experiment_arm("greeting_line_v1", "missing", service=svc))

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert exc.detail == "missing arm"

    def test_runtime_error_becomes_503(self) -> None:
        svc = MagicMock()
        svc.delete_arm.side_effect = RuntimeError("Connection pool not initialized")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(delete_experiment_arm("greeting_line_v1", "warm_welcome", service=svc))

        exc: Any = exc_info.value
        assert exc.status_code == 503
        assert "Connection pool not initialized" in exc.detail


class TestReadRouteHttpExceptions:
    def test_route_preserves_http_exceptions_from_read_paths(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.experiments.get_connection", return_value=mock_conn),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(HTTPException),
        ):
            asyncio.run(get_experiment("nonexistent"))
