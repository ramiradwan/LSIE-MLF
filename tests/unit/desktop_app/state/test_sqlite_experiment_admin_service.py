"""SQLite-backed desktop experiment admin service tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.experiments import (
    ExperimentArmCreateRequest,
    ExperimentArmPatchRequest,
    ExperimentArmSeedRequest,
    ExperimentCreateRequest,
)
from services.api.services.experiment_admin_service import (
    ExperimentArmAlreadyExistsError,
    ExperimentMutationValidationError,
    ExperimentNotFoundError,
)
from services.desktop_app.state.sqlite_experiment_admin_service import (
    SqliteExperimentAdminService,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")


def _service(db: Path) -> SqliteExperimentAdminService:
    return SqliteExperimentAdminService(db, clock=lambda: _NOW)


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text=text,
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


def _create_request() -> ExperimentCreateRequest:
    return ExperimentCreateRequest(
        experiment_id="desktop_exp",
        label="Desktop experiment",
        arms=[
            ExperimentArmSeedRequest(arm="warm", stimulus_definition=_stimulus_definition("Hei"))
        ],
    )


def _seed_experiment(db: Path) -> None:
    _service(db).create_experiment(_create_request())


def _fetch_arm(db: Path, arm: str) -> sqlite3.Row | None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT experiment_id, label, arm, stimulus_definition, alpha_param, beta_param,
                   enabled, end_dated_at, updated_at
            FROM experiments
            WHERE experiment_id = ? AND arm = ?
            """,
            ("desktop_exp", arm),
        ).fetchone()
        return cast("sqlite3.Row | None", row)
    finally:
        conn.close()


def test_create_experiment_seeds_arms_at_beta_prior(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"

    response = _service(db).create_experiment(_create_request())

    assert response.experiment_id == "desktop_exp"
    assert response.label == "Desktop experiment"
    assert len(response.arms) == 1
    arm = response.arms[0]
    assert arm.arm == "warm"
    assert arm.stimulus_definition == _stimulus_definition("Hei")
    assert arm.alpha_param == 1.0
    assert arm.beta_param == 1.0
    assert arm.enabled is True
    assert arm.updated_at == _NOW
    row = _fetch_arm(db, "warm")
    assert row is not None
    assert json.loads(str(row["stimulus_definition"])) == _stimulus_definition("Hei").model_dump(
        mode="json"
    )


def test_add_arm_inserts_enabled_beta_prior_arm(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)

    arm = _service(db).add_arm(
        "desktop_exp",
        ExperimentArmCreateRequest(
            arm="direct", stimulus_definition=_stimulus_definition("Question")
        ),
    )

    assert arm.arm == "direct"
    assert arm.label == "Desktop experiment"
    assert arm.stimulus_definition == _stimulus_definition("Question")
    assert arm.alpha_param == 1.0
    assert arm.beta_param == 1.0
    assert arm.enabled is True
    row = _fetch_arm(db, "direct")
    assert row is not None
    assert json.loads(str(row["stimulus_definition"])) == _stimulus_definition(
        "Question"
    ).model_dump(mode="json")


def test_add_arm_rejects_missing_experiment_or_duplicate_arm(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    service = _service(db)

    with pytest.raises(ExperimentNotFoundError):
        service.add_arm(
            "missing",
            ExperimentArmCreateRequest(arm="a", stimulus_definition=_stimulus_definition("A")),
        )

    _seed_experiment(db)
    with pytest.raises(ExperimentArmAlreadyExistsError):
        service.add_arm(
            "desktop_exp",
            ExperimentArmCreateRequest(arm="warm", stimulus_definition=_stimulus_definition("A")),
        )


def test_patch_arm_updates_stimulus_definition_and_can_disable(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)

    patched = _service(db).patch_arm(
        "desktop_exp",
        "warm",
        ExperimentArmPatchRequest(stimulus_definition=_stimulus_definition("Moi"), enabled=False),
    )

    assert patched.stimulus_definition == _stimulus_definition("Moi")
    assert patched.enabled is False
    assert patched.end_dated_at == _NOW
    row = _fetch_arm(db, "warm")
    assert row is not None
    assert json.loads(str(row["stimulus_definition"])) == _stimulus_definition("Moi").model_dump(
        mode="json"
    )


def test_patch_arm_rejects_enabled_true(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)

    request = ExperimentArmPatchRequest.model_construct(stimulus_definition=None, enabled=True)
    with pytest.raises(ExperimentMutationValidationError):
        _service(db).patch_arm("desktop_exp", "warm", request)


def test_delete_arm_hard_deletes_unused_beta_prior_arm(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)

    response = _service(db).delete_arm("desktop_exp", "warm")

    assert response.deleted is True
    assert response.posterior_preserved is False
    assert _fetch_arm(db, "warm") is None


def test_delete_arm_soft_disables_arm_with_posterior_history(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "UPDATE experiments SET alpha_param = 2.0 WHERE experiment_id = ? AND arm = ?",
            ("desktop_exp", "warm"),
        )
    finally:
        conn.close()

    response = _service(db).delete_arm("desktop_exp", "warm")

    assert response.deleted is False
    assert response.posterior_preserved is True
    assert response.arm_state is not None
    assert response.arm_state.enabled is False
    assert response.arm_state.end_dated_at == _NOW
    row = _fetch_arm(db, "warm")
    assert row is not None
    assert row["enabled"] == 0


def test_delete_arm_soft_disables_arm_with_selection_count(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_experiment(db)
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT INTO sessions (session_id, stream_url, experiment_id, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (str(SESSION_ID), "test://stream", "desktop_exp", "2026-04-01 12:00:00"),
        )
        conn.execute(
            """
            INSERT INTO encounter_log (
                session_id, segment_id, experiment_id, arm, timestamp_utc, gated_reward,
                p90_intensity, semantic_gate, n_frames_in_window
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(SESSION_ID),
                "a" * 64,
                "desktop_exp",
                "warm",
                "2026-04-01 12:01:00",
                0.0,
                0.0,
                0,
                0,
            ),
        )
    finally:
        conn.close()

    response = _service(db).delete_arm("desktop_exp", "warm")

    assert response.deleted is False
    assert response.posterior_preserved is True
    assert response.arm_state is not None
    assert response.arm_state.enabled is False
