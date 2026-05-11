"""SQLite-backed session lifecycle service for the desktop API shell."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np

from packages.schemas.operator_console import (
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
)
from services.api.services.session_lifecycle_service import (
    SessionLifecycleConflictError,
    SessionLifecycleService,
    _stable_session_id_for_action,
)
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema

BANDIT_POLICY_VERSION = "desktop_replay_v1"


@dataclass(frozen=True)
class ThompsonSelection:
    experiment_id: str
    experiment_row_id: int
    arm: str
    greeting_text: str
    candidate_arm_ids: list[str]
    posterior_by_arm: dict[str, dict[str, float]]
    sampled_theta_by_arm: dict[str, float]
    decision_context_hash: str
    random_seed: int


class LiveSessionControlPublisher(Protocol):
    def publish(self, message: LiveSessionControlMessage) -> None: ...


class SqliteSessionLifecycleService(SessionLifecycleService):
    """Accept session lifecycle writes directly into desktop SQLite."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        control_publisher: LiveSessionControlPublisher | None = None,
    ) -> None:
        self._db_path = db_path
        self._clock = clock
        self._control_publisher = control_publisher

    def request_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        session_id = _stable_session_id_for_action(request.client_action_id)
        now = self._clock()
        with self._connection() as conn:
            existing = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE ended_at IS NULL AND session_id != ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (str(session_id),),
            ).fetchone()
            if existing is not None:
                raise SessionLifecycleConflictError(
                    f"session {existing['session_id']} is already active; start not accepted"
                )
            selection = select_thompson_arm(
                conn,
                request.experiment_id,
                session_id=session_id,
                selection_time_utc=now,
            )
            selection_snapshot = (
                _snapshot_json(selection, selection_time_utc=now) if selection is not None else None
            )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, stream_url, experiment_id, active_arm, expected_greeting,
                    bandit_decision_snapshot, started_at, ended_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    str(session_id),
                    request.stream_url,
                    request.experiment_id,
                    selection.arm if selection is not None else None,
                    selection.greeting_text if selection is not None else None,
                    selection_snapshot,
                    _iso_utc(now),
                ),
            )
        if cursor.rowcount > 0:
            self._publish_control(
                LiveSessionControlMessage(
                    action="start",
                    session_id=session_id,
                    stream_url=request.stream_url,
                    experiment_id=request.experiment_id,
                    active_arm=selection.arm if selection is not None else None,
                    expected_greeting=selection.greeting_text if selection is not None else None,
                    timestamp_utc=now,
                )
            )
        return SessionLifecycleAccepted(
            action="start",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
        )

    def request_session_end(
        self,
        session_id: UUID,
        request: SessionEndRequest,
    ) -> SessionLifecycleAccepted:
        now = self._clock()
        with self._connection() as conn:
            target_row = conn.execute(
                """
                SELECT session_id, ended_at
                FROM sessions
                WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
            active_row = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            active_session_id = str(active_row["session_id"]) if active_row is not None else None
            if target_row is None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not active; end not accepted"
                )
            if target_row["ended_at"] is not None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} has already ended; end not accepted"
                )
            if active_row is None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not active; end not accepted"
                )
            if active_session_id != str(session_id):
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not the active session; end not accepted"
                )
            conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?
                WHERE session_id = ? AND ended_at IS NULL
                """,
                (_iso_utc(now), str(session_id)),
            )
        self._publish_control(
            LiveSessionControlMessage(
                action="end",
                session_id=session_id,
                timestamp_utc=now,
            )
        )
        return SessionLifecycleAccepted(
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            bootstrap_schema(conn)
            apply_writer_pragmas(conn)
            yield conn
        finally:
            conn.close()

    def _publish_control(self, message: LiveSessionControlMessage) -> None:
        if self._control_publisher is not None:
            self._control_publisher.publish(message)


def select_thompson_arm(
    conn: sqlite3.Connection,
    experiment_id: str | None,
    *,
    session_id: UUID,
    selection_time_utc: datetime,
) -> ThompsonSelection | None:
    if experiment_id is None:
        return None
    rows = conn.execute(
        """
        SELECT id, experiment_id, arm, greeting_text, alpha_param, beta_param
        FROM experiments
        WHERE experiment_id = ?
          AND enabled = 1
          AND end_dated_at IS NULL
        ORDER BY arm ASC
        """,
        (experiment_id,),
    ).fetchall()
    if not rows:
        return None
    seed = _bandit_random_seed(
        session_id=session_id,
        selection_time_utc=selection_time_utc,
        experiment_id=experiment_id,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    posterior_by_arm: dict[str, dict[str, float]] = {}
    sampled_theta_by_arm: dict[str, float] = {}
    selected = rows[0]
    best_sample = -1.0
    for row in rows:
        arm = str(row["arm"])
        alpha = float(row["alpha_param"])
        beta_param = float(row["beta_param"])
        posterior_by_arm[arm] = {"alpha": alpha, "beta": beta_param}
        sample = float(rng.beta(alpha, beta_param))
        sampled_theta_by_arm[arm] = sample
        if sample > best_sample:
            best_sample = sample
            selected = row
    selected_arm = str(selected["arm"])
    candidate_arm_ids = list(posterior_by_arm)
    return ThompsonSelection(
        experiment_id=str(selected["experiment_id"]),
        experiment_row_id=int(selected["id"]),
        arm=selected_arm,
        greeting_text=str(selected["greeting_text"]),
        candidate_arm_ids=candidate_arm_ids,
        posterior_by_arm=posterior_by_arm,
        sampled_theta_by_arm=sampled_theta_by_arm,
        decision_context_hash=_decision_context_hash(
            experiment_id=experiment_id,
            experiment_row_id=int(selected["id"]),
            candidate_arm_ids=candidate_arm_ids,
            posterior_by_arm=posterior_by_arm,
            selected_arm_id=selected_arm,
        ),
        random_seed=seed,
    )


def _bandit_random_seed(
    *,
    session_id: UUID,
    selection_time_utc: datetime,
    experiment_id: str,
) -> int:
    seed_material = "".join(
        (
            str(session_id),
            _iso_utc(selection_time_utc),
            experiment_id,
            BANDIT_POLICY_VERSION,
        )
    )
    return int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")


def _decision_context_hash(
    *,
    experiment_id: str,
    experiment_row_id: int,
    candidate_arm_ids: list[str],
    posterior_by_arm: dict[str, dict[str, float]],
    selected_arm_id: str,
) -> str:
    context = {
        "experiment_code": experiment_id,
        "experiment_row_id": experiment_row_id,
        "candidate_arm_ids": candidate_arm_ids,
        "posterior_by_arm": posterior_by_arm,
        "selected_arm_id": selected_arm_id,
        "policy_version": BANDIT_POLICY_VERSION,
    }
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def snapshot_dict(
    selection: ThompsonSelection,
    *,
    selection_time_utc: datetime,
) -> dict[str, object]:
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": selection_time_utc,
        "experiment_id": selection.experiment_row_id,
        "policy_version": BANDIT_POLICY_VERSION,
        "selected_arm_id": selection.arm,
        "candidate_arm_ids": selection.candidate_arm_ids,
        "posterior_by_arm": selection.posterior_by_arm,
        "sampled_theta_by_arm": selection.sampled_theta_by_arm,
        "expected_greeting": selection.greeting_text,
        "decision_context_hash": selection.decision_context_hash,
        "random_seed": selection.random_seed,
    }


def _snapshot_json(selection: ThompsonSelection, *, selection_time_utc: datetime) -> str:
    return json.dumps(
        snapshot_dict(selection, selection_time_utc=selection_time_utc),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
