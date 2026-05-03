"""SQLite-backed OperatorReadService.

Specializes :class:`services.api.services.operator_read_service
.OperatorReadService` for the v4.0 desktop graph: query backend points
at :mod:`services.desktop_app.state.sqlite_operator_queries` and the
cursor lifecycle goes through :class:`SqliteReader`'s query-only
``sqlite3.Connection`` instead of a psycopg2 pool.

The DTO assembly layer is unchanged — the parent class' ``_build_*``
methods consume row dicts that this query backend produces in the same
shape as the Postgres repo. The two backends differ only in SQL dialect
and parameter style; the surface that ``operator.py`` routes touch is
identical, so the FastAPI dependency override is a one-liner in
``ui_api_shell``.

This module also packages a no-op ``subsystem_probe_runner``: the
desktop graph has no Postgres / Redis / Whisper-worker peers to probe,
so the operator console's Health page surfaces only the freshness
heuristics derived from the local SQLite write timestamps.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from packages.schemas.operator_console import HealthState, HealthSubsystemStatus
from services.api.services.operator_read_service import OperatorReadService
from services.api.services.subsystem_probes import ProbeResult
from services.desktop_app.state import sqlite_operator_queries
from services.desktop_app.state.sqlite_reader import SqliteReader


async def _no_subsystem_probes(**_: Any) -> list[ProbeResult]:
    """No-op probe runner for the desktop graph.

    The probes in :mod:`services.api.services.subsystem_probes` target
    Postgres, Redis, Azure OpenAI, and a server-side worker health
    endpoint — none of which exist in the v4.0 desktop process graph.
    The ``HealthSnapshot.subsystems`` rollup remains driven by the §12
    freshness heuristics over the SQLite write timestamps.
    """
    return []


class SqliteOperatorReadService(OperatorReadService):
    """OperatorReadService backed by the desktop SQLite store."""

    def __init__(self, reader: SqliteReader) -> None:
        super().__init__(
            get_conn=self._unused_get_conn,
            put_conn=self._unused_put_conn,
            redis_factory=None,
            subsystem_probe_runner=_no_subsystem_probes,
            queries=sqlite_operator_queries,
        )
        self._reader = reader

    def _build_health_rows(
        self,
        pulse: dict[str, Any],
        now: Any,
    ) -> list[HealthSubsystemStatus]:
        return self._build_desktop_process_health_rows(pulse, now)

    def _build_desktop_process_health_rows(
        self,
        pulse: dict[str, Any],
        now: Any,
    ) -> list[HealthSubsystemStatus]:
        subsystems: list[tuple[str, str, str, str, str, str | None, str | None, str | None]] = [
            (
                "ui_api_shell",
                "UI API Shell",
                "last_ui_api_shell_at",
                "process_restart",
                "verify ui_api_shell process is running",
                None,
                None,
                None,
            ),
            (
                "capture_supervisor",
                "Capture Supervisor",
                "last_capture_supervisor_at",
                "process_restart",
                "verify capture_supervisor process is running",
                None,
                None,
                None,
            ),
            (
                "module_c_orchestrator",
                "Module C Orchestrator",
                "last_module_c_orchestrator_at",
                "process_restart",
                "verify module_c_orchestrator process is running",
                None,
                None,
                None,
            ),
            (
                "gpu_ml_worker",
                "GPU ML Worker",
                "last_gpu_ml_worker_at",
                "process_restart",
                "verify gpu_ml_worker process is running",
                None,
                None,
                None,
            ),
            (
                "analytics_state_worker",
                "Analytics State Worker",
                "last_analytics_state_worker_at",
                "process_restart",
                "verify analytics_state_worker process is running",
                None,
                None,
                None,
            ),
            (
                "cloud_sync_worker",
                "Cloud Sync Worker",
                "last_cloud_sync_worker_at",
                "process_restart",
                "verify cloud_sync_worker process is running",
                None,
                None,
                None,
            ),
            (
                "adb",
                "Android Device Bridge",
                "last_adb_at",
                "device_reconnect",
                "connect Android device via USB and allow debugging",
                "adb_state",
                "adb_detail",
                "adb_hint",
            ),
            (
                "audio_capture",
                "Audio Capture",
                "last_audio_capture_at",
                "capture_restart",
                "verify scrcpy audio recording is running",
                "audio_capture_state",
                "audio_capture_detail",
                "audio_capture_hint",
            ),
            (
                "video_capture",
                "Video Capture",
                "last_video_capture_at",
                "capture_restart",
                "verify scrcpy video recording is running",
                "video_capture_state",
                "video_capture_detail",
                "video_capture_hint",
            ),
        ]
        rows: list[HealthSubsystemStatus] = []
        for (
            key,
            label,
            field,
            recovery_mode,
            hint,
            state_field,
            detail_field,
            hint_field,
        ) in subsystems:
            state, detail = self._classify_subsystem_for_desktop(pulse.get(field), now)
            if state_field is not None and isinstance(pulse.get(state_field), str):
                state = HealthState(pulse[state_field])
            if detail_field is not None and pulse.get(detail_field) is not None:
                detail = str(pulse[detail_field])
            row_hint = hint
            if hint_field is not None and pulse.get(hint_field) is not None:
                row_hint = str(pulse[hint_field])
            needs_action = state.value in {"degraded", "recovering", "unknown"}
            rows.append(
                HealthSubsystemStatus(
                    subsystem_key=key,
                    label=label,
                    state=state,
                    last_success_utc=self._ensure_utc_for_desktop(pulse.get(field)),
                    detail=detail,
                    recovery_mode=recovery_mode if needs_action else None,
                    operator_action_hint=row_hint if needs_action else None,
                )
            )
        rows.append(
            HealthSubsystemStatus(
                subsystem_key="live_analytics_producer",
                label="Live Analytics Producer",
                state=HealthState.DEGRADED,
                last_success_utc=None,
                detail=(
                    "Desktop Module C live dispatch is release-gated; connected capture "
                    "and ML worker health do not create encounter analytics."
                ),
                recovery_mode="deferred_activation",
                operator_action_hint=(
                    "Smile, semantic, reward, and transcription rows require a future "
                    "desktop-safe inference producer with authoritative stimulus timing."
                ),
            )
        )
        return rows

    def _classify_subsystem_for_desktop(self, value: Any, now: Any) -> Any:
        from services.api.services.operator_read_service import _classify_subsystem

        return _classify_subsystem(self._ensure_utc_for_desktop(value), now)

    def _ensure_utc_for_desktop(self, value: Any) -> Any:
        from services.api.services.operator_read_service import _ensure_utc

        return _ensure_utc(value)

    @staticmethod
    def _unused_get_conn() -> Any:
        # The base class' default ``get_conn`` is wired to the psycopg2
        # pool, which is unconfigured in the desktop runtime. The
        # SQLite override owns connection lifecycle in ``_cursor``;
        # this stub exists only so the parent ``__init__`` signature
        # accepts a callable.
        raise RuntimeError("SqliteOperatorReadService routes connections through _cursor")

    @staticmethod
    def _unused_put_conn(_conn: Any) -> None:
        return None

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Yield a query-only ``sqlite3.Cursor`` from the reader pool."""
        with self._reader.connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
