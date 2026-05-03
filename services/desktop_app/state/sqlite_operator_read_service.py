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
