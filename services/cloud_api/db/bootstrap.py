"""Deterministic cloud PostgreSQL bootstrap entrypoint."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

from services.cloud_api.db import connection
from services.cloud_api.db.schema import SQL_BOOTSTRAP_FILES


class BootstrapCursor(Protocol):
    def execute(self, sql: str) -> None: ...

    def close(self) -> None: ...


class BootstrapConnection(Protocol):
    def cursor(self) -> BootstrapCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def apply_bootstrap_files(files: Sequence[Path] | None = None) -> tuple[str, ...]:
    ordered_files = SQL_BOOTSTRAP_FILES if files is None else files
    conn = connection.get_connection()
    applied: list[str] = []
    try:
        cur = conn.cursor()
        try:
            for path in ordered_files:
                cur.execute(path.read_text(encoding="utf-8"))
                applied.append(path.name)
        finally:
            cur.close()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        connection.put_connection(conn)
    return tuple(applied)


async def run_bootstrap(stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    try:
        await connection.init_pool(minconn=1, maxconn=1)
        applied = apply_bootstrap_files()
    except Exception:
        print("cloud-db-bootstrap status=failed", file=stderr)
        return 1
    finally:
        await connection.close_pool()

    print(f"cloud-db-bootstrap status=ok files_applied={len(applied)}", file=stdout)
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(run_bootstrap())


if __name__ == "__main__":
    raise SystemExit(main())
