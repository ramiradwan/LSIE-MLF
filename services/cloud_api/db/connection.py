"""PostgreSQL connection pool for the cloud control-plane API."""

from __future__ import annotations

import os
from typing import Any

_pool: Any = None


def _get_dsn() -> dict[str, Any]:
    return {
        "host": os.environ.get("CLOUD_POSTGRES_HOST", os.environ.get("POSTGRES_HOST", "postgres")),
        "port": int(os.environ.get("CLOUD_POSTGRES_PORT", os.environ.get("POSTGRES_PORT", "5432"))),
        "user": os.environ.get("CLOUD_POSTGRES_USER", os.environ.get("POSTGRES_USER", "postgres")),
        "password": os.environ.get(
            "CLOUD_POSTGRES_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "postgres")
        ),
        "dbname": os.environ.get("CLOUD_POSTGRES_DB", os.environ.get("POSTGRES_DB", "postgres")),
    }


async def init_pool(minconn: int = 2, maxconn: int = 10) -> None:
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return
    from psycopg2 import pool as pg_pool

    _pool = pg_pool.ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, **_get_dsn())


async def close_pool() -> None:
    global _pool  # noqa: PLW0603
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_connection() -> Any:
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call init_pool() first.")
    return _pool.getconn()


def put_connection(conn: Any) -> None:
    if _pool is not None:
        _pool.putconn(conn)


async def check_readiness() -> bool:
    conn: Any | None = None
    try:
        await init_pool()
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception:
        return False
    finally:
        if conn is not None:
            put_connection(conn)
    return True
