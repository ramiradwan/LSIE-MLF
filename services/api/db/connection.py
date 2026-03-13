"""
Database Connection Pool — §2 step 7

psycopg2-binary PostgreSQL connection pool for the API Server.
Used for querying metrics from the Persistent Store.

§2 step 7: parameterized INSERT, DOUBLE PRECISION metrics, TIMESTAMPTZ timestamps.
Connection parameters sourced from environment variables.
"""

from __future__ import annotations

import os
from typing import Any

# Module-level pool instance — initialized once during app lifespan (§9.6)
_pool: Any = None


def _get_dsn() -> dict[str, Any]:
    """Build DSN kwargs from environment variables (§2 step 7)."""
    return {
        "host": os.environ.get("POSTGRES_HOST", "postgres"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": os.environ["POSTGRES_DB"],
    }


async def init_pool(minconn: int = 2, maxconn: int = 10) -> None:
    """
    Initialize psycopg2 threaded connection pool to Persistent Store.

    §2 step 7 — Connection pool is created at API startup via FastAPI lifespan.
    Environment variables: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
    POSTGRES_PASSWORD, POSTGRES_DB.
    """
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return
    from psycopg2 import pool as pg_pool  # Lazy import — container-only dep

    dsn = _get_dsn()
    _pool = pg_pool.ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, **dsn)


async def close_pool() -> None:
    """Close all connections in the pool. Called during API shutdown."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_connection() -> Any:
    """
    Get a connection from the pool.

    Returns a psycopg2 connection object. Caller MUST return it via
    put_connection() or use it as a context manager.

    Raises RuntimeError if the pool has not been initialized.
    """
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call init_pool() first.")
    return _pool.getconn()


def put_connection(conn: Any) -> None:
    """Return a connection to the pool."""
    if _pool is not None:
        _pool.putconn(conn)
