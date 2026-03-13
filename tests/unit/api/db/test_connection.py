"""
Tests for services/api/db/connection.py — Phase 2.2 validation.

Verifies connection pool lifecycle against §2 step 7.
Uses mocked psycopg2 pool to avoid requiring a live database.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_psycopg2(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock psycopg2 into sys.modules before importing connection."""
    mock_pg = MagicMock()
    mock_pool_mod = MagicMock()
    mock_pg.pool = mock_pool_mod
    monkeypatch.setitem(sys.modules, "psycopg2", mock_pg)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", mock_pool_mod)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", mock_pg.extensions)
    return mock_pg


@pytest.fixture(autouse=True)
def _reset_pool() -> None:
    """Ensure pool is reset between tests."""
    # Import after mock is in place
    from services.api.db import connection
    connection._pool = None


class TestConnectionPool:
    """§2 step 7 — psycopg2 connection pool lifecycle."""

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
    })
    def test_init_pool_creates_pool(self, mock_psycopg2: MagicMock) -> None:
        """§2 step 7 — init_pool creates ThreadedConnectionPool."""
        from services.api.db import connection
        asyncio.get_event_loop().run_until_complete(connection.init_pool())
        assert connection._pool is not None

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
    })
    def test_init_pool_idempotent(self, mock_psycopg2: MagicMock) -> None:
        """init_pool only creates pool once."""
        from services.api.db import connection
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connection.init_pool())
        first_pool = connection._pool
        loop.run_until_complete(connection.init_pool())
        assert connection._pool is first_pool

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
    })
    def test_close_pool(self, mock_psycopg2: MagicMock) -> None:
        """close_pool calls closeall and resets to None."""
        from services.api.db import connection
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connection.init_pool())
        pool_ref = connection._pool
        loop.run_until_complete(connection.close_pool())
        pool_ref.closeall.assert_called_once()
        assert connection._pool is None

    def test_get_connection_raises_without_pool(self) -> None:
        """get_connection raises RuntimeError if pool not initialized."""
        from services.api.db import connection
        with pytest.raises(RuntimeError, match="not initialized"):
            connection.get_connection()

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
    })
    def test_get_connection_returns_conn(self, mock_psycopg2: MagicMock) -> None:
        """get_connection returns a connection from the pool."""
        from services.api.db import connection
        asyncio.get_event_loop().run_until_complete(connection.init_pool())
        mock_conn = MagicMock()
        connection._pool.getconn.return_value = mock_conn
        result = connection.get_connection()
        assert result is mock_conn

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
    })
    def test_put_connection(self, mock_psycopg2: MagicMock) -> None:
        """put_connection returns connection to pool."""
        from services.api.db import connection
        asyncio.get_event_loop().run_until_complete(connection.init_pool())
        mock_conn = MagicMock()
        connection.put_connection(mock_conn)
        connection._pool.putconn.assert_called_once_with(mock_conn)

    @patch.dict("os.environ", {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "testdb",
        "POSTGRES_HOST": "myhost",
        "POSTGRES_PORT": "5433",
    })
    def test_dsn_from_env(self) -> None:
        """§2 step 7 — DSN built from environment variables."""
        from services.api.db import connection
        dsn = connection._get_dsn()
        assert dsn["host"] == "myhost"
        assert dsn["port"] == 5433
        assert dsn["user"] == "test"
        assert dsn["dbname"] == "testdb"
