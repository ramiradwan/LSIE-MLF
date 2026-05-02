"""Transaction helper for cloud API services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from services.cloud_api.db.connection import get_connection, put_connection

T = TypeVar("T")


def run_in_transaction(operation: Callable[[Any], T]) -> T:
    conn = None
    try:
        conn = get_connection()
        result = operation(conn)
        conn.commit()
        return result
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            put_connection(conn)
