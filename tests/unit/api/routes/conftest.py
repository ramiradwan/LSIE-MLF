"""
Shared mocks for API route tests.

FastAPI and psycopg2 are container-only dependencies. Mock them at
sys.modules level before any route module is imported.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# --- Mock FastAPI before any route import ---
_mock_fastapi = MagicMock()
# Provide real-ish APIRouter and HTTPException so route decorators work
_mock_fastapi.APIRouter = type("APIRouter", (), {
    "__init__": lambda self: None,
    "get": lambda self, *a, **kw: lambda fn: fn,
    "post": lambda self, *a, **kw: lambda fn: fn,
})
_mock_fastapi.HTTPException = type("HTTPException", (Exception,), {
    "__init__": lambda self, status_code=500, detail="": (
        setattr(self, "status_code", status_code) or  # type: ignore[func-returns-value]
        setattr(self, "detail", detail)
    ),
})
_mock_fastapi.Query = lambda default=None, **kw: default

sys.modules.setdefault("fastapi", _mock_fastapi)

# --- Mock psycopg2 ---
_mock_pg = MagicMock()
sys.modules.setdefault("psycopg2", _mock_pg)
sys.modules.setdefault("psycopg2.pool", _mock_pg.pool)
sys.modules.setdefault("psycopg2.extensions", _mock_pg.extensions)
