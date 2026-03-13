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
_mock_fastapi.APIRouter = type(
    "APIRouter",
    (),
    {
        "__init__": lambda self: None,
        "get": lambda self, *a, **kw: lambda fn: fn,
        "post": lambda self, *a, **kw: lambda fn: fn,
    },
)


def _http_exc_init(self: object, status_code: int = 500, detail: str = "") -> None:
    self.status_code = status_code  # type: ignore[attr-defined]
    self.detail = detail  # type: ignore[attr-defined]


_mock_fastapi.HTTPException = type(
    "HTTPException",
    (Exception,),
    {"__init__": _http_exc_init},
)
_mock_fastapi.Query = lambda default=None, **kw: default

sys.modules.setdefault("fastapi", _mock_fastapi)

# --- Mock psycopg2 ---
_mock_pg = MagicMock()
sys.modules.setdefault("psycopg2", _mock_pg)
sys.modules.setdefault("psycopg2.pool", _mock_pg.pool)
sys.modules.setdefault("psycopg2.extensions", _mock_pg.extensions)
