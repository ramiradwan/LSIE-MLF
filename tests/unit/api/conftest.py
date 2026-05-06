"""Shared mocks for API unit tests.

FastAPI and psycopg2 are container-only dependencies. Mock them at the
sys.modules level before any API module is imported.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from typing import Any, TypeAlias
from unittest.mock import MagicMock

Handler: TypeAlias = Callable[..., Any]
RouteDecorator: TypeAlias = Callable[[Handler], Handler]

# --- Mock FastAPI before any API import when the dependency is unavailable ---
# The scoped unit tests can run with a shim in minimal environments, but the
# full unit+integration CI scope installs real FastAPI and later integration
# tests require its dependency override/TestClient behavior. Prefer the real
# dependency when importable so this subtree conftest does not poison
# sys.modules for unrelated integration collection.
_real_fastapi_available = (
    find_spec("fastapi") is not None
    and find_spec("fastapi.middleware") is not None
    and find_spec("fastapi.middleware.cors") is not None
    and find_spec("fastapi.testclient") is not None
)

_mock_fastapi: Any = ModuleType("fastapi")


class APIRouter:
    """Minimal APIRouter shim for unit tests."""

    def __init__(
        self,
        *,
        prefix: str = "",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.prefix = prefix
        self.tags = tags or []
        self.routes: list[SimpleNamespace] = []

    def get(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> RouteDecorator:
        del args, kwargs

        def decorator(fn: Handler) -> Handler:
            self.routes.append(SimpleNamespace(path=path, endpoint=fn, methods={"GET"}))
            return fn

        return decorator

    def post(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> RouteDecorator:
        del args, kwargs

        def decorator(fn: Handler) -> Handler:
            self.routes.append(SimpleNamespace(path=path, endpoint=fn, methods={"POST"}))
            return fn

        return decorator

    def patch(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> RouteDecorator:
        del args, kwargs

        def decorator(fn: Handler) -> Handler:
            self.routes.append(SimpleNamespace(path=path, endpoint=fn, methods={"PATCH"}))
            return fn

        return decorator

    def delete(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> RouteDecorator:
        del args, kwargs

        def decorator(fn: Handler) -> Handler:
            self.routes.append(SimpleNamespace(path=path, endpoint=fn, methods={"DELETE"}))
            return fn

        return decorator


class FastAPI:
    """Minimal FastAPI shim that records included routes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.routes: list[SimpleNamespace] = []

    def include_router(
        self,
        router: APIRouter,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None:
        del tags
        router_prefix = getattr(router, "prefix", "")
        for route in router.routes:
            self.routes.append(
                SimpleNamespace(
                    path=f"{prefix}{router_prefix}{route.path}",
                    endpoint=route.endpoint,
                    methods=route.methods,
                )
            )

    def add_middleware(self, *args: Any, **kwargs: Any) -> None:
        """Accept middleware registration calls used by main.py."""
        del args, kwargs


class HTTPError(Exception):
    """Minimal HTTPException-compatible shim."""

    def __init__(self, status_code: int = 500, detail: str = "") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Request:
    """Placeholder FastAPI Request type for unit tests."""

    async def body(self) -> bytes:
        raise NotImplementedError


def query(default: Any = None, **kwargs: Any) -> Any:
    """Return the default value for Query parameters in unit tests."""
    del kwargs
    return default


def body(default: Any = None, **kwargs: Any) -> Any:
    """Return the default value for Body parameters in unit tests."""
    del kwargs
    return default


def depends(dependency: Any = None) -> Any:
    """Return dependency placeholder for unit tests."""
    return dependency


def header(default: Any = None, **kwargs: Any) -> Any:
    """Return the default value for Header parameters in unit tests."""
    del kwargs
    return default


_mock_fastapi.APIRouter = APIRouter
_mock_fastapi.FastAPI = FastAPI
_mock_fastapi.HTTPException = HTTPError
_mock_fastapi.Request = Request
_mock_fastapi.Query = query
_mock_fastapi.Body = body
_mock_fastapi.Depends = depends
_mock_fastapi.Header = header

if not _real_fastapi_available:
    sys.modules.setdefault("fastapi", _mock_fastapi)

    # Optional FastAPI submodules sometimes imported by main.py/tests.
    _mock_fastapi_middleware: Any = ModuleType("fastapi.middleware")
    _mock_fastapi_middleware_cors: Any = ModuleType("fastapi.middleware.cors")

    class CORSMiddleware:
        """Placeholder CORSMiddleware for app.add_middleware()."""

        pass

    _mock_fastapi_middleware_cors.CORSMiddleware = CORSMiddleware

    sys.modules.setdefault("fastapi.middleware", _mock_fastapi_middleware)
    sys.modules.setdefault("fastapi.middleware.cors", _mock_fastapi_middleware_cors)

    # Optional placeholder if any test imports fastapi.testclient.
    _mock_fastapi_testclient: Any = ModuleType("fastapi.testclient")
    _mock_fastapi_testclient.TestClient = MagicMock
    sys.modules.setdefault("fastapi.testclient", _mock_fastapi_testclient)

# --- Mock psycopg2 ---
_mock_pg: Any = MagicMock()
sys.modules.setdefault("psycopg2", _mock_pg)
sys.modules.setdefault("psycopg2.pool", _mock_pg.pool)
sys.modules.setdefault("psycopg2.extensions", _mock_pg.extensions)
sys.modules.setdefault("psycopg2.extras", _mock_pg.extras)
