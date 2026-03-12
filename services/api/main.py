"""
API Server — §3.1 / §9.1

FastAPI application serving REST endpoints on port 8000.
ASGI entry point via Uvicorn. Dependency injection for
Persistent Store connection pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.db.connection import close_pool, init_pool
from services.api.routes import health, metrics, sessions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: init and teardown DB pool."""
    await init_pool()
    yield
    await close_pool()


app = FastAPI(
    title="LSIE-MLF API Server",
    description="REST interface for the Live Stream Inference Engine.",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
