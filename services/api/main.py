"""
API Server — §3.1 / §9.1

FastAPI application serving REST endpoints on port 8000.
ASGI entry point via Uvicorn. Dependency injection for
Persistent Store connection pool.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI

from services.api.db.connection import close_pool, init_pool
from services.api.routes import (
    comodulation,
    encounters,
    experiments,
    health,
    metrics,
    operator,
    physiology,
    sessions,
)
from services.api.services.oura_hydration_service import OuraHydrationService

logger = logging.getLogger(__name__)


def _create_redis_client() -> Any:
    import redis as redis_lib

    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis_lib.Redis.from_url(redis_url, decode_responses=True)


def _start_oura_hydration_worker() -> tuple[threading.Thread, Any] | None:
    if not os.environ.get("OURA_CLIENT_ID", "").strip():
        logger.info("Oura hydration worker disabled; OURA_CLIENT_ID not configured")
        return None

    redis_client = _create_redis_client()
    service = OuraHydrationService(redis_client=redis_client)
    thread = threading.Thread(
        target=service.run_forever,
        name="oura-hydration-worker",
        daemon=True,
    )
    thread.start()
    logger.info("Started Oura hydration worker")
    return thread, redis_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: init and teardown Persistent Store pool."""
    del app
    await init_pool()
    hydration_runtime = _start_oura_hydration_worker()
    try:
        yield
    finally:
        if hydration_runtime is not None:
            _, redis_client = hydration_runtime
            if hasattr(redis_client, "close"):
                with suppress(Exception):
                    redis_client.close()
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
app.include_router(encounters.router, prefix="/api/v1", tags=["encounters"])
app.include_router(experiments.router, prefix="/api/v1", tags=["experiments"])
app.include_router(physiology.router, prefix="/api/v1", tags=["physiology"])
app.include_router(comodulation.router, prefix="/api/v1", tags=["comodulation"])
# §4.E.1/§10.2 — Operator Console aggregate surface.
app.include_router(operator.router, prefix="/api/v1")
