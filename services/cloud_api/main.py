"""Cloud control-plane FastAPI application for WS5 P1."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.cloud_api.db.connection import close_pool, init_pool
from services.cloud_api.middleware.forbid_raw import forbid_raw_payload_middleware
from services.cloud_api.routes import auth, experiments, sessions, telemetry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    await init_pool()
    try:
        yield
    finally:
        await close_pool()


app = FastAPI(
    title="LSIE-MLF Cloud Control Plane",
    description="HTTPS telemetry and experiment-control surface for v4 desktop clients.",
    version="4.0.0-ws5-p1",
    lifespan=lifespan,
)
app.middleware("http")(forbid_raw_payload_middleware)

app.include_router(telemetry.router, prefix="/v4", tags=["telemetry"])
app.include_router(experiments.router, prefix="/v4", tags=["experiments"])
app.include_router(sessions.router, prefix="/v4", tags=["sessions"])
app.include_router(auth.router, prefix="/v4", tags=["auth"])
