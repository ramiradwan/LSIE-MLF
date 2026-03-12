"""
Celery Application — ML Worker Entry Point

Configures the Celery distributed task queue connected to the
Message Broker (Redis) per §9.1 and §9.6.
"""

from __future__ import annotations

import os

from celery import Celery

# §0.3 — Message Broker: Redis in-memory data store
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "lsie_mlf",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # §6 — Redis append-only persistence ensures tasks survive broker restarts
    broker_transport_options={"visibility_timeout": 3600},
)

# Auto-discover tasks in the worker tasks module
celery_app.autodiscover_tasks(["services.worker.tasks"])
