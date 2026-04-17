"""Runtime configuration for the Operator Console.

All values are read from environment variables so the operator can point
the console at any API endpoint (local dev, staging, production) without
editing code. Defaults target the local docker-compose stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ConsoleConfig:
    """Frozen configuration bundle passed through the app on startup."""

    api_base_url: str
    api_timeout_seconds: float
    session_poll_interval_ms: int
    experiment_poll_interval_ms: int
    physiology_poll_interval_ms: int
    default_experiment_id: str


def load_config() -> ConsoleConfig:
    """Build the config from environment variables with sensible defaults.

    Environment variables mirror the CLI: ``LSIE_API_URL`` is the canonical
    knob operators already know from `scripts/lsie_cli.py`.
    """
    api_base = os.environ.get("LSIE_API_URL", "http://localhost:8000").rstrip("/")
    return ConsoleConfig(
        api_base_url=api_base,
        api_timeout_seconds=float(os.environ.get("LSIE_API_TIMEOUT_S", "10")),
        session_poll_interval_ms=int(os.environ.get("LSIE_SESSION_POLL_MS", "5000")),
        experiment_poll_interval_ms=int(os.environ.get("LSIE_EXPERIMENT_POLL_MS", "10000")),
        physiology_poll_interval_ms=int(os.environ.get("LSIE_PHYSIO_POLL_MS", "5000")),
        default_experiment_id=os.environ.get("LSIE_EXPERIMENT_ID", "greeting_line_v1"),
    )
