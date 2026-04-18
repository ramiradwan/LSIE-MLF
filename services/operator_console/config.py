"""
Runtime configuration for the Operator Console — Phase 3 of the Operator
Console cycle (SPEC-AMEND-008).

Everything the console needs to bootstrap lives here: the API base URL,
the request timeout, an `environment_label` for the title bar, and a
per-surface poll cadence. Each operator page polls its own endpoint on
its own interval so a slow `/health` does not back-pressure the
overview card, and so Phase 4's `PollingCoordinator` can scope jobs to
the active route without re-deriving intervals elsewhere.

All values are read from environment variables at startup. Defaults
target the local docker-compose stack. Every interval is validated
positive — a zero-or-negative poll interval would either busy-loop Qt
timers or silently stop polling, both of which are worse than a hard
failure at startup.

Spec references:
  §4.E.1         — operator-facing execution details
  SPEC-AMEND-008 — PySide6 Operator Console replaces Streamlit
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorConsoleConfig:
    """Frozen configuration bundle passed through the app on startup.

    `environment_label` surfaces in the title/status bar so operators
    never confuse a staging console with a production one. The ten
    `*_poll_ms` fields each own one page's refresh cadence; Phase 4's
    `PollingCoordinator` matches them to route lifecycle.
    """

    api_base_url: str
    api_timeout_seconds: float
    environment_label: str

    # Per-surface poll intervals. Each corresponds to one operator page
    # or attention queue. Phase 4 route-scopes the jobs so a hidden page
    # does not poll.
    overview_poll_ms: int
    session_header_poll_ms: int
    live_encounters_poll_ms: int
    experiments_poll_ms: int
    physiology_poll_ms: int
    comodulation_poll_ms: int
    health_poll_ms: int
    alerts_poll_ms: int
    sessions_poll_ms: int

    # Carried over from the scaffold — the CLI operator already knows
    # the `LSIE_EXPERIMENT_ID` knob and can point the console at a
    # non-default experiment for staging runs.
    default_experiment_id: str

    # --- scaffold-compat aliases -----------------------------------
    # `SessionsView`/`MainWindow` from the initial scaffold still read
    # the old `*_interval_ms` names. Phase 6/10 rewrite those consumers
    # around the new names; until then the aliases keep the scaffold
    # functional without duplicating env reads.

    @property
    def session_poll_interval_ms(self) -> int:
        return self.sessions_poll_ms

    @property
    def experiment_poll_interval_ms(self) -> int:
        return self.experiments_poll_ms

    @property
    def physiology_poll_interval_ms(self) -> int:
        return self.physiology_poll_ms


# Backwards-compat alias — external imports of `ConsoleConfig` keep
# working through the Phase-3 rename; Phase 6/10 remove the last
# references.
ConsoleConfig = OperatorConsoleConfig


def _int_env(environ: Mapping[str, str], key: str, default: int) -> int:
    """Read an integer env var; fall back to `default` if unset or blank."""
    raw = environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float_env(environ: Mapping[str, str], key: str, default: float) -> float:
    """Read a float env var; fall back to `default` if unset or blank."""
    raw = environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def load_config(environ: Mapping[str, str] | None = None) -> OperatorConsoleConfig:
    """Build the config from environment variables with sensible defaults.

    `environ` is injectable so tests can pass a dict without monkey-
    patching `os.environ`. Every `*_poll_ms` value is validated positive
    — a non-positive interval at runtime either busy-loops the QTimer
    or silently stops the poll, so we refuse at startup instead.
    """
    env = environ if environ is not None else os.environ
    api_base = env.get("LSIE_API_URL", "http://localhost:8000").rstrip("/")

    config = OperatorConsoleConfig(
        api_base_url=api_base,
        api_timeout_seconds=_float_env(env, "LSIE_API_TIMEOUT_S", 10.0),
        environment_label=env.get("LSIE_ENV_LABEL", "local"),
        overview_poll_ms=_int_env(env, "LSIE_OVERVIEW_POLL_MS", 3000),
        session_header_poll_ms=_int_env(env, "LSIE_SESSION_HEADER_POLL_MS", 3000),
        live_encounters_poll_ms=_int_env(env, "LSIE_LIVE_ENCOUNTERS_POLL_MS", 2000),
        experiments_poll_ms=_int_env(env, "LSIE_EXPERIMENTS_POLL_MS", 10000),
        physiology_poll_ms=_int_env(env, "LSIE_PHYSIO_POLL_MS", 5000),
        comodulation_poll_ms=_int_env(env, "LSIE_COMOD_POLL_MS", 15000),
        health_poll_ms=_int_env(env, "LSIE_HEALTH_POLL_MS", 5000),
        alerts_poll_ms=_int_env(env, "LSIE_ALERTS_POLL_MS", 5000),
        sessions_poll_ms=_int_env(env, "LSIE_SESSION_POLL_MS", 5000),
        default_experiment_id=env.get("LSIE_EXPERIMENT_ID", "greeting_line_v1"),
    )

    _validate_positive_intervals(config)
    return config


def _validate_positive_intervals(config: OperatorConsoleConfig) -> None:
    """Raise `ValueError` if any poll interval is not strictly positive."""
    checks = {
        "overview_poll_ms": config.overview_poll_ms,
        "session_header_poll_ms": config.session_header_poll_ms,
        "live_encounters_poll_ms": config.live_encounters_poll_ms,
        "experiments_poll_ms": config.experiments_poll_ms,
        "physiology_poll_ms": config.physiology_poll_ms,
        "comodulation_poll_ms": config.comodulation_poll_ms,
        "health_poll_ms": config.health_poll_ms,
        "alerts_poll_ms": config.alerts_poll_ms,
        "sessions_poll_ms": config.sessions_poll_ms,
    }
    bad = [f"{name}={value}" for name, value in checks.items() if value <= 0]
    if bad:
        raise ValueError("operator console poll intervals must be > 0; got " + ", ".join(bad))
    if config.api_timeout_seconds <= 0:
        raise ValueError(f"api_timeout_seconds must be > 0; got {config.api_timeout_seconds}")
