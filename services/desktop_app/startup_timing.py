from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, MutableMapping

STARTUP_EPOCH_ENV = "LSIE_STARTUP_EPOCH_PERF_COUNTER_NS"


def ensure_startup_epoch(
    env: MutableMapping[str, str] | None = None,
    *,
    now_ns: int | None = None,
) -> int:
    target = os.environ if env is None else env
    existing = _startup_epoch_ns(target)
    if existing is not None:
        return existing
    epoch_ns = time.perf_counter_ns() if now_ns is None else now_ns
    target[STARTUP_EPOCH_ENV] = str(epoch_ns)
    return epoch_ns


def format_startup_milestone(
    milestone: str,
    *,
    environ: Mapping[str, str] | None = None,
    now_ns: int | None = None,
) -> str:
    elapsed_ms = startup_elapsed_ms(environ=environ, now_ns=now_ns)
    if elapsed_ms is None:
        return f"startup milestone={milestone}"
    return f"startup milestone={milestone} elapsed_ms={elapsed_ms:.1f}"


def log_startup_milestone(
    milestone: str,
    *,
    logger: logging.Logger,
    environ: Mapping[str, str] | None = None,
    now_ns: int | None = None,
) -> None:
    message = format_startup_milestone(
        milestone,
        environ=environ,
        now_ns=now_ns,
    )
    logger.info(message)
    if not logger.isEnabledFor(logging.INFO):
        print(message, flush=True)


def startup_elapsed_ms(
    *,
    environ: Mapping[str, str] | None = None,
    now_ns: int | None = None,
) -> float | None:
    source = os.environ if environ is None else environ
    epoch_ns = _startup_epoch_ns(source)
    if epoch_ns is None:
        return None
    current_ns = time.perf_counter_ns() if now_ns is None else now_ns
    elapsed_ns = max(0, current_ns - epoch_ns)
    return elapsed_ns / 1_000_000.0


def _startup_epoch_ns(environ: Mapping[str, str]) -> int | None:
    raw = str(environ.get(STARTUP_EPOCH_ENV, "")).strip()
    if not raw:
        return None
    try:
        epoch_ns = int(raw)
    except ValueError:
        return None
    if epoch_ns < 0:
        return None
    return epoch_ns
