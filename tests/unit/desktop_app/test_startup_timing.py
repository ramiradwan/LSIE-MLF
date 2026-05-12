from __future__ import annotations

import logging

import pytest

from services.desktop_app.startup_timing import (
    STARTUP_EPOCH_ENV,
    ensure_startup_epoch,
    format_startup_milestone,
    log_startup_milestone,
)


def test_format_startup_milestone_includes_elapsed_when_epoch_exists() -> None:
    env = {STARTUP_EPOCH_ENV: "1000000000"}

    message = format_startup_milestone(
        "api_ready",
        environ=env,
        now_ns=1_250_000_000,
    )

    assert message == "startup milestone=api_ready elapsed_ms=250.0"


def test_ensure_startup_epoch_preserves_existing_value() -> None:
    env = {STARTUP_EPOCH_ENV: "123"}

    assert ensure_startup_epoch(env, now_ns=456) == 123
    assert env[STARTUP_EPOCH_ENV] == "123"


def test_log_startup_milestone_prints_when_info_logging_is_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("tests.startup_timing.disabled")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    log_startup_milestone(
        "capture_ready",
        logger=logger,
        environ={STARTUP_EPOCH_ENV: "1000000000"},
        now_ns=1_500_000_000,
    )

    captured = capsys.readouterr()
    assert captured.out == "startup milestone=capture_ready elapsed_ms=500.0\n"
    assert captured.err == ""


def test_log_startup_milestone_does_not_print_when_info_logging_is_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("tests.startup_timing.enabled")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    log_startup_milestone(
        "api_ready",
        logger=logger,
        environ={STARTUP_EPOCH_ENV: "1000000000"},
        now_ns=1_500_000_000,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
