"""Suppress crash dialogs and Windows Error Reporting LocalDumps (§5.1.7).

A modal crash dialog ("application has stopped working") prevents the
parent ProcessGraph from observing the child exit code in time, leaving
a stale entry in ``capture_pid_manifest`` for the next recovery sweep
to clean up. Worse, Windows Error Reporting's LocalDumps feature, when
enabled globally, will write a ``.dmp`` of the crashing process under
``%LOCALAPPDATA%\\CrashDumps\\``. The §5.1.7 volatile-memory controls
require child-process crash dumps to be disabled or redirected to
scrubbed diagnostics, because a crash mid-segment would otherwise freeze
raw PCM bytes outside the governed cleanup path.

The two functions below close both windows. They MUST be called early
in every desktop-app process (the parent + each child); calling them
once in the parent is not enough because crash-dialog suppression is
per-process state on Windows.

POSIX is a no-op for both — Linux and macOS do not raise modal dialogs
on segfault and have their own coredump policies that the operator
controls outside the app.
"""

from __future__ import annotations

import logging

from services.desktop_app.os_adapter import (
    register_localdumps_exclusion,
    suppress_crash_dialogs,
)

logger = logging.getLogger(__name__)

DEFAULT_APP_BINARY: str = "lsie-mlf-desktop.exe"


def install_crash_privacy_guards(app_name: str = DEFAULT_APP_BINARY) -> None:
    """Suppress crash dialogs (always) and exclude the app from LocalDumps (Windows).

    Idempotent. Call from the top of every process's ``run()`` and from
    :mod:`services.desktop_app.__main__` ahead of ``ProcessGraph.start_all``.
    The two guards are independent — ``suppress_crash_dialogs`` is
    per-process state, ``register_localdumps_exclusion`` is a one-off
    HKCU registry write.
    """
    suppress_crash_dialogs()
    if register_localdumps_exclusion(app_name):
        logger.info("WER LocalDumps exclusion registered for %s", app_name)
