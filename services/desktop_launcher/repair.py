"""Runtime repair path for the desktop launcher."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from services.desktop_app import os_adapter
from services.desktop_launcher import health_check, install_manager

SQLITE_FILENAME = "desktop.sqlite"

StatusCallback = Callable[[str], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class RepairResult:
    runtime_dir: Path
    sqlite_path: Path
    preserved_tables: tuple[str, ...]


class RepairError(RuntimeError):
    """Raised when the local runtime repair cannot complete."""


def default_runtime_dir() -> Path:
    return os_adapter.resolve_state_dir().parent / "runtime"


def default_sqlite_path() -> Path:
    return os_adapter.resolve_state_dir() / SQLITE_FILENAME


def repair_runtime(
    *,
    runtime_dir: Path | None = None,
    repo_root: Path | None = None,
    sqlite_path: Path | None = None,
    status: StatusCallback | None = None,
    log: LogCallback | None = None,
) -> RepairResult:
    target_runtime = runtime_dir or default_runtime_dir()
    target_sqlite = sqlite_path or default_sqlite_path()
    root = repo_root or Path(__file__).resolve().parents[2]
    emit_status = status or _ignore
    emit_log = log or _ignore

    emit_status("Repairing desktop runtime")
    python_dir = target_runtime / "python"
    python_exe = install_manager.find_runtime_python(python_dir)

    emit_status("Rebuilding ML backend")
    install_manager.run_uv_sync(
        repo_root=root,
        staging_dir=target_runtime,
        python_exe=python_exe,
        log=emit_log,
        reinstall=True,
    )

    emit_status("Running runtime health check")
    smoke_output = health_check.run_runtime_smoke_test(target_runtime)
    if smoke_output:
        emit_log(smoke_output)

    _remove_staging_directory(target_runtime)
    emit_status("Repair complete")
    return RepairResult(
        runtime_dir=target_runtime,
        sqlite_path=target_sqlite,
        preserved_tables=("attribution_event", "metrics", "physiology_log"),
    )


def _remove_staging_directory(runtime_dir: Path) -> None:
    staging = runtime_dir.with_name("runtime.staging")
    if staging.exists():
        shutil.rmtree(staging)


def _ignore(_message: str) -> None:
    return None
