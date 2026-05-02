"""Post-hydration runtime smoke checks and application handoff."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from services.desktop_launcher import manifest

_HEALTH_CHECK_SNIPPET = (
    "import torch; "
    "import mediapipe; "
    "import ctranslate2; "
    "import faster_whisper; "
    "print('lsie-mlf runtime smoke ok')"
)


class RuntimeHealthCheckError(RuntimeError):
    """Raised when the staged ML runtime cannot import required backends."""


def runtime_python(runtime_dir: Path) -> Path:
    if sys.platform == "win32":
        candidates = (
            runtime_dir / ".venv" / "Scripts" / "python.exe",
            runtime_dir / "python" / "python.exe",
        )
    else:
        candidates = (
            runtime_dir / ".venv" / "bin" / "python3",
            runtime_dir / ".venv" / "bin" / "python",
            runtime_dir / "python" / "bin" / "python3",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def run_runtime_smoke_test(runtime_dir: Path, timeout_s: float = 120.0) -> str:
    python_exe = runtime_python(runtime_dir)
    result = subprocess.run(
        [str(python_exe), "-c", _HEALTH_CHECK_SNIPPET],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeHealthCheckError(output or f"runtime smoke failed with {result.returncode}")
    return output


def finalize_install(
    *,
    staging_dir: Path,
    active_runtime_dir: Path,
    python_runtime: str,
    scrcpy_version: str,
) -> Path:
    if active_runtime_dir.exists():
        backup = active_runtime_dir.with_name(f"{active_runtime_dir.name}.previous")
        if backup.exists():
            _remove_tree(backup)
        active_runtime_dir.replace(backup)
    staging_dir.replace(active_runtime_dir)
    manifest.write_manifest(
        active_runtime_dir,
        manifest.build_manifest(
            python_runtime=python_runtime,
            scrcpy_version=scrcpy_version,
        ),
    )
    return active_runtime_dir


def launch_desktop_app(runtime_dir: Path) -> subprocess.Popen[str]:
    python_exe = runtime_python(runtime_dir)
    return subprocess.Popen(
        [str(python_exe), "-m", "services.desktop_app"],
        cwd=Path.cwd(),
        text=True,
    )


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)
