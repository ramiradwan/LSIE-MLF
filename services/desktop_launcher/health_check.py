"""Post-hydration runtime smoke checks and application handoff."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from services.desktop_app import os_adapter
from services.desktop_launcher import manifest, preflight

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
        creationflags=subprocess.CREATE_NO_WINDOW,
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
    _repair_venv_home(active_runtime_dir)
    manifest.write_manifest(
        active_runtime_dir,
        manifest.build_manifest(
            python_runtime=python_runtime,
            scrcpy_version=scrcpy_version,
        ),
    )
    return active_runtime_dir


def launch_desktop_app(runtime_dir: Path, app_root: Path | None = None) -> subprocess.Popen[str]:
    preflight.ensure_preflight()
    python_exe = runtime_python(runtime_dir)
    root = (app_root or _default_app_root()).resolve()
    _validate_app_root(root)
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not pythonpath else f"{root}{os.pathsep}{pythonpath}"
    log_path = _launch_log_path()
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n--- launching LSIE-MLF from {root} with {python_exe} ---\n")
    log_handle.flush()
    try:
        return subprocess.Popen(
            [str(python_exe), "-m", "services.desktop_app"],
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        log_handle.close()
        raise


def _default_app_root() -> Path:
    from services.desktop_launcher.install_manager import resolve_app_root

    return resolve_app_root()


def _repair_venv_home(runtime_dir: Path) -> None:
    pyvenv_cfg = runtime_dir / ".venv" / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        return
    python_home = runtime_dir / "python" / "python"
    lines = pyvenv_cfg.read_text(encoding="utf-8").splitlines()
    repaired = [f"home = {python_home}" if line.startswith("home = ") else line for line in lines]
    pyvenv_cfg.write_text("\n".join(repaired) + "\n", encoding="utf-8")


def _validate_app_root(root: Path) -> None:
    if not (root / "services" / "desktop_app" / "__main__.py").is_file():
        raise RuntimeError(
            "LSIE-MLF app source root was not found. Set LSIE_APP_ROOT to the repository root "
            "before launching the packaged installer."
        )


def _launch_log_path() -> Path:
    log_dir = os_adapter.resolve_state_dir().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "desktop-launch.log"


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)
