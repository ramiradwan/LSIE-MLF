"""First-run installer manager for progressive desktop runtime hydration."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from PySide6.QtCore import QObject, Signal

from services.desktop_app import os_adapter
from services.desktop_launcher import health_check

PYTHON_STANDALONE_VERSION = "cpython-3.11.15+20260414"
SCRCPY_VERSION = "v3.3.4"
PYTHON_STANDALONE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260414/"
    "cpython-3.11.15%2B20260414-x86_64-pc-windows-msvc-install_only.tar.gz"
)
SCRCPY_WIN64_URL = (
    "https://github.com/Genymobile/scrcpy/releases/download/v3.3.4/scrcpy-win64-v3.3.4.zip"
)
PYTHON_STANDALONE_SHA256 = "8e69ecf1d9fc194e029aafa608d483bf24ccaa8f56d456d7009f20462d62ad23"
SCRCPY_WIN64_SHA256 = "d8a155b7c180b7ca4cdadd40712b8750b63f3aab48cb5b8a2a39ac2d0d4c5d38"

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class DownloadAsset:
    name: str
    url: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class LauncherPaths:
    base_dir: Path
    downloads_dir: Path
    staging_dir: Path
    active_runtime_dir: Path
    repo_root: Path


class InstallerSignals(QObject):
    status_changed = Signal(str)
    progress_changed = Signal(int)
    log_line = Signal(str)
    finished = Signal(Path)
    failed = Signal(str)


class InstallManager(QObject):
    def __init__(
        self,
        *,
        paths: LauncherPaths | None = None,
        assets: tuple[DownloadAsset, DownloadAsset] | None = None,
        signals: InstallerSignals | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths or default_launcher_paths()
        self.assets = assets or default_assets()
        self.signals = signals or InstallerSignals()
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(
            target=self._run,
            name="lsie-first-run-installer",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def _run(self) -> None:
        try:
            runtime_dir = install_runtime(
                paths=self.paths,
                assets=self.assets,
                status=self.signals.status_changed.emit,
                progress=self.signals.progress_changed.emit,
                log=self.signals.log_line.emit,
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(runtime_dir)


def default_launcher_paths() -> LauncherPaths:
    base_dir = os_adapter.resolve_state_dir().parent
    return LauncherPaths(
        base_dir=base_dir,
        downloads_dir=base_dir / "downloads",
        staging_dir=base_dir / "runtime.staging",
        active_runtime_dir=base_dir / "runtime",
        repo_root=Path(__file__).resolve().parents[2],
    )


def default_assets() -> tuple[DownloadAsset, DownloadAsset]:
    return (
        DownloadAsset(
            name="python-build-standalone",
            url=PYTHON_STANDALONE_URL,
            filename="python-build-standalone.tar.gz",
            sha256=PYTHON_STANDALONE_SHA256,
        ),
        DownloadAsset(
            name="scrcpy",
            url=SCRCPY_WIN64_URL,
            filename="scrcpy-win64-v3.3.4.zip",
            sha256=SCRCPY_WIN64_SHA256,
        ),
    )


def install_runtime(
    *,
    paths: LauncherPaths,
    assets: tuple[DownloadAsset, DownloadAsset],
    status: StatusCallback,
    progress: ProgressCallback,
    log: LogCallback,
) -> Path:
    paths.downloads_dir.mkdir(parents=True, exist_ok=True)
    _reset_staging(paths.staging_dir)
    progress(0)

    downloaded: list[Path] = []
    for index, asset in enumerate(assets):
        status(f"Downloading {asset.name}")
        target = paths.downloads_dir / asset.filename
        download_with_resume(
            asset,
            target,
            progress=_scaled_progress(progress, index, len(assets), 40),
        )
        verify_sha256(target, asset.sha256)
        downloaded.append(target)

    python_archive, scrcpy_archive = downloaded
    status("Extracting Python runtime")
    extract_archive(python_archive, paths.staging_dir / "python")
    progress(45)

    status("Extracting scrcpy")
    extract_archive(scrcpy_archive, paths.staging_dir / "tools" / "scrcpy")
    progress(55)

    status("Hydrating ML backend")
    python_exe = find_runtime_python(paths.staging_dir / "python")
    run_uv_sync(
        repo_root=paths.repo_root,
        staging_dir=paths.staging_dir,
        python_exe=python_exe,
        log=log,
    )
    progress(85)

    status("Running runtime health check")
    smoke_output = health_check.run_runtime_smoke_test(paths.staging_dir)
    if smoke_output:
        log(smoke_output)
    progress(95)

    status("Finalizing installation")
    runtime_dir = health_check.finalize_install(
        staging_dir=paths.staging_dir,
        active_runtime_dir=paths.active_runtime_dir,
        python_runtime=PYTHON_STANDALONE_VERSION,
        scrcpy_version=SCRCPY_VERSION,
    )
    progress(100)
    status("Setup complete")
    return runtime_dir


def download_with_resume(
    asset: DownloadAsset,
    target: Path,
    *,
    progress: ProgressCallback,
) -> Path:
    partial = target.with_suffix(f"{target.suffix}.part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    with (
        httpx.Client(follow_redirects=True, timeout=None) as client,
        client.stream("GET", asset.url, headers=headers) as response,
    ):
        response.raise_for_status()
        total = _total_download_size(response, existing)
        downloaded = existing
        with partial.open(mode) as handle:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress(min(100, int(downloaded * 100 / total)))
    partial.replace(target)
    progress(100)
    return target


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"{path.name} sha256 mismatch: expected {expected}, got {actual}")


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for zip_member in zf.infolist():
                _ensure_archive_member_safe(destination, zip_member.filename)
            zf.extractall(destination)
        return
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            for tar_member in tf.getmembers():
                _ensure_archive_member_safe(destination, tar_member.name)
            tf.extractall(destination)
        return
    raise ValueError(f"unsupported archive format: {archive}")


def find_runtime_python(root: Path) -> Path:
    names = ("python.exe",) if sys.platform == "win32" else ("python3", "python")
    for name in names:
        for candidate in root.rglob(name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"could not locate Python executable under {root}")


def run_uv_sync(
    *,
    repo_root: Path,
    staging_dir: Path,
    python_exe: Path,
    log: LogCallback,
) -> None:
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(staging_dir / ".venv")
    cmd = ["uv", "sync", "--frozen", "--extra", "ml_backend", "--python", str(python_exe)]
    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"uv sync failed with exit code {returncode}")


def _reset_staging(staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)


def _total_download_size(response: httpx.Response, existing: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        _, total = content_range.rsplit("/", 1)
        if total.isdigit():
            return int(total)
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return existing + int(content_length)
    return None


def _ensure_archive_member_safe(destination: Path, member_name: str) -> None:
    target = (destination / member_name).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"archive member escapes destination: {member_name}")


def _scaled_progress(
    progress: ProgressCallback,
    index: int,
    total_items: int,
    span: int,
) -> ProgressCallback:
    start = int(index * span / total_items)
    width = int(span / total_items)

    def emit(value: int) -> None:
        progress(start + int(width * value / 100))

    return emit
