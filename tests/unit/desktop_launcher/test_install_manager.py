"""Install manager tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Self, cast

import httpx
import pytest

from services.desktop_app import os_adapter
from services.desktop_launcher import health_check, install_manager, manifest, preflight
from services.desktop_launcher.install_manager import DownloadAsset, LauncherPaths


def _assert_subprocess_policy(kwargs: dict[str, object]) -> None:
    if sys.platform == "win32":
        creationflags = cast(int, kwargs["creationflags"])
        assert creationflags & subprocess.CREATE_NO_WINDOW
        assert creationflags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert "creationflags" not in kwargs


class _FakeStream:
    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None) -> None:
        self._chunks = chunks
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self) -> list[bytes]:
        return self._chunks


class _FakeClient:
    calls: list[tuple[str, str, dict[str, str]]]
    chunks: list[bytes]
    headers: dict[str, str]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls = _FakeClient.calls

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def stream(self, method: str, url: str, *, headers: dict[str, str]) -> _FakeStream:
        self.calls.append((method, url, headers))
        return _FakeStream(_FakeClient.chunks, _FakeClient.headers)


@pytest.fixture(autouse=True)
def reset_fake_client() -> None:
    _FakeClient.calls = []
    _FakeClient.chunks = [b" world"]
    _FakeClient.headers = {"Content-Range": "bytes 5-10/11"}


def test_download_with_resume_uses_range_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    target = tmp_path / "payload.zip"
    target.with_suffix(".zip.part").write_bytes(b"hello")
    progress: list[int] = []

    result = install_manager.download_with_resume(
        DownloadAsset("payload", "https://example.test/payload.zip", "payload.zip", "unused"),
        target,
        progress=progress.append,
    )

    assert result == target
    assert target.read_bytes() == b"hello world"
    assert _FakeClient.calls == [("GET", "https://example.test/payload.zip", {"Range": "bytes=5-"})]
    assert progress[-1] == 100


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        install_manager.verify_sha256(payload, "0" * 64)


def test_is_valid_cached_asset_accepts_matching_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()

    assert install_manager.is_valid_cached_asset(payload, digest) is True


def test_is_valid_cached_asset_rejects_missing_or_mismatched_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_text("payload", encoding="utf-8")

    assert install_manager.is_valid_cached_asset(payload, "0" * 64) is False
    assert install_manager.is_valid_cached_asset(tmp_path / "missing.bin", "0" * 64) is False


def test_extract_zip_archive(tmp_path: Path) -> None:
    archive = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tool/scrcpy.exe", "binary")

    install_manager.extract_archive(archive, tmp_path / "out")

    assert (tmp_path / "out" / "tool" / "scrcpy.exe").read_text(encoding="utf-8") == "binary"


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="escapes destination"):
        install_manager.extract_archive(archive, tmp_path / "out")


def test_run_uv_sync_uses_ml_backend_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str], dict[str, object]]] = []

    class FakeStdout:
        def __iter__(self) -> Self:
            return self

        def __next__(self) -> str:
            raise StopIteration

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    def fake_popen(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout: int,
        stderr: int,
        text: bool,
        **kwargs: object,
    ) -> FakeProcess:
        del stdout, stderr, text
        calls.append((cmd, cwd, env, kwargs))
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    install_manager.run_uv_sync(
        repo_root=tmp_path,
        staging_dir=tmp_path / "runtime.staging",
        python_exe=tmp_path / "python.exe",
        log=lambda _line: None,
    )

    cmd, cwd, env, kwargs = calls[0]
    assert cwd == tmp_path
    assert cmd[:5] == ["uv", "sync", "--frozen", "--extra", "ml_backend"]
    assert "--reinstall" not in cmd
    assert "--python" in cmd
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "runtime.staging" / ".venv")
    _assert_subprocess_policy(kwargs)


def test_run_uv_sync_can_force_reinstall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    class FakeStdout:
        def __iter__(self) -> Self:
            return self

        def __next__(self) -> str:
            raise StopIteration

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    def fake_popen(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout: int,
        stderr: int,
        text: bool,
        **kwargs: object,
    ) -> FakeProcess:
        del cwd, env, stdout, stderr, text
        _assert_subprocess_policy(kwargs)
        calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    install_manager.run_uv_sync(
        repo_root=tmp_path,
        staging_dir=tmp_path / "runtime",
        python_exe=tmp_path / "python.exe",
        log=lambda _line: None,
        reinstall=True,
    )

    assert "--reinstall" in calls[0]


def test_resolve_app_root_prefers_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "repo"
    (app_root / "services" / "desktop_app").mkdir(parents=True)
    (app_root / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("LSIE_APP_ROOT", str(app_root))

    assert install_manager.resolve_app_root() == app_root


def test_resolve_app_root_prefers_bundled_app_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    (app_root / "services" / "desktop_app").mkdir(parents=True)
    (app_root / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("LSIE_APP_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert install_manager.resolve_app_root() == app_root


def test_resolve_app_root_finds_cwd_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "services" / "desktop_app").mkdir(parents=True)
    (tmp_path / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("LSIE_APP_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert install_manager.resolve_app_root() == tmp_path


def test_has_current_runtime_accepts_matching_manifest_and_python(
    tmp_path: Path,
) -> None:
    python_exe = (
        tmp_path
        / ".venv"
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python3")
    )
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")
    manifest.write_manifest(
        tmp_path,
        manifest.build_manifest(
            python_runtime=install_manager.PYTHON_STANDALONE_VERSION,
            scrcpy_version=install_manager.SCRCPY_VERSION,
        ),
    )

    assert install_manager.has_current_runtime(tmp_path) is True


def test_has_current_runtime_rejects_missing_runtime(tmp_path: Path) -> None:
    assert install_manager.has_current_runtime(tmp_path) is False


def test_create_app_shortcut_uses_launcher_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = LauncherPaths(
        base_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
        staging_dir=tmp_path / "runtime.staging",
        active_runtime_dir=tmp_path / "runtime",
        repo_root=tmp_path / "repo",
    )
    calls: list[tuple[Path, Path, Path, str]] = []

    def fake_create_shortcut(
        *,
        target: Path,
        shortcut: Path,
        working_dir: Path,
        description: str,
    ) -> bool:
        calls.append((target, shortcut, working_dir, description))
        return True

    monkeypatch.setattr(os_adapter, "create_shortcut", fake_create_shortcut)

    assert (
        install_manager.create_app_shortcut(paths, launcher_exe=tmp_path / "launcher.exe") is True
    )
    assert calls == [
        (
            tmp_path / "launcher.exe",
            tmp_path / "LSIE-MLF.lnk",
            tmp_path / "repo",
            "Launch LSIE-MLF",
        )
    ]


def test_install_runtime_reuses_valid_cached_archives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = LauncherPaths(
        base_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
        staging_dir=tmp_path / "runtime.staging",
        active_runtime_dir=tmp_path / "runtime",
        repo_root=tmp_path,
    )
    paths.downloads_dir.mkdir()
    asset_payloads = (b"python", b"scrcpy")
    assets = (
        DownloadAsset(
            "python-build-standalone",
            "https://example.test/python.tar.gz",
            "python.tar.gz",
            hashlib.sha256(asset_payloads[0]).hexdigest(),
        ),
        DownloadAsset(
            "scrcpy",
            "https://example.test/scrcpy.zip",
            "scrcpy.zip",
            hashlib.sha256(asset_payloads[1]).hexdigest(),
        ),
    )
    for asset, payload in zip(assets, asset_payloads, strict=True):
        (paths.downloads_dir / asset.filename).write_bytes(payload)
    statuses: list[str] = []
    download_calls: list[str] = []

    monkeypatch.setattr(preflight, "ensure_preflight", lambda: None)
    monkeypatch.setattr(
        install_manager,
        "download_with_resume",
        lambda asset, *_args, **_kwargs: download_calls.append(asset.name),
    )
    monkeypatch.setattr(install_manager, "extract_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        install_manager,
        "find_runtime_python",
        lambda _root: tmp_path / "python.exe",
    )
    monkeypatch.setattr(install_manager, "run_uv_sync", lambda **_kwargs: None)
    monkeypatch.setattr(health_check, "run_runtime_smoke_test", lambda _root: "")
    monkeypatch.setattr(
        health_check,
        "finalize_install",
        lambda **_kwargs: paths.active_runtime_dir,
    )

    install_manager.install_runtime(
        paths=paths,
        assets=assets,
        status=statuses.append,
        progress=lambda _value: None,
        log=lambda _line: None,
    )

    assert download_calls == []
    assert "Using cached python-build-standalone" in statuses
    assert "Using cached scrcpy" in statuses


def test_install_runtime_runs_preflight_before_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = LauncherPaths(
        base_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
        staging_dir=tmp_path / "runtime.staging",
        active_runtime_dir=tmp_path / "runtime",
        repo_root=tmp_path,
    )
    asset = DownloadAsset("payload", "https://example.test/payload", "payload.bin", "0" * 64)
    statuses: list[str] = []
    preflight_calls: list[str] = []
    progress_values: list[int] = []

    monkeypatch.setattr(
        preflight,
        "ensure_preflight",
        lambda: preflight_calls.append("called"),
    )
    monkeypatch.setattr(
        install_manager,
        "download_with_resume",
        lambda *_args, **_kwargs: paths.downloads_dir / asset.filename,
    )
    monkeypatch.setattr(install_manager, "verify_sha256", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(install_manager, "extract_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        install_manager,
        "find_runtime_python",
        lambda _root: tmp_path / "python.exe",
    )
    monkeypatch.setattr(install_manager, "run_uv_sync", lambda **_kwargs: None)
    monkeypatch.setattr(health_check, "run_runtime_smoke_test", lambda _root: "")
    monkeypatch.setattr(
        health_check,
        "finalize_install",
        lambda **_kwargs: paths.active_runtime_dir,
    )

    install_manager.install_runtime(
        paths=paths,
        assets=(asset, asset),
        status=statuses.append,
        progress=progress_values.append,
        log=lambda _line: None,
    )

    assert preflight_calls == ["called"]
    assert statuses[0] == "Running hardware preflight"


def test_install_manager_starts_background_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = LauncherPaths(
        base_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
        staging_dir=tmp_path / "runtime.staging",
        active_runtime_dir=tmp_path / "runtime",
        repo_root=tmp_path,
    )
    asset = DownloadAsset("payload", "https://example.test/payload", "payload.bin", "0" * 64)
    manager = install_manager.InstallManager(paths=paths, assets=(asset, asset))
    monkeypatch.setattr(
        install_manager,
        "install_runtime",
        lambda **_kwargs: tmp_path / "runtime",
    )

    thread = manager.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
