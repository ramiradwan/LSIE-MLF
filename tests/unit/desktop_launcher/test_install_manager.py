"""WS1 P2 — install manager tests."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Self

import httpx
import pytest

from services.desktop_launcher import install_manager
from services.desktop_launcher.install_manager import DownloadAsset, LauncherPaths


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
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

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
    ) -> FakeProcess:
        calls.append((cmd, cwd, env))
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    install_manager.run_uv_sync(
        repo_root=tmp_path,
        staging_dir=tmp_path / "runtime.staging",
        python_exe=tmp_path / "python.exe",
        log=lambda _line: None,
    )

    cmd, cwd, env = calls[0]
    assert cwd == tmp_path
    assert cmd[:5] == ["uv", "sync", "--frozen", "--extra", "ml_backend"]
    assert "--python" in cmd
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "runtime.staging" / ".venv")


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
