"""WS1 P2 — install manifest tests."""

from __future__ import annotations

from pathlib import Path

from services.desktop_launcher import manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    record = manifest.build_manifest(
        python_runtime="cpython-3.11.15+20260414",
        scrcpy_version="v3.3.4",
    )

    target = manifest.write_manifest(tmp_path, record)
    loaded = manifest.read_manifest(tmp_path)

    assert target.name == manifest.MANIFEST_FILENAME
    assert loaded == record
    assert loaded.uv_sync_extra == "ml_backend"
