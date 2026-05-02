"""Install manifest helpers for the desktop launcher runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_FILENAME = "install_manifest.json"


@dataclass(frozen=True)
class InstallManifest:
    version: str
    installed_at_utc: str
    python_runtime: str
    scrcpy_version: str
    uv_sync_extra: str


def build_manifest(
    *,
    python_runtime: str,
    scrcpy_version: str,
    uv_sync_extra: str = "ml_backend",
) -> InstallManifest:
    return InstallManifest(
        version="4.0.0",
        installed_at_utc=datetime.now(tz=UTC).isoformat(),
        python_runtime=python_runtime,
        scrcpy_version=scrcpy_version,
        uv_sync_extra=uv_sync_extra,
    )


def write_manifest(runtime_dir: Path, manifest: InstallManifest) -> Path:
    target = runtime_dir / MANIFEST_FILENAME
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_manifest(runtime_dir: Path) -> InstallManifest:
    raw = json.loads((runtime_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return InstallManifest(**raw)
