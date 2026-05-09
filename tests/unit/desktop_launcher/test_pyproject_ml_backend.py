"""WS1 P2 — progressive hydration dependency metadata tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

_HEAVY_ML_PACKAGES = {"torch", "ctranslate2", "faster-whisper", "mediapipe"}


def test_heavy_ml_dependencies_live_in_ml_backend_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = set(pyproject["project"]["dependencies"])
    ml_backend = set(pyproject["project"]["optional-dependencies"]["ml_backend"])

    assert not any(_starts_with_package(dep, _HEAVY_ML_PACKAGES) for dep in dependencies)
    assert all(
        any(_starts_with_package(dep, {pkg}) for dep in ml_backend) for pkg in _HEAVY_ML_PACKAGES
    )
    assert any(_starts_with_package(dep, {"pyside6"}) for dep in dependencies)
    assert any(_starts_with_package(dep, {"fastapi"}) for dep in dependencies)
    assert any(_starts_with_package(dep, {"uvicorn"}) for dep in dependencies)
    assert any(_starts_with_package(dep, {"httpx"}) for dep in dependencies)


def test_uv_lock_exposes_ml_backend_extra() -> None:
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    project = next(pkg for pkg in lock["package"] if pkg["name"] == "lsie-mlf-desktop")

    optional = project["optional-dependencies"]["ml-backend"]
    names = {entry["name"] for entry in optional}

    assert names >= _HEAVY_ML_PACKAGES


def _starts_with_package(requirement: str, names: set[str]) -> bool:
    normalized = requirement.lower().replace("_", "-")
    return any(normalized.startswith(name) for name in names)
