# PyInstaller one-dir build spec for the LSIE-MLF Operator Console.
#
# Builds to `dist/operator_console/` as a one-dir app (not one-file) so
# start-up stays fast and so the PySide6 Qt plugins ship as loose files
# the OS can mmap without a temp-dir extraction pass.
#
# Build with:
#
#   pyinstaller build/operator_console.spec --clean --noconfirm
#
# The entrypoint is `services.operator_console.__main__`, which is what
# `python -m services.operator_console` runs — keeping the one launch
# path regardless of whether the operator is on a dev checkout or on a
# packaged install.
#
# Spec references:
#   §4.E.1         — operator-facing console is a host app, not a container
#   SPEC-AMEND-008 — PySide6 Operator Console; the operator host must not
#                    pull ML-worker dependencies (CUDA, cuDNN, faster-whisper,
#                    mediapipe, Praat, spaCy). These are explicitly excluded
#                    below so a stray transitive import doesn't re-introduce
#                    the API/worker coupling the amendment removed.
# ruff: noqa

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ  # type: ignore[import-not-found]
from PyInstaller.building.build_main import Analysis  # type: ignore[import-not-found]

_PROJECT_ROOT = Path(os.environ.get("LSIE_PROJECT_ROOT", Path.cwd())).resolve()

# Entry point. Running `python -m services.operator_console` resolves to
# `services/operator_console/__main__.py` which just calls `app.main()`.
_ENTRY = _PROJECT_ROOT / "services" / "operator_console" / "__main__.py"

# Hidden imports — modules PyInstaller's static analysis cannot see
# because they are referenced by string at runtime (Qt plugins, Pydantic
# discriminated unions). Keep the list narrow; broad includes undo the
# size savings of excluding the ML stack.
_HIDDEN_IMPORTS = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # Pydantic v2 uses a Rust core that PyInstaller picks up normally,
    # but the validators module is imported lazily by enums.
    "pydantic",
    "pydantic.validators",
]

# Explicitly excluded — these belong to the ML Worker / API containers
# and must never land in the operator-host bundle. Listed verbatim so a
# future reader can see the intent, not just the flag.
_EXCLUDES = [
    "torch",
    "ctranslate2",
    "faster_whisper",
    "mediapipe",
    "parselmouth",
    "spacy",
    "celery",
    "redis",
    "psycopg2",
    "psycopg2-binary",
    "fastapi",
    "uvicorn",
    "patchright",
    "TikTokLive",
]

a = Analysis(
    [str(_ENTRY)],
    pathex=[str(_PROJECT_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=_HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="operator_console",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="operator_console",
)
