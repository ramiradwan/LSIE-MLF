"""Desktop launcher / installer surface for the v4.0 build.

The launcher is the small, signed binary an end-user double-clicks.
Its responsibilities, listed in roughly the order they execute:

* :mod:`preflight` — validate the host (Python version, NVIDIA GPU
  compute capability, developer-marker semantics) before any heavy
  Python wheel is loaded.
* :mod:`install_manager` — download and hash-verify the
  ``python-build-standalone`` runtime, hydrate the ML virtual
  environment via ``uv sync --frozen``, and atomically promote the
  staged install into ``%LOCALAPPDATA%\\LSIE-MLF\\runtime\\``.
* :mod:`repair` — rebuild a corrupted runtime in place while
  preserving the operator's local SQLite analytics state.

This package is intentionally separate from the runtime
``services/desktop_app/`` package: the launcher must run in a
minimal Python interpreter (``python-build-standalone`` shipped with
the installer) and must not transitively import any ML wheel or
``services.desktop_app.processes.gpu_ml_worker`` symbol.
"""

from __future__ import annotations
