"""WS3 P1 canary — ML import isolation across the desktop process graph.

The v4.0 desktop topology pivots away from a single Docker image that
loads every ML library into one process. The new contract is:
``torch`` / ``mediapipe`` / ``faster_whisper`` / ``ctranslate2`` are
imported only inside ``services.desktop_app.processes.gpu_ml_worker``,
and only after the spawned child re-imports it through
``services.desktop_app.process_graph._launch``.

If a future PR adds an ML import to (or transitively pulls one into)
the parent process or any non-ML process module, these tests fail
immediately, before the cuDNN-collision and spawn-deadlock symptoms
documented in the v4.0 implementation plan can surface in CI or in
production.

The tests run each import in a clean subprocess so the test runner's
own ``sys.modules`` does not pollute the result.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

ML_LIB_ROOTS: tuple[str, ...] = ("torch", "mediapipe", "faster_whisper", "ctranslate2")

NON_ML_PROCESS_MODULES: tuple[str, ...] = (
    "services.desktop_app.processes.ui_api_shell",
    "services.desktop_app.processes.capture_supervisor",
    "services.desktop_app.processes.module_c_orchestrator",
    "services.desktop_app.processes.analytics_state_worker",
    "services.desktop_app.processes.cloud_sync_worker",
)


def _imported_ml_roots(target_import: str) -> list[str]:
    """Spawn a clean Python, import ``target_import``, return ML roots in sys.modules."""
    code = (
        "import sys, json\n"
        f"import {target_import}\n"
        f"roots = {list(ML_LIB_ROOTS)!r}\n"
        "found = sorted({k.split('.')[0] for k in sys.modules if k.split('.')[0] in roots})\n"
        "print(json.dumps(found))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return list(json.loads(proc.stdout.strip()))


def test_process_graph_parent_module_is_clean() -> None:
    """Importing process_graph itself must not pull torch et al. into the parent."""
    assert _imported_ml_roots("services.desktop_app.process_graph") == []


def test_main_entry_module_is_clean() -> None:
    """Importing the __main__ module (without running it) must stay clean."""
    assert _imported_ml_roots("services.desktop_app.__main__") == []


@pytest.mark.parametrize("module_path", NON_ML_PROCESS_MODULES)
def test_non_ml_process_module_is_clean(module_path: str) -> None:
    """Each non-ML process module must not import any ML library, even transitively."""
    assert _imported_ml_roots(module_path) == []


def test_process_modules_table_is_complete() -> None:
    """The PROCESS_MODULES dict must list exactly the six v4.0 §9 processes."""
    from services.desktop_app.process_graph import PROCESS_MODULES

    assert set(PROCESS_MODULES) == {
        "ui_api_shell",
        "capture_supervisor",
        "module_c_orchestrator",
        "gpu_ml_worker",
        "analytics_state_worker",
        "cloud_sync_worker",
    }


def test_process_graph_starts_and_stops_cleanly() -> None:
    """Round-trip the full six-process graph: spawn, signal shutdown, join."""
    from services.desktop_app.process_graph import PROCESS_MODULES, ProcessGraph

    graph = ProcessGraph()
    graph.start_all()
    try:
        assert set(graph.children) == set(PROCESS_MODULES)
        for proc in graph.children.values():
            assert proc.is_alive(), f"{proc.name} failed to start"
            assert proc.pid is not None
        assert graph.channels is not None
        assert graph.channels.analytics_inbox is not None
    finally:
        graph.stop_all(timeout=15.0)

    assert graph.children == {}
    assert graph.shutdown_events == {}
