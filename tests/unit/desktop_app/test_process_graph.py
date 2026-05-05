"""ML import isolation canary for the desktop process graph.

The v4.0 desktop topology pivots away from a single all-in-one runtime
that loads every ML library into one process. The new contract is:
``torch`` / ``mediapipe`` / ``faster_whisper`` / ``ctranslate2`` are
imported only inside ``services.desktop_app.processes.gpu_ml_worker``,
and only after the spawned child re-imports it through
``services.desktop_app.process_graph._launch``.

If a future PR adds an ML import to (or transitively pulls one into)
the parent process or any non-ML process module, these tests fail
immediately, before the cuDNN-collision and spawn-deadlock symptoms can
surface in CI or in production.

The tests run each import in a clean subprocess so the test runner's
own ``sys.modules`` does not pollute the result.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

ML_LIB_ROOTS: tuple[str, ...] = ("torch", "mediapipe", "faster_whisper", "ctranslate2")

NON_ML_PROCESS_MODULES: tuple[str, ...] = (
    "services.desktop_app.processes.ui_api_shell",
    "services.desktop_app.processes.operator_api_runtime",
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


def test_main_smoke_runs_preflight_and_skips_process_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.desktop_app import __main__ as desktop_main

    preflight_calls: list[str] = []
    guard_calls: list[str] = []

    monkeypatch.setattr(
        "services.desktop_app.__main__.preflight.ensure_preflight",
        lambda: preflight_calls.append("called"),
    )
    monkeypatch.setattr(
        desktop_main,
        "install_crash_privacy_guards",
        lambda: guard_calls.append("called"),
    )

    class UnexpectedGraph:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("ProcessGraph should not be constructed in --smoke mode")

    monkeypatch.setattr(desktop_main, "ProcessGraph", UnexpectedGraph)

    assert desktop_main.main(["--smoke"]) == 0
    assert preflight_calls == ["called"]
    assert guard_calls == ["called"]


def test_main_operator_api_uses_operator_api_runtime_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.desktop_app import __main__ as desktop_main

    modes: list[str] = []
    lifecycle: list[str] = []

    monkeypatch.setattr("services.desktop_app.__main__.preflight.ensure_preflight", lambda: None)
    monkeypatch.setattr(desktop_main, "install_crash_privacy_guards", lambda: None)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)

    class FakeGraph:
        def __init__(self, *, runtime_mode: str) -> None:
            modes.append(runtime_mode)

        def request_shutdown(self) -> None:
            lifecycle.append("request_shutdown")

        def start_all(self) -> None:
            lifecycle.append("start_all")

        def wait(self) -> None:
            lifecycle.append("wait")

        def stop_all(self) -> None:
            lifecycle.append("stop_all")

    monkeypatch.setattr(desktop_main, "ProcessGraph", FakeGraph)

    assert desktop_main.main(["--operator-api"]) == 0
    assert modes == ["operator_api"]
    assert lifecycle == ["start_all", "wait", "stop_all"]


def test_main_smoke_skips_operator_api_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.desktop_app import __main__ as desktop_main

    monkeypatch.setattr("services.desktop_app.__main__.preflight.ensure_preflight", lambda: None)
    monkeypatch.setattr(desktop_main, "install_crash_privacy_guards", lambda: None)

    class UnexpectedGraph:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("ProcessGraph should not be constructed in --smoke mode")

    monkeypatch.setattr(desktop_main, "ProcessGraph", UnexpectedGraph)

    assert desktop_main.main(["--smoke", "--operator-api"]) == 0


@pytest.mark.parametrize("module_path", NON_ML_PROCESS_MODULES)
def test_non_ml_process_module_is_clean(module_path: str) -> None:
    """Each non-ML process module must not import any ML library, even transitively."""
    assert _imported_ml_roots(module_path) == []


def test_process_modules_table_is_complete() -> None:
    """The PROCESS_MODULES dict must list exactly the six §9.1 processes."""
    from services.desktop_app.process_graph import PROCESS_MODULES

    assert set(PROCESS_MODULES) == {
        "ui_api_shell",
        "capture_supervisor",
        "module_c_orchestrator",
        "gpu_ml_worker",
        "analytics_state_worker",
        "cloud_sync_worker",
    }


def test_operator_api_mode_replaces_only_shell_module() -> None:
    from services.desktop_app.process_graph import PROCESS_MODULES, process_modules_for_mode

    modules = process_modules_for_mode("operator_api")

    assert set(modules) == set(PROCESS_MODULES)
    assert modules["ui_api_shell"] == "services.desktop_app.processes.operator_api_runtime"
    for name in set(PROCESS_MODULES) - {"ui_api_shell"}:
        assert modules[name] == PROCESS_MODULES[name]


def test_process_graph_operator_api_mode_uses_selected_module_table() -> None:
    from services.desktop_app import process_graph

    graph = process_graph.ProcessGraph(runtime_mode="operator_api")
    modules = process_graph.process_modules_for_mode("operator_api")
    launched: list[tuple[str, str]] = []

    context = MagicMock()
    context.Queue.side_effect = lambda maxsize=0: MagicMock(maxsize=maxsize)
    context.Event.side_effect = lambda: MagicMock()

    def fake_process(
        *,
        name: str,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
    ) -> MagicMock:
        launched.append((name, cast(str, args[0])))
        proc = MagicMock()
        proc.pid = len(launched)
        return proc

    context.Process.side_effect = fake_process

    with (
        patch("services.desktop_app.process_graph._prepare_runtime_state"),
        patch("services.desktop_app.process_graph.mp.get_context", return_value=context),
    ):
        graph.start_all()

    assert launched == list(modules.items())
    assert set(graph.children) == set(modules)


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
        assert graph.channels.pcm_acks is not None
    finally:
        graph.stop_all(timeout=15.0)

    assert graph.children == {}
    assert graph.shutdown_events == {}


def test_request_shutdown_sets_every_child_event() -> None:
    from services.desktop_app.process_graph import ProcessGraph

    evt_a = MagicMock()
    evt_b = MagicMock()
    graph = ProcessGraph(shutdown_events={"a": evt_a, "b": evt_b})

    graph.request_shutdown()

    assert graph._shutdown_requested is True
    evt_a.set.assert_called_once_with()
    evt_b.set.assert_called_once_with()


def test_stop_all_uses_cooperative_shutdown_timeout_by_default() -> None:
    from services.desktop_app.process_graph import COOPERATIVE_SHUTDOWN_TIMEOUT_S, ProcessGraph

    child = MagicMock()
    child.is_alive.return_value = False
    graph = ProcessGraph(children=cast(Any, {"capture_supervisor": child}))

    with patch.object(graph, "cleanup_capture_artifacts"):
        graph.stop_all()

    child.join.assert_called_once_with(timeout=COOPERATIVE_SHUTDOWN_TIMEOUT_S)


def test_stop_all_cleans_capture_artifacts_after_children_stop() -> None:
    from services.desktop_app.process_graph import ProcessGraph

    child = MagicMock()
    child.is_alive.return_value = False
    graph = ProcessGraph(children=cast(Any, {"capture_supervisor": child}))

    with patch.object(graph, "cleanup_capture_artifacts") as cleanup_capture_artifacts:
        graph.stop_all(timeout=1.0)

    child.join.assert_called_once_with(timeout=1.0)
    cleanup_capture_artifacts.assert_called_once_with()
    assert graph.children == {}


def test_cleanup_capture_artifacts_uses_shutdown_retry_policy(tmp_path: Path) -> None:
    from services.desktop_app.process_graph import ProcessGraph

    calls: list[tuple[Path, int, float]] = []

    def fake_cleanup_capture_files(
        capture_dir: Path,
        *,
        attempts: int,
        retry_delay_s: float,
    ) -> tuple[list[Path], list[Path]]:
        calls.append((capture_dir, attempts, retry_delay_s))
        return [], []

    with (
        patch("services.desktop_app.os_adapter.resolve_capture_dir", return_value=tmp_path),
        patch("services.desktop_app.os_adapter.resolve_state_dir", return_value=tmp_path),
        patch(
            "services.desktop_app.state.recovery.reap_orphan_capture_processes",
            return_value=([], []),
        ),
        patch(
            "services.desktop_app.privacy.zeroize.cleanup_capture_files",
            fake_cleanup_capture_files,
        ),
    ):
        ProcessGraph().cleanup_capture_artifacts()

    assert calls == [(tmp_path, 12, 0.5)]


def test_cleanup_capture_artifacts_reaps_capture_processes_before_file_cleanup(
    tmp_path: Path,
) -> None:
    from services.desktop_app.process_graph import ProcessGraph

    calls: list[str] = []

    def fake_reap_orphan_capture_processes(db_path: Path) -> tuple[list[int], list[int]]:
        assert db_path == tmp_path / "desktop.sqlite"
        calls.append("reap")
        return [101], []

    def fake_cleanup_capture_files(
        capture_dir: Path,
        *,
        attempts: int,
        retry_delay_s: float,
    ) -> tuple[list[Path], list[Path]]:
        assert capture_dir == tmp_path
        assert attempts == 12
        assert retry_delay_s == 0.5
        calls.append("cleanup")
        return [], []

    with (
        patch("services.desktop_app.os_adapter.resolve_capture_dir", return_value=tmp_path),
        patch("services.desktop_app.os_adapter.resolve_state_dir", return_value=tmp_path),
        patch(
            "services.desktop_app.state.recovery.reap_orphan_capture_processes",
            fake_reap_orphan_capture_processes,
        ),
        patch(
            "services.desktop_app.privacy.zeroize.cleanup_capture_files",
            fake_cleanup_capture_files,
        ),
    ):
        ProcessGraph().cleanup_capture_artifacts()

    assert calls == ["reap", "cleanup"]


def test_cleanup_capture_artifacts_raises_on_retained_raw_media(tmp_path: Path) -> None:
    from services.desktop_app.process_graph import ProcessGraph

    video = tmp_path / "video_stream.mkv"
    with (
        patch("services.desktop_app.os_adapter.resolve_capture_dir", return_value=tmp_path),
        patch("services.desktop_app.os_adapter.resolve_state_dir", return_value=tmp_path),
        patch(
            "services.desktop_app.state.recovery.reap_orphan_capture_processes",
            return_value=([], []),
        ),
        patch(
            "services.desktop_app.privacy.zeroize.cleanup_capture_files",
            return_value=([], [video]),
        ),
        pytest.raises(RuntimeError, match="retained transient capture artifacts"),
    ):
        ProcessGraph().cleanup_capture_artifacts()


def test_wait_requests_shutdown_when_any_child_exits() -> None:
    from services.desktop_app.process_graph import ProcessGraph

    class FakeProcess:
        def __init__(self, states: list[bool], exitcode: int | None) -> None:
            self._states = list(states)
            self.exitcode = exitcode

        def is_alive(self) -> bool:
            if self._states:
                return self._states.pop(0)
            return False

    survivor = FakeProcess([True, True, False], exitcode=None)
    exiting = FakeProcess([False, False, False], exitcode=7)

    graph = ProcessGraph(
        children=cast(
            Any,
            {
                "ui_api_shell": exiting,
                "capture_supervisor": survivor,
            },
        )
    )

    with (
        patch.object(graph, "request_shutdown", wraps=graph.request_shutdown) as request_shutdown,
        patch("services.desktop_app.process_graph.time.sleep"),
    ):
        graph.wait(poll_interval=0.0, shutdown_timeout=10.0)

    request_shutdown.assert_called_once_with()


def test_wait_force_stops_after_shutdown_timeout() -> None:
    from services.desktop_app.process_graph import ProcessGraph

    child = MagicMock()
    child.is_alive.return_value = True
    graph = ProcessGraph(children=cast(Any, {"capture_supervisor": child}))
    graph.request_shutdown()

    monotonic_values = iter([10.0, 10.0, 16.0])
    with (
        patch.object(graph, "stop_all") as stop_all,
        patch(
            "services.desktop_app.process_graph.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ),
        patch("services.desktop_app.process_graph.time.sleep"),
    ):
        graph.wait(poll_interval=0.0, shutdown_timeout=5.0)

    stop_all.assert_called_once_with(timeout=5.0)
