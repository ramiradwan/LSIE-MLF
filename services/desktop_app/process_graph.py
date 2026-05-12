"""Six-process desktop graph bootstrap.

The desktop runtime runs as six named ``multiprocessing.Process``
children spawned via the ``spawn`` start method, matching §9.1. The
parent never imports a process module directly, so the ML libraries
remain isolated to ``gpu_ml_worker``.

Each process module exposes a ``run(shutdown_event, channels)``
callable. ``channels`` is the :class:`IpcChannels` bundle carrying the
queues that connect the graph.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import multiprocessing.context as mpcontext
import multiprocessing.synchronize as mpsync
import time
from dataclasses import dataclass, field
from typing import Literal

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.startup_timing import log_startup_milestone

logger = logging.getLogger(__name__)

RuntimeMode = Literal["operator_console", "operator_api"]

# Canonical process-name → module-path mapping. The order is the order
# in which processes are spawned and is the same order the v4.0 spec
# §9 process table uses.
PROCESS_MODULES: dict[str, str] = {
    "ui_api_shell": "services.desktop_app.processes.ui_api_shell",
    "capture_supervisor": "services.desktop_app.processes.capture_supervisor",
    "module_c_orchestrator": "services.desktop_app.processes.module_c_orchestrator",
    "gpu_ml_worker": "services.desktop_app.processes.gpu_ml_worker",
    "analytics_state_worker": "services.desktop_app.processes.analytics_state_worker",
    "cloud_sync_worker": "services.desktop_app.processes.cloud_sync_worker",
}


def process_modules_for_mode(runtime_mode: RuntimeMode) -> dict[str, str]:
    modules = dict(PROCESS_MODULES)
    if runtime_mode == "operator_api":
        modules["ui_api_shell"] = "services.desktop_app.processes.operator_api_runtime"
    return modules


# Modules that MUST NOT appear in the parent's ``sys.modules`` after
# importing process_graph. A canary test re-asserts this in a clean
# subprocess on every CI run.
ML_LIB_ROOTS: frozenset[str] = frozenset({"torch", "mediapipe", "faster_whisper", "ctranslate2"})

SQLITE_FILENAME = "desktop.sqlite"
ML_INBOX_MAXSIZE = 8
PCM_ACKS_MAXSIZE = 32
COOPERATIVE_SHUTDOWN_TIMEOUT_S = 15.0


def _prepare_runtime_state() -> None:
    import sqlite3

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.recovery import run_recovery_sweep
    from services.desktop_app.state.sqlite_schema import bootstrap_schema

    db_path = resolve_state_dir() / SQLITE_FILENAME
    bootstrap_conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(bootstrap_conn)
    finally:
        bootstrap_conn.close()
    run_recovery_sweep(db_path)


def _launch(
    module_name: str,
    shutdown_event: mpsync.Event,
    channels: IpcChannels,
) -> None:
    """Pickleable child entrypoint — runs only inside the spawned child.

    Spawn-mode children re-import this module to find ``_launch``, then
    invoke it. The child installs the crash-privacy guards first,
    because crash-dialog state is per-process on Windows and does not
    propagate from the parent, then imports its target module.
    """
    import contextlib
    import signal as _signal

    from services.desktop_app.privacy.crash_dumps import install_crash_privacy_guards

    install_crash_privacy_guards()

    # Windows broadcasts CTRL_BREAK_EVENT (SIGBREAK) and CTRL_C_EVENT
    # (SIGINT) to every process in the parent's
    # CREATE_NEW_PROCESS_GROUP. Python's default behaviour for those
    # signals is to abort the child mid-instruction
    # (STATUS_CONTROL_C_EXIT, exit code 0xC000013A), which (a) skips
    # the run-loop ``finally`` blocks that own scrcpy/SQLite cleanup
    # and (b) leaves capture artefacts retained until the next
    # startup's recovery sweep — exactly the §5.2 transient-cleanup
    # contract violation the privacy-baseline runs were designed to
    # surface. Install a no-op handler so the child stays alive long
    # enough to observe the parent's cooperative shutdown_event in its
    # normal run-loop poll. Some children may restrict signal
    # registration; the parent's cooperative shutdown_event still
    # works, so a registration failure is non-fatal.
    def _child_shutdown_noop(_signum: int, _frame: object) -> None:
        return

    for _signame in ("SIGBREAK", "SIGINT", "SIGTERM"):
        _signo = getattr(_signal, _signame, None)
        if _signo is not None:
            with contextlib.suppress(OSError, ValueError):
                _signal.signal(_signo, _child_shutdown_noop)

    mod = importlib.import_module(module_name)
    mod.run(shutdown_event=shutdown_event, channels=channels)


@dataclass
class ProcessGraph:
    """Lifecycle manager for the six v4.0 desktop processes.

    Public surface is intentionally narrow: ``start_all`` to spawn,
    ``stop_all`` to signal cooperative shutdown then force-terminate
    after a grace period, ``wait`` for blocking join. The dict members
    are exposed for the canary tests to introspect PIDs and exit codes
    without needing private accessors.
    """

    runtime_mode: RuntimeMode = "operator_console"
    children: dict[str, mpcontext.SpawnProcess] = field(default_factory=dict)
    shutdown_events: dict[str, mpsync.Event] = field(default_factory=dict)
    channels: IpcChannels | None = field(default=None)
    _shutdown_requested: bool = field(default=False, init=False)
    _shutdown_signaled: bool = field(default=False, init=False)

    def start_all(self) -> None:
        self._shutdown_requested = False
        self._shutdown_signaled = False
        _prepare_runtime_state()
        ctx = mp.get_context("spawn")
        if self.channels is None:
            self.channels = IpcChannels(
                ml_inbox=ctx.Queue(maxsize=ML_INBOX_MAXSIZE),
                drift_updates=ctx.Queue(),
                analytics_inbox=ctx.Queue(),
                pcm_acks=ctx.Queue(maxsize=PCM_ACKS_MAXSIZE),
                live_control=ctx.Queue(),
                segment_control=ctx.Queue(),
            )
        for name, module in process_modules_for_mode(self.runtime_mode).items():
            evt = ctx.Event()
            proc = ctx.Process(
                name=name,
                target=_launch,
                args=(module, evt, self.channels),
                daemon=False,
            )
            proc.start()
            self.children[name] = proc
            self.shutdown_events[name] = evt
            logger.info("spawned %s pid=%s", name, proc.pid)
        log_startup_milestone("process_graph_spawned", logger=logger)

    def signal_shutdown(self) -> None:
        """Signal-safe: set a flag without touching multiprocessing primitives.

        ``multiprocessing.Event.set`` acquires a Condition variable that
        deadlocks when invoked from a Windows signal handler after the
        first event has been signalled. The wait loop polls
        ``_shutdown_signaled`` and calls :meth:`request_shutdown` from
        the main thread, which is safe.
        """
        self._shutdown_signaled = True

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        self._shutdown_signaled = True
        for evt in self.shutdown_events.values():
            evt.set()

    def stop_all(self, timeout: float = COOPERATIVE_SHUTDOWN_TIMEOUT_S) -> None:
        self.request_shutdown()
        for name, proc in self.children.items():
            proc.join(timeout=timeout)
            if proc.is_alive():
                logger.warning("force-terminating %s pid=%s", name, proc.pid)
                proc.terminate()
                proc.join(timeout=timeout)
        self.children.clear()
        self.shutdown_events.clear()
        self.cleanup_capture_artifacts()

    def cleanup_capture_artifacts(self) -> None:
        from services.desktop_app.os_adapter import resolve_capture_dir, resolve_state_dir
        from services.desktop_app.privacy.zeroize import cleanup_capture_files
        from services.desktop_app.state.recovery import reap_orphan_capture_processes

        reaped, survived = reap_orphan_capture_processes(resolve_state_dir() / SQLITE_FILENAME)
        if reaped or survived:
            logger.info(
                "desktop shutdown capture process reap: reaped=%s survived=%s",
                reaped,
                survived,
            )

        # Generous retry budget so transient holders (Windows Defender
        # real-time scan of a freshly-closed recording, Search Indexer,
        # scrcpy descendants finishing their unwind) have time to drop
        # the file handle before we give up. The §5.2 24-hour Ephemeral
        # Vault bound is honored by raising on retention, so the budget
        # only governs how patient cleanup is, not whether it must
        # eventually succeed.
        deleted, retained = cleanup_capture_files(
            resolve_capture_dir(),
            attempts=60,
            retry_delay_s=0.5,
        )
        if retained:
            retained_paths = [str(path) for path in retained]
            logger.error(
                "desktop shutdown retained transient capture artifacts: %s",
                retained_paths,
            )
            raise RuntimeError(
                "desktop shutdown retained transient capture artifacts: "
                + ", ".join(retained_paths)
            )
        elif deleted:
            logger.info(
                "desktop shutdown deleted transient capture artifacts: %s",
                [str(path) for path in deleted],
            )

    def wait(
        self,
        poll_interval: float = 0.25,
        shutdown_timeout: float = COOPERATIVE_SHUTDOWN_TIMEOUT_S,
    ) -> None:
        shutdown_started_at: float | None = None
        while self.children:
            exited: list[tuple[str, int | None]] = []
            alive = False
            for name, proc in self.children.items():
                if proc.is_alive():
                    alive = True
                    continue
                exited.append((name, proc.exitcode))

            if exited and not self._shutdown_requested:
                name, exitcode = exited[0]
                logger.info("%s exited with code %s; requesting graph shutdown", name, exitcode)
                self.request_shutdown()

            # The signal handler only flips ``_shutdown_signaled`` (it
            # must not invoke ``Event.set`` itself; see
            # :meth:`signal_shutdown`). Drain the multiprocessing
            # primitives here on the main thread.
            if self._shutdown_signaled and not self._shutdown_requested:
                self.request_shutdown()

            if self._shutdown_requested and shutdown_started_at is None:
                shutdown_started_at = time.monotonic()

            if not alive:
                return

            if (
                shutdown_started_at is not None
                and (time.monotonic() - shutdown_started_at) >= shutdown_timeout
            ):
                self.stop_all(timeout=max(1.0, shutdown_timeout))
                return

            time.sleep(poll_interval)
