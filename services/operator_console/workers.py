"""
Qt worker primitives for non-blocking API access — Phase 4 revision.

Two primitives live here:

  * `PollingWorker` — a QObject that, once moved to a QThread, calls a
    provided fetch callable on a fixed interval and emits the result.
    In Phase 4 the worker carries a `job_name` so one signal bus
    (`PollingCoordinator`) can multiplex many independent jobs through
    the same slot surface — previously every view wired its own worker
    and its own signal, which does not scale to nine parallel polls.
  * `run_one_shot` — convenience for a single POST (e.g. operator
    stimulus submission) that runs the callable on the global
    `QThreadPool` and reports via `OneShotSignals`. The signal bus
    carries the job name so the coordinator can tell which submission
    finished when several are in flight.

Design constraints:
  - No widget references in this module. Panels subscribe to signals,
    never import workers.
  - Exceptions are caught and re-emitted as `error` / `failed` signals;
    an uncaught exception in the worker thread would kill the UI.
  - `ApiError` is surfaced as the full object (not just `str`) so the
    coordinator can read `retryable` / `endpoint` and route the error
    to the right operator-facing scope.

Spec references:
  §12            — error-handling matrix (retry vs surface)
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)

from services.operator_console.api_client import ApiError


class PollingWorker(QObject):
    """Periodic fetcher. Move to a QThread, call ``run()``.

    `fetch` is expected to be network-bound and synchronous; the
    owning thread isolates it from the UI thread so a slow call never
    freezes the window. Every signal carries `job_name` so a single
    coordinator can multiplex many workers.
    """

    # fmt: off
    started     = Signal(str)           # job_name
    data_ready  = Signal(str, object)   # job_name, payload
    error       = Signal(str, object)   # job_name, ApiError
    stopped     = Signal(str)           # job_name
    # fmt: on

    def __init__(
        self,
        job_name: str,
        interval_ms: int,
        fetch: Callable[[], object],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._job_name = job_name
        self._interval_ms = interval_ms
        self._fetch = fetch
        self._timer: QTimer | None = None

    @property
    def job_name(self) -> str:
        return self._job_name

    @Slot()
    def run(self) -> None:
        """Start the periodic fetch. Emits the first tick immediately so
        the UI shows data without waiting one interval."""
        if self._timer is not None:
            return
        self.started.emit(self._job_name)
        self._run_once()
        timer = QTimer(self)
        timer.setInterval(self._interval_ms)
        timer.timeout.connect(self._run_once)
        timer.start()
        self._timer = timer

    @Slot()
    def stop(self) -> None:
        if self._timer is None:
            return
        self._timer.stop()
        self._timer.deleteLater()
        self._timer = None
        self.stopped.emit(self._job_name)

    @Slot()
    def refresh_now(self) -> None:
        """Force an immediate tick outside the regular cadence."""
        self._run_once()

    def _run_once(self) -> None:
        try:
            payload = self._fetch()
        except ApiError as exc:
            self.error.emit(self._job_name, exc)
            return
        except Exception as exc:  # defensive — don't let the worker die
            # §12: a non-ApiError in a worker is a bug on our side;
            # surface it as a non-retryable ApiError so the UI treats
            # it like any other permanent failure.
            wrapped = ApiError(
                message=f"unexpected error: {exc}",
                endpoint=None,
                retryable=False,
            )
            self.error.emit(self._job_name, wrapped)
            return
        self.data_ready.emit(self._job_name, payload)


class OneShotSignals(QObject):
    """Signals bundle for `run_one_shot`. Every signal carries the job
    name so callers can disambiguate when several POSTs are in flight
    (e.g., the operator double-clicks the stimulus button)."""

    # fmt: off
    succeeded  = Signal(str, object)   # job_name, payload
    failed     = Signal(str, object)   # job_name, ApiError
    finished   = Signal(str)           # job_name — fires after succeeded/failed
    # fmt: on


class _OneShotRunnable(QRunnable):
    """Runs one callable on the global QThreadPool and reports via signals."""

    def __init__(
        self,
        job_name: str,
        fn: Callable[[], object],
        signals: OneShotSignals,
    ) -> None:
        super().__init__()
        self._job_name = job_name
        self._fn = fn
        self._signals = signals

    def run(self) -> None:
        try:
            result = self._fn()
        except ApiError as exc:
            self._signals.failed.emit(self._job_name, exc)
        except Exception as exc:
            wrapped = ApiError(
                message=f"unexpected error: {exc}",
                endpoint=None,
                retryable=False,
            )
            self._signals.failed.emit(self._job_name, wrapped)
        else:
            self._signals.succeeded.emit(self._job_name, result)
        finally:
            self._signals.finished.emit(self._job_name)


def run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
    """Dispatch a single callable on the global QThreadPool.

    The returned signals object must be kept alive by the caller
    (typically the `PollingCoordinator`) until `finished` fires; Qt
    otherwise garbage-collects it mid-flight and the slots never see
    the emission.
    """
    signals = OneShotSignals()
    QThreadPool.globalInstance().start(_OneShotRunnable(job_name, fn, signals))
    return signals
