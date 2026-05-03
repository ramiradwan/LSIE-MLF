"""
Qt worker primitives for non-blocking API access.

Two primitives live here:

  * `PollingWorker` — a QObject that, once moved to a QThread, calls a
    provided fetch callable on a fixed interval and emits the result.
    The worker carries a `job_name` so one signal bus
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
        self._stopped = False

    @property
    def job_name(self) -> str:
        return self._job_name

    @Slot()
    def run(self) -> None:
        """Start the periodic fetch. Emits the first tick immediately so
        the UI shows data without waiting one interval."""
        if self._timer is not None or self._stopped:
            return
        self.started.emit(self._job_name)
        timer = QTimer(self)
        timer.setInterval(self._interval_ms)
        timer.timeout.connect(self._run_once)
        timer.start()
        self._timer = timer
        QTimer.singleShot(0, self._run_once)

    @Slot()
    def stop(self) -> None:
        # Stop the QTimer on its owning thread (this slot is invoked
        # via QueuedConnection from the coordinator, so we are on the
        # worker thread). Drop our Python ref but leave Qt's parent-
        # child ownership in place — the timer is destroyed when the
        # worker itself is destroyed via the `stopped → deleteLater`
        # connection wired in `PollingCoordinator._start_job`. That
        # delete also runs on the worker thread, which is the only
        # way to avoid the cross-thread `killTimer` warning.
        if self._stopped:
            return
        self._stopped = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.stopped.emit(self._job_name)

    @Slot()
    def refresh_now(self) -> None:
        """Force an immediate tick outside the regular cadence."""
        if self._stopped:
            return
        self._run_once()

    def _run_once(self) -> None:
        if self._stopped:
            return
        try:
            payload = self._fetch()
        except ApiError as exc:
            if not self._stopped:
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
            if not self._stopped:
                self.error.emit(self._job_name, wrapped)
            return
        if not self._stopped:
            self.data_ready.emit(self._job_name, payload)


class OneShotSignals(QObject):
    """Signals bundle for `run_one_shot`. Every signal carries the job
    name so callers can disambiguate when several POSTs are in flight
    (e.g., the operator double-clicks the stimulus button)."""

    # fmt: off
    succeeded  = Signal(str, object)        # job_name, payload
    failed     = Signal(str, object)        # job_name, ApiError
    finished   = Signal(str)                # job_name — fires after succeeded/failed
    _completed = Signal(str, bool, object)  # job_name, succeeded, payload_or_error
    # fmt: on

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._completed.connect(self._deliver_completed)

    @Slot(str, bool, object)
    def _deliver_completed(self, job_name: str, succeeded: bool, payload: object) -> None:
        if succeeded:
            self.succeeded.emit(job_name, payload)
        else:
            self.failed.emit(job_name, payload)
        self.finished.emit(job_name)


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
            self._signals._completed.emit(self._job_name, False, exc)
        except Exception as exc:
            wrapped = ApiError(
                message=f"unexpected error: {exc}",
                endpoint=None,
                retryable=False,
            )
            self._signals._completed.emit(self._job_name, False, wrapped)
        else:
            self._signals._completed.emit(self._job_name, True, result)


def run_one_shot(job_name: str, fn: Callable[[], object]) -> OneShotSignals:
    """Dispatch a single callable on the global QThreadPool.

    The returned signals object must be kept alive by the caller
    (typically the `PollingCoordinator`) until `finished` fires; Qt
    otherwise garbage-collects it mid-flight and the slots never see
    the emission.
    """
    signals = OneShotSignals()
    QTimer.singleShot(
        0,
        lambda: QThreadPool.globalInstance().start(_OneShotRunnable(job_name, fn, signals)),
    )
    return signals
