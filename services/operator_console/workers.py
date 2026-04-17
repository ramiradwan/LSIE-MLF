"""Qt worker primitives for non-blocking API access.

Two shapes are exposed:

* ``PollingWorker`` — a QObject that, once moved to a QThread, calls a
  provided fetch callable on a fixed interval and emits ``data_ready`` or
  ``error`` signals. Views attach their refresh logic to the signals.
* ``run_one_shot`` — convenience for a single POST (e.g. stimulus inject)
  that runs the callable on a QThreadPool and emits the result through
  a small ``OneShotSignals`` QObject.

Keeping the worker layer tiny and signal-driven means panel code never
calls ``ApiClient`` directly — it subscribes to signals. Swapping the
transport (to httpx, or to a WebSocket) is a single-file change.
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
    """Periodic fetcher. Move to a QThread, call ``start()``.

    ``fetch`` is expected to be network-bound and synchronous; the
    QThread isolates it from the UI thread so a slow call never freezes
    the window.
    """

    data_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        fetch: Callable[[], object],
        interval_ms: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._fetch = fetch
        self._interval_ms = interval_ms
        self._timer: QTimer | None = None

    @Slot()
    def start(self) -> None:
        self._run_once()
        timer = QTimer(self)
        timer.setInterval(self._interval_ms)
        timer.timeout.connect(self._run_once)
        timer.start()
        self._timer = timer

    @Slot()
    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None

    @Slot()
    def refresh_now(self) -> None:
        self._run_once()

    def _run_once(self) -> None:
        try:
            payload = self._fetch()
        except ApiError as exc:
            self.error.emit(str(exc))
            return
        except Exception as exc:  # defensive — don't let the worker die
            self.error.emit(f"unexpected error: {exc}")
            return
        self.data_ready.emit(payload)


class OneShotSignals(QObject):
    """Signals bundle for ``run_one_shot``."""

    finished = Signal(object)
    failed = Signal(str)


class _OneShotRunnable(QRunnable):
    def __init__(self, fn: Callable[[], object], signals: OneShotSignals) -> None:
        super().__init__()
        self._fn = fn
        self._signals = signals

    def run(self) -> None:
        try:
            result = self._fn()
        except ApiError as exc:
            self._signals.failed.emit(str(exc))
            return
        except Exception as exc:
            self._signals.failed.emit(f"unexpected error: {exc}")
            return
        self._signals.finished.emit(result)


def run_one_shot(fn: Callable[[], object]) -> OneShotSignals:
    """Dispatch a single callable on the global QThreadPool.

    The returned signals object must be kept alive by the caller
    (typically a panel instance) until one of its signals fires.
    """
    signals = OneShotSignals()
    QThreadPool.globalInstance().start(_OneShotRunnable(fn, signals))
    return signals
