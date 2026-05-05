"""UI shell process for the PySide Operator Console."""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.processes.operator_api_runtime import start_operator_api_runtime

logger = logging.getLogger(__name__)

SHUTDOWN_POLL_INTERVAL_MS = 250


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("ui_api_shell starting")

    # Late imports preserve the ML-isolation canary contract
    # and keep the parent process free of Qt state.
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from services.operator_console.app import (
        build_api_client,
        build_main_window,
        build_polling_coordinator,
        build_store,
    )
    from services.operator_console.config import load_config
    from services.operator_console.theme import build_stylesheet

    api_runtime = start_operator_api_runtime(channels)

    qt_app = QApplication([])
    qt_app.setApplicationName("LSIE-MLF Operator Console")
    qt_app.setOrganizationName("LSIE-MLF")
    qt_app.setStyleSheet(build_stylesheet())

    config = load_config()
    store = build_store()
    client = build_api_client(config)
    coordinator = build_polling_coordinator(config, client, store)
    window = build_main_window(config, store, coordinator)
    window.show()
    coordinator.start()
    qt_app.aboutToQuit.connect(coordinator.stop)

    # Bridge the multiprocessing shutdown event to QApplication.quit so
    # the parent's stop_all() reaches the Qt event loop.
    def _check_shutdown() -> None:
        if shutdown_event.is_set():
            qt_app.quit()

    shutdown_timer = QTimer()
    shutdown_timer.setInterval(SHUTDOWN_POLL_INTERVAL_MS)
    shutdown_timer.timeout.connect(_check_shutdown)
    shutdown_timer.start()

    logger.info("ui_api_shell Qt event loop active")
    try:
        qt_app.exec()
    finally:
        api_runtime.stop()
        logger.info("ui_api_shell stopped")
