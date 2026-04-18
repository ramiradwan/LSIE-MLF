"""QApplication bootstrap and ``main()`` entry point for the Operator
Console.

Phase 6 factors the bootstrap into four tiny factories (`build_*`) so
tests can construct the dependency graph without also standing up a
`QApplication`, and so the ordering constraint that actually matters —
*instantiate the store before any view, start polling only after the
window is shown* — lives in one place.

Ordering rationale:
  1. Store first: views bind their slots to store signals on
     construction, so the store has to exist before `build_main_window`.
  2. Window.show() before coordinator.start(): if polling kicks off
     while Qt is still laying out the shell, the first fetch can land
     before the stacked widgets are visible, which manifests as a
     flicker of "Loading…" cards that never had a chance to render.
  3. `app.aboutToQuit` → `coordinator.stop()` as a belt to
     `closeEvent`'s braces — covers the case where the app exits via
     `Ctrl+C` or similar paths that bypass the window's close.

Spec references:
  §4.E.1         — operator-facing execution details
  §12            — clean shutdown joins polling threads within 2s
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from services.operator_console.api_client import ApiClient
from services.operator_console.config import OperatorConsoleConfig, load_config
from services.operator_console.polling import PollingCoordinator
from services.operator_console.state import OperatorStore
from services.operator_console.theme import build_stylesheet
from services.operator_console.views.main_window import MainWindow


def build_api_client(config: OperatorConsoleConfig) -> ApiClient:
    """Instantiate the typed REST client bound to the configured API."""
    return ApiClient(config.api_base_url, config.api_timeout_seconds)


def build_store() -> OperatorStore:
    """Instantiate the app-scoped state holder."""
    return OperatorStore()


def build_polling_coordinator(
    config: OperatorConsoleConfig,
    client: ApiClient,
    store: OperatorStore,
) -> PollingCoordinator:
    """Instantiate the polling coordinator.

    The coordinator is NOT started here — callers must invoke
    `coordinator.start()` after `window.show()` per the ordering
    documented in the module docstring.
    """
    return PollingCoordinator(config, client, store)


def build_main_window(
    config: OperatorConsoleConfig,
    store: OperatorStore,
    coordinator: PollingCoordinator,
) -> MainWindow:
    """Instantiate the shell window wired to the store + coordinator."""
    return MainWindow(config, store, coordinator)


def main(argv: Sequence[str] | None = None) -> int:
    """Boot the Operator Console.

    The function is intentionally straight-line: factories compose the
    graph, `show()` lays out the shell, `start()` begins polling, and
    `app.exec()` hands control to Qt. Any exception before `exec()`
    bubbles out — there is no recovery path for a mis-configured
    startup on the operator host.
    """
    config = load_config()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("LSIE-MLF Operator Console")
    app.setOrganizationName("LSIE-MLF")
    app.setStyleSheet(build_stylesheet())

    # 1. store first — views and the coordinator both subscribe to its
    #    signals on construction.
    store = build_store()
    client = build_api_client(config)
    coordinator = build_polling_coordinator(config, client, store)

    # 2. window constructed; no polling yet, no network traffic.
    window = build_main_window(config, store, coordinator)
    window.show()

    # 3. start polling only after the shell is visible.
    coordinator.start()

    # 4. shutdown belt: `aboutToQuit` covers Ctrl+C and other paths
    #    that bypass `MainWindow.closeEvent`. Idempotent — calling
    #    `stop()` twice is a no-op.
    app.aboutToQuit.connect(coordinator.stop)

    return app.exec()
