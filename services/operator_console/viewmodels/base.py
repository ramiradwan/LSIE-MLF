"""Base viewmodel for Operator Console pages — Phase 7.

`ViewModelBase` is the thin `QObject` superclass every page-scoped
viewmodel (`OverviewVM`, `LiveSessionVM`, `ExperimentsVM`, `PhysiologyVM`,
`HealthVM`) inherits in Phase 8. It exists so every view can bind to the
same three-signal surface without each VM re-declaring the boilerplate:

  * ``changed`` — the VM's rendered state changed; the view re-reads.
  * ``toast_requested(level, message)`` — operator-facing notice that is
    orthogonal to the page's primary data (e.g. "Stimulus accepted" after
    the write path succeeds). `level` is a free-form tag (``"info"``,
    ``"warn"``, ``"error"``) so the shell can style the toast.
  * ``error_changed(message)`` — VM-scoped error message; empty string
    means "cleared". Views render this into the page-level error slot,
    separate from the store's scope-keyed `error_changed`.

The VM holds a reference to the shared `OperatorStore` so subclasses can
read the current snapshot lazily in their getters rather than caching
state locally — the store is the single source of truth.

Spec references:
  §4.E.1         — operator-facing page surfaces (Overview, Live Session,
                   Experiments, Physiology, Health, Sessions)
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from services.operator_console.state import OperatorStore


class ViewModelBase(QObject):
    """Shared signal surface + store accessor for page viewmodels.

    Subclasses are free to add their own signals for VM-specific slices
    (e.g. `LiveSessionVM.encounter_selected`), but they must emit the
    base `changed` signal whenever the view needs to re-render its
    primary content so that the view's wiring stays uniform.
    """

    # fmt: off
    changed         = Signal()
    toast_requested = Signal(str, str)
    error_changed   = Signal(str)
    # fmt: on

    def __init__(self, store: OperatorStore, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._store: OperatorStore = store
        self._error: str = ""

    @property
    def store(self) -> OperatorStore:
        """Expose the shared store for subclasses; views must not touch it."""
        return self._store

    # ------------------------------------------------------------------
    # Helpers — subclasses drive the signal bus through these so tests
    # can exercise them without constructing a full view tree.
    # ------------------------------------------------------------------

    def emit_changed(self) -> None:
        """Notify the paired view that the primary render should refresh."""
        self.changed.emit()

    def set_error(self, message: str | None) -> None:
        """Set or clear the VM-scoped error message.

        Passing ``None`` or an empty string clears the error. A debounce
        guard avoids re-emitting the same value — an operator-facing
        error banner should not flicker on every poll tick that carries
        the same failure reason.
        """
        normalized = message or ""
        if normalized == self._error:
            return
        self._error = normalized
        self.error_changed.emit(normalized)

    def error(self) -> str:
        """Current error message; empty string means no error set."""
        return self._error

    def emit_toast(self, level: str, message: str) -> None:
        """Request the shell to show a transient toast.

        `level` is a free-form tag (``"info"``, ``"warn"``, ``"error"``)
        so the shell can style the toast without the VM owning widget
        concerns.
        """
        self.toast_requested.emit(level, message)
