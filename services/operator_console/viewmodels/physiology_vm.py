"""Physiology page viewmodel — Phase 8.

The physiology surface has to communicate four distinct states per
`subject_role`:
  * **fresh** — recent, non-stale snapshot (§4.C.4 freshness clock)
  * **stale** — snapshot exists but `is_stale=True` (§4.C.4)
  * **absent** — no snapshot at all (operator/streamer not wearing strap)
  * **null-valid** co-modulation — §7C: insufficient aligned non-stale
    pairs is a *valid* outcome, not an error; `null_reason` renders it

Collapsing those states into a generic "no data" reads as a bug; this
VM keeps them separate via dedicated getters and the explanation string.

Spec references:
  §4.C.4         — Physiological State Buffer freshness / staleness
  §4.E.2         — physiology_log fields (rmssd_ms, heart_rate_bpm, ...)
  §7C            — Co-Modulation Index rolling Pearson, null-valid
  SPEC-AMEND-007 — v3.1 physiology transport
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from packages.schemas.operator_console import (
    CoModulationSummary,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
)
from services.operator_console.formatters import build_physiology_explanation
from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.base import ViewModelBase


class PhysiologyViewModel(ViewModelBase):
    """Exposes per-role snapshots + the §7C co-modulation summary."""

    def __init__(self, store: OperatorStore, parent: QObject | None = None) -> None:
        super().__init__(store, parent)
        store.physiology_changed.connect(self._on_any_change)
        store.error_changed.connect(self._on_error)
        store.error_cleared.connect(self._on_error_cleared)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def snapshot(self) -> SessionPhysiologySnapshot | None:
        return self._store.physiology()

    def operator_snapshot(self) -> PhysiologyCurrentSnapshot | None:
        snap = self._store.physiology()
        return snap.operator if snap is not None else None

    def streamer_snapshot(self) -> PhysiologyCurrentSnapshot | None:
        snap = self._store.physiology()
        return snap.streamer if snap is not None else None

    def comodulation(self) -> CoModulationSummary | None:
        snap = self._store.physiology()
        return snap.comodulation if snap is not None else None

    def comodulation_explanation(self) -> str:
        """Operator-readable co-modulation summary.

        §7C — a null index with a `null_reason` (e.g. "insufficient
        aligned pairs") must read as legitimate, not as an error.
        """
        summary = self.comodulation()
        if summary is None:
            return "No co-modulation window yet."
        if summary.co_modulation_index is None:
            reason = summary.null_reason or "insufficient aligned non-stale pairs"
            return f"Co-Modulation Index: null — {reason}."
        return build_physiology_explanation(self._store.physiology())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_any_change(self, _payload: object) -> None:
        self.emit_changed()

    def _on_error(self, scope: str, message: str) -> None:
        if scope == "physiology":
            self.set_error(message)

    def _on_error_cleared(self, scope: str) -> None:
        if scope == "physiology":
            self.set_error(None)
