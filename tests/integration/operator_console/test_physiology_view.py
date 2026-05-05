"""Integration: Physiology page — stale badge + null-valid co-modulation.

Two composition-level assertions:

  1. When `is_stale=True` on the streamer snapshot the StatusPill reads
     "stale" — operator trust hinges on §4.C.4's four-state distinction
     (fresh / stale / absent / no-rmssd) being visible, not collapsed.
  2. When `co_modulation_index=None` the panel renders the §7C
     `null_reason` as a legitimate INFO outcome, not as an error. A
     null index is a valid signal that not enough aligned non-stale
     pairs existed in the window.

Spec references:
  §4.C.4         — Physiological State Buffer freshness distinction
  §7C            — Co-Modulation Index null-valid semantics
  §12            — null-valid must not surface as degraded/error
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from packages.schemas.operator_console import (
    CoModulationSummary,
    PhysiologyCurrentSnapshot,
    SessionPhysiologySnapshot,
    UiStatusKind,
)
from services.operator_console.state import OperatorStore
from services.operator_console.viewmodels.physiology_vm import PhysiologyViewModel
from services.operator_console.views.physiology_view import PhysiologyView

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def test_physiology_view_stale_snapshot_surfaces_warn_badge() -> None:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    view = PhysiologyView(vm)

    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=uuid4(),
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=42.0,
                heart_rate_bpm=71,
                is_stale=True,
                freshness_s=120.0,
                provider="oura",
            ),
            generated_at_utc=_NOW,
        )
    )
    panel = view._streamer_panel  # type: ignore[attr-defined]
    assert panel._status._label.text() == "stale"  # type: ignore[attr-defined]
    assert panel._status._kind is UiStatusKind.WARN  # type: ignore[attr-defined]
    # The freshness card's status mirrors the pill; operator should read
    # both as the same warning signal, not two different things.
    assert panel._freshness_card._status._kind is UiStatusKind.WARN  # type: ignore[attr-defined]


def test_physiology_view_null_comodulation_renders_null_reason_as_info() -> None:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    view = PhysiologyView(vm)

    session_id = uuid4()
    null_reason = "insufficient aligned non-stale pairs in window"
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=session_id,
            generated_at_utc=_NOW,
            comodulation=CoModulationSummary(
                session_id=session_id,
                co_modulation_index=None,
                n_paired_observations=1,
                coverage_ratio=0.1,
                null_reason=null_reason,
                window_start_utc=_NOW,
                window_end_utc=_NOW,
            ),
        )
    )
    panel = view._comodulation_panel  # type: ignore[attr-defined]

    # Null-valid pill kind is INFO, not WARN/ERROR. §7C draws this line.
    assert panel._index_card._status._kind is UiStatusKind.INFO  # type: ignore[attr-defined]
    assert panel._index_card._status._label.text() == "not enough data yet"  # type: ignore[attr-defined]

    # And the rendered explanation surfaces the stored null_reason —
    # operator must see *why* the index is null, not just "null".
    explanation_text = panel._explanation.text()  # type: ignore[attr-defined]
    assert "insufficient" in explanation_text
