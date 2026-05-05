"""Tests for `PhysiologyView` — Phase 10.

Locks the four explicit states §4.C.4 requires the operator to see:
fresh / stale / absent / no-rmssd — and the §7C null-valid co-modulation
path which is a legitimate outcome, not an error.
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


def _view() -> tuple[PhysiologyView, OperatorStore]:
    store = OperatorStore()
    vm = PhysiologyViewModel(store)
    return PhysiologyView(vm), store


def test_physiology_view_empty_until_snapshot_set() -> None:
    view, _store = _view()
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is True  # type: ignore[attr-defined]
    assert view._scroll.widget() is view._body_container  # type: ignore[attr-defined]


def test_physiology_view_fresh_streamer_renders_ok_pill() -> None:
    view, store = _view()
    session_id = uuid4()
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=session_id,
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=45.0,
                heart_rate_bpm=72,
                is_stale=False,
                freshness_s=4.0,
                provider="oura",
            ),
            generated_at_utc=_NOW,
        )
    )
    panel = view._streamer_panel  # type: ignore[attr-defined]
    assert panel._status._label.text() == "fresh"  # type: ignore[attr-defined]
    assert panel._status._kind is UiStatusKind.OK  # type: ignore[attr-defined]
    assert "45" in panel._rmssd_card._primary.text()  # type: ignore[attr-defined]


def test_physiology_view_stale_streamer_renders_warn_pill() -> None:
    view, store = _view()
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=uuid4(),
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=40.0,
                heart_rate_bpm=68,
                is_stale=True,
                freshness_s=90.0,
                provider="oura",
            ),
            generated_at_utc=_NOW,
        )
    )
    panel = view._streamer_panel  # type: ignore[attr-defined]
    assert panel._status._label.text() == "stale"  # type: ignore[attr-defined]
    assert panel._status._kind is UiStatusKind.WARN  # type: ignore[attr-defined]


def test_physiology_view_absent_operator_panel_reads_absent() -> None:
    view, store = _view()
    # Only the streamer is populated; the operator side is absent.
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=uuid4(),
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=45.0,
                is_stale=False,
                freshness_s=4.0,
            ),
            generated_at_utc=_NOW,
        )
    )
    panel = view._operator_panel  # type: ignore[attr-defined]
    assert panel._status._label.text() == "absent"  # type: ignore[attr-defined]
    assert panel._status._kind is UiStatusKind.NEUTRAL  # type: ignore[attr-defined]


def test_physiology_view_no_rmssd_path_is_distinct_from_absent() -> None:
    view, store = _view()
    # Sample exists but carries no RMSSD — the §4.C.4 "no-rmssd" state.
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=uuid4(),
            streamer=PhysiologyCurrentSnapshot(
                subject_role="streamer",
                rmssd_ms=None,
                heart_rate_bpm=70,
                is_stale=False,
                freshness_s=6.0,
                provider="oura",
            ),
            generated_at_utc=_NOW,
        )
    )
    panel = view._streamer_panel  # type: ignore[attr-defined]
    assert panel._status._label.text() == "no variability"  # type: ignore[attr-defined]
    assert panel._rmssd_card._primary.text() == "—"  # type: ignore[attr-defined]
    assert "heart-rate variability" in panel._rmssd_card._secondary.text()  # type: ignore[attr-defined]


def test_physiology_view_comodulation_null_valid_renders_as_info() -> None:
    view, store = _view()
    # §7C: when too few aligned pairs exist, the index is legitimately
    # null — render as INFO ("null-valid"), not WARN/ERROR.
    session_id = uuid4()
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=session_id,
            generated_at_utc=_NOW,
            comodulation=CoModulationSummary(
                session_id=session_id,
                co_modulation_index=None,
                n_paired_observations=1,
                coverage_ratio=0.1,
                null_reason="insufficient aligned non-stale pairs",
                window_start_utc=_NOW,
                window_end_utc=_NOW,
            ),
        )
    )
    assert view._scroll.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.widget() is view._body_container  # type: ignore[attr-defined]
    co_modulation_panel = view._co_modulation_summary_panel  # type: ignore[attr-defined]
    assert co_modulation_panel.accessibleName() == "Co-Modulation Index"
    assert co_modulation_panel._title.text() == "Co-Modulation Index"  # type: ignore[attr-defined]
    assert "Paired heart-data trends" in co_modulation_panel._subtitle.text()  # type: ignore[attr-defined]
    assert co_modulation_panel._status._kind is UiStatusKind.INFO  # type: ignore[attr-defined]
    assert co_modulation_panel._primary_card._secondary.text() == "Sync data accumulating"  # type: ignore[attr-defined]
    assert "insufficient aligned non-stale pairs" in co_modulation_panel._explanation.text()  # type: ignore[attr-defined]

    panel = view._comodulation_panel  # type: ignore[attr-defined]
    assert panel._index_card._status._label.text() == "not enough data yet"  # type: ignore[attr-defined]
    assert panel._index_card._status._kind is UiStatusKind.INFO  # type: ignore[attr-defined]
    assert panel._title.text() == "Shared stress/recovery movement"  # type: ignore[attr-defined]


def test_physiology_view_comodulation_numeric_updates_summary_panel() -> None:
    view, store = _view()
    session_id = uuid4()
    store.set_physiology(
        SessionPhysiologySnapshot(
            session_id=session_id,
            generated_at_utc=_NOW,
            comodulation=CoModulationSummary(
                session_id=session_id,
                co_modulation_index=0.82,
                n_paired_observations=8,
                coverage_ratio=0.75,
                null_reason=None,
                window_start_utc=_NOW,
                window_end_utc=_NOW,
            ),
        )
    )

    co_modulation_panel = view._co_modulation_summary_panel  # type: ignore[attr-defined]
    assert co_modulation_panel._primary_card._primary.text() == "+0.82"  # type: ignore[attr-defined]
    assert "moving together" in co_modulation_panel._primary_card._secondary.text()  # type: ignore[attr-defined]
    assert "8 fresh paired observation" in co_modulation_panel._explanation.text()  # type: ignore[attr-defined]


def test_physiology_view_error_changed_shows_alert_banner() -> None:
    view, store = _view()
    store.set_error("physiology", "ingest backlog")
    assert view._error_banner.isHidden() is False  # type: ignore[attr-defined]
