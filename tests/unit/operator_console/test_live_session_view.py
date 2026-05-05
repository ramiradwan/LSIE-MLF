"""Tests for `LiveSessionView` — Phase 9.

Locks the render path the operator relies on: empty-state when no
session is selected, header populated from the live-session DTO (not
from row data), and the detail pane surfacing the §7B reward
explanation after a row is selected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from PySide6.QtWidgets import QDialog, QHeaderView

from packages.schemas.operator_console import (
    AttributionSummary,
    EncounterState,
    EncounterSummary,
    ExperimentSummary,
    HealthSnapshot,
    HealthState,
    HealthSubsystemStatus,
    ObservationalAcousticSummary,
    SemanticEvaluationSummary,
    SessionSummary,
    StimulusActionState,
    UiStatusKind,
)
from services.operator_console.formatters import build_acoustic_detail_display
from services.operator_console.state import OperatorStore, StimulusUiContext
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.viewmodels.live_session_vm import LiveSessionViewModel
from services.operator_console.views.live_session_view import (
    _NARROW_PHONE_PREVIEW_HEIGHT,
    _WIDE_PHONE_PREVIEW_HEIGHT,
    LiveSessionView,
    _StartSessionDialog,
)
from services.operator_console.widgets.responsive_layout import ResponsiveWidthBand

pytestmark = pytest.mark.usefixtures("qt_app")


_NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)


def _session(
    session_id: UUID | None = None,
    *,
    is_calibrating: bool | None = None,
    calibration_frames_accumulated: int | None = None,
    calibration_frames_required: int | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id or uuid4(),
        status="active",
        started_at_utc=_NOW,
        active_arm="greeting_v7",
        expected_greeting="hei rakas",
        is_calibrating=is_calibrating,
        calibration_frames_accumulated=calibration_frames_accumulated,
        calibration_frames_required=calibration_frames_required,
    )


def _encounter(
    encounter_id: str,
    *,
    state: EncounterState = EncounterState.COMPLETED,
    semantic_gate: int | None = 1,
    semantic_confidence: float | None = 0.9,
    p90: float | None = 0.42,
    gated_reward: float | None = 0.42,
    frames: int | None = 150,
    session_id: UUID | None = None,
    observational_acoustic: ObservationalAcousticSummary | None = None,
    semantic_evaluation: SemanticEvaluationSummary | None = None,
    attribution: AttributionSummary | None = None,
    segment_timestamp_utc: datetime = _NOW,
    transcription: str | None = None,
) -> EncounterSummary:
    return EncounterSummary(
        encounter_id=encounter_id,
        session_id=session_id or uuid4(),
        segment_timestamp_utc=segment_timestamp_utc,
        state=state,
        active_arm="greeting_v7",
        expected_greeting="hei rakas",
        semantic_gate=semantic_gate,
        semantic_confidence=semantic_confidence,
        p90_intensity=p90,
        gated_reward=gated_reward,
        n_frames_in_window=frames,
        au12_baseline_pre=0.1,
        transcription=transcription,
        observational_acoustic=observational_acoustic,
        semantic_evaluation=semantic_evaluation,
        attribution=attribution,
    )


def _valid_acoustic(**overrides: Any) -> ObservationalAcousticSummary:
    values: dict[str, Any] = {
        "f0_valid_measure": True,
        "f0_valid_baseline": True,
        "perturbation_valid_measure": True,
        "perturbation_valid_baseline": True,
        "voiced_coverage_measure_s": 3.25,
        "voiced_coverage_baseline_s": 2.75,
        "f0_mean_measure_hz": 220.0,
        "f0_mean_baseline_hz": 200.0,
        "f0_delta_semitones": 1.65,
        "jitter_mean_measure": 0.0123,
        "jitter_mean_baseline": 0.0100,
        "jitter_delta": 0.0023,
        "shimmer_mean_measure": 0.0456,
        "shimmer_mean_baseline": 0.0500,
        "shimmer_delta": -0.0044,
    }
    values.update(overrides)
    return ObservationalAcousticSummary(**values)


def _build_view() -> tuple[LiveSessionView, OperatorStore, LiveSessionViewModel]:
    store = OperatorStore()
    model = EncountersTableModel()
    vm = LiveSessionViewModel(store, model)
    view = LiveSessionView(vm)
    return view, store, vm


def test_live_session_view_shows_empty_state_without_session() -> None:
    view, _store, _vm = _build_view()
    # Offscreen QPA: `isVisible()` is False until the parent chain is shown.
    # `isHidden()` reads the local flag and tells us the page was wired to
    # show the empty state rather than the body container.
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is True  # type: ignore[attr-defined]


def test_live_session_view_ttv_waiting_for_device_uses_instructional_empty_state() -> None:
    view, _store, _vm = _build_view()
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is True  # type: ignore[attr-defined]
    assert "Setup not ready" in view._empty_state._title.text()  # type: ignore[attr-defined]


def test_live_session_view_ttv_gate_shows_connected_capture_status() -> None:
    view, store, _vm = _build_view()
    store.set_health(
        HealthSnapshot(
            generated_at_utc=_NOW,
            overall_state=HealthState.OK,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="adb",
                    label="Android Device Bridge",
                    state=HealthState.OK,
                ),
                HealthSubsystemStatus(
                    subsystem_key="audio_capture",
                    label="Audio Capture",
                    state=HealthState.OK,
                ),
                HealthSubsystemStatus(
                    subsystem_key="video_capture",
                    label="Video Capture",
                    state=HealthState.OK,
                ),
            ],
        )
    )

    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is True  # type: ignore[attr-defined]
    assert "Setup not ready" in view._empty_state._title.text()  # type: ignore[attr-defined]
    assert "Audio capture healthy" in view._empty_state._message.text()  # type: ignore[attr-defined]
    assert "Video capture healthy" in view._empty_state._message.text()  # type: ignore[attr-defined]


def test_live_session_view_ttv_waiting_for_face_shows_muted_dashboard_overlay() -> None:
    view, store, _vm = _build_view()
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=12,
        calibration_frames_required=45,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    assert view._empty_state.isHidden() is True  # type: ignore[attr-defined]
    assert view._body_container.isHidden() is False  # type: ignore[attr-defined]
    assert view._setup_overlay.isHidden() is False  # type: ignore[attr-defined]
    assert view._phone_preview.isEnabled() is False  # type: ignore[attr-defined]
    assert "visible face" in view._setup_overlay._message.text()  # type: ignore[attr-defined]
    assert "12/45 face frames" in view._setup_overlay._detail.text()  # type: ignore[attr-defined]
    preview_status = view._phone_preview._status.text()  # type: ignore[attr-defined]
    assert "Raw phone frames are not shown" in preview_status
    assert "12/45 face frames" in preview_status


def test_live_session_view_ttv_ready_shows_dashboard_and_smile_timeline() -> None:
    view, store, _vm = _build_view()
    assert view._readiness_panel.accessibleName() == "Live Session next step"  # type: ignore[attr-defined]
    assert view._telemetry_panel.accessibleName() == "Live response ticker"  # type: ignore[attr-defined]
    assert view._phone_preview.accessibleName() == "Live visual status"  # type: ignore[attr-defined]
    assert view._cause_effect_panel.accessibleName() == "Observed response"  # type: ignore[attr-defined]
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=45,
        calibration_frames_required=45,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    store.set_encounters([_encounter("e-smile", session_id=session.session_id, p90=0.64)])

    assert view._empty_state.isHidden() is True  # type: ignore[attr-defined]
    assert view._scroll.isHidden() is False  # type: ignore[attr-defined]
    assert view._body_container.isHidden() is False  # type: ignore[attr-defined]
    assert view._setup_overlay.isHidden() is True  # type: ignore[attr-defined]
    assert view._phone_preview.isEnabled() is True  # type: ignore[attr-defined]
    assert "Healthy" in view._phone_preview._status.text()  # type: ignore[attr-defined]
    assert "Preview placeholder" not in view._phone_preview._placeholder.text()  # type: ignore[attr-defined]
    assert view._live_analytics_notice.isHidden() is True  # type: ignore[attr-defined]
    assert view._smile_card._primary.text() == "64%"  # type: ignore[attr-defined]
    assert view._timeline_model.rowCount() == 1  # type: ignore[attr-defined]


def test_live_session_view_applies_narrow_width_responsive_layout() -> None:
    view, store, _vm = _build_view()
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=45,
        calibration_frames_required=45,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    store.set_encounters([_encounter("e-smile", session_id=session.session_id, p90=0.64)])

    view.resize(900, 640)
    view._apply_responsive_layout()  # type: ignore[attr-defined]

    assert view._dashboard_grid.column_count() == 2  # type: ignore[attr-defined]
    assert view._phone_preview._placeholder.minimumHeight() == _WIDE_PHONE_PREVIEW_HEIGHT  # type: ignore[attr-defined]
    assert view._detail_panel._reward_grid.column_count() == 2  # type: ignore[attr-defined]
    assert view._detail_panel._acoustic_grid.column_count() == 2  # type: ignore[attr-defined]
    assert view._detail_panel._semantic_grid.column_count() == 2  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(2) is True  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(6) is False  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(7) is True  # type: ignore[attr-defined]
    header = view._table.horizontalHeader()  # type: ignore[attr-defined]
    assert header.sectionResizeMode(4) == QHeaderView.ResizeMode.ResizeToContents
    assert view._timeline.current_width_band() == ResponsiveWidthBand.MEDIUM  # type: ignore[attr-defined]


def test_live_session_view_applies_extra_narrow_column_policy_and_preview_height() -> None:
    view, store, _vm = _build_view()
    session = _session(
        is_calibrating=True,
        calibration_frames_accumulated=45,
        calibration_frames_required=45,
    )
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    store.set_encounters([_encounter("e-smile", session_id=session.session_id, p90=0.64)])

    view.resize(620, 640)
    view._apply_responsive_layout()  # type: ignore[attr-defined]

    assert view._dashboard_grid.column_count() == 1  # type: ignore[attr-defined]
    assert view._phone_preview._placeholder.minimumHeight() == _NARROW_PHONE_PREVIEW_HEIGHT  # type: ignore[attr-defined]
    assert view._detail_panel._reward_grid.column_count() == 1  # type: ignore[attr-defined]
    assert view._detail_panel._acoustic_grid.column_count() == 1  # type: ignore[attr-defined]
    assert view._detail_panel._semantic_grid.column_count() == 1  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(2) is True  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(6) is True  # type: ignore[attr-defined]
    assert view._table.isColumnHidden(7) is True  # type: ignore[attr-defined]
    assert view._timeline.current_width_band() == ResponsiveWidthBand.NARROW  # type: ignore[attr-defined]


def test_live_session_view_ready_shows_no_producer_notice_without_error_banner() -> None:
    view, store, _vm = _build_view()
    session = _session(is_calibrating=True)
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    store.set_health(
        HealthSnapshot(
            generated_at_utc=_NOW,
            overall_state=HealthState.DEGRADED,
            degraded_count=1,
            subsystems=[
                HealthSubsystemStatus(
                    subsystem_key="adb",
                    label="Android Device Bridge",
                    state=HealthState.OK,
                ),
                HealthSubsystemStatus(
                    subsystem_key="gpu_ml_worker",
                    label="GPU ML Worker",
                    state=HealthState.OK,
                ),
                HealthSubsystemStatus(
                    subsystem_key="live_analytics_producer",
                    label="Live Analytics Producer",
                    state=HealthState.DEGRADED,
                ),
            ],
        )
    )

    assert view._empty_state.isHidden() is True  # type: ignore[attr-defined]
    assert view._body_container.isHidden() is False  # type: ignore[attr-defined]
    assert view._setup_overlay.isHidden() is True  # type: ignore[attr-defined]
    assert view._error_banner.isHidden() is True  # type: ignore[attr-defined]
    assert view._live_analytics_notice.isHidden() is False  # type: ignore[attr-defined]
    notice_text = view._live_analytics_notice._message.text()  # type: ignore[attr-defined]
    assert "Waiting for the first result" in notice_text
    assert view._smile_card._primary.text() == "—"  # type: ignore[attr-defined]
    assert view._smile_card._secondary.text() == "Waiting for first result"  # type: ignore[attr-defined]


def test_live_session_view_header_shows_session_status_without_repeating_action_context() -> None:
    view, store, _vm = _build_view()
    store.set_live_session(_session())
    panel = view._session_panel
    assert panel.accessibleName() == "Session controls"  # type: ignore[attr-defined]
    assert "Session" in panel._session_label.text()  # type: ignore[attr-defined]
    assert "Session" in panel._session_label.accessibleDescription()  # type: ignore[attr-defined]
    assert "active" in panel._session_meta_label.text()  # type: ignore[attr-defined]
    assert "active" in panel._session_meta_label.accessibleDescription()  # type: ignore[attr-defined]
    assert panel._start_button.accessibleName() == "Start new session"  # type: ignore[attr-defined]
    assert "connected capture source" in panel._start_button.accessibleDescription()  # type: ignore[attr-defined]
    assert panel._end_button.accessibleName() == "End session"  # type: ignore[attr-defined]
    assert panel._calibration_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]
    assert panel._calibration_pill.text() == "Healthy"  # type: ignore[attr-defined]


def test_live_session_view_header_shows_calibration_progress() -> None:
    view, store, _vm = _build_view()
    store.set_live_session(
        _session(
            is_calibrating=True,
            calibration_frames_accumulated=12,
            calibration_frames_required=45,
        )
    )
    panel = view._session_panel
    assert panel._calibration_pill.kind() == UiStatusKind.PROGRESS  # type: ignore[attr-defined]
    assert panel._calibration_pill.text() == "Preparing response baseline · 12/45 face frames"  # type: ignore[attr-defined]


def test_live_session_view_header_ready_at_safe_submit_threshold() -> None:
    view, store, _vm = _build_view()
    store.set_live_session(
        _session(
            is_calibrating=True,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
        )
    )
    panel = view._session_panel
    assert panel._calibration_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]
    assert panel._calibration_pill.text() == "Healthy"  # type: ignore[attr-defined]


def test_start_session_dialog_uses_source_summary_and_experiment_picker() -> None:
    _view, _store, vm = _build_view()
    dialog = _StartSessionDialog(
        source_summary=vm.start_session_source_summary(),
        summaries=[
            ExperimentSummary(experiment_id="exp-a", label="Experiment A", arm_count=2),
            ExperimentSummary(experiment_id="greeting_line_v1", label=None, arm_count=3),
        ],
        current_experiment_id="greeting_line_v1",
        disabled_reason=None,
        validator=vm.validate_start_session_inputs,
    )

    assert dialog.accessibleName() == "Start new session"
    assert "connected-phone capture" in dialog.accessibleDescription()
    assert "connected Android device" in dialog._source_summary.text()  # type: ignore[attr-defined]
    assert dialog._source_summary.accessibleName() == "Session source"  # type: ignore[attr-defined]
    assert dialog._experiment_label.buddy() is dialog._experiment_picker  # type: ignore[attr-defined]
    assert dialog._experiment_picker.count() == 2  # type: ignore[attr-defined]
    assert dialog._experiment_picker.currentData() == "greeting_line_v1"  # type: ignore[attr-defined]
    assert dialog._experiment_picker.accessibleName() == "Experiment"  # type: ignore[attr-defined]
    assert "stimulus strategies" in dialog._experiment_picker.accessibleDescription()  # type: ignore[attr-defined]
    assert dialog._start_button.isEnabled() is True  # type: ignore[attr-defined]
    assert dialog._start_button.accessibleName() == "Start session"  # type: ignore[attr-defined]
    assert "selected experiment" in dialog._start_button.accessibleDescription()  # type: ignore[attr-defined]
    assert dialog.values() == "greeting_line_v1"


def test_start_session_dialog_disables_start_without_experiments() -> None:
    _view, _store, vm = _build_view()
    dialog = _StartSessionDialog(
        source_summary=vm.start_session_source_summary(),
        summaries=[],
        current_experiment_id=None,
        disabled_reason=vm.start_session_disabled_reason(),
        validator=vm.validate_start_session_inputs,
    )

    assert dialog._start_button.isEnabled() is False  # type: ignore[attr-defined]
    assert "experiments page" in dialog._validation_label.text().lower()  # type: ignore[attr-defined]
    assert "experiments page" in dialog._validation_label.accessibleDescription().lower()  # type: ignore[attr-defined]


def test_live_session_view_start_button_dispatches_modal_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _store, vm = _build_view()
    captured: list[str] = []

    def fake_start(experiment_id: str) -> None:
        captured.append(experiment_id)

    monkeypatch.setattr(vm, "start_new_session", fake_start)

    class _DialogStub:
        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def values(self) -> str:
            return "greeting_line_v1"

    monkeypatch.setattr(view, "_create_start_session_dialog", lambda: _DialogStub())

    view._session_panel._start_button.click()  # type: ignore[attr-defined]
    assert captured == ["greeting_line_v1"]


def test_live_session_view_end_button_only_shows_for_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, store, vm = _build_view()
    assert view._session_panel._end_button.isHidden() is True  # type: ignore[attr-defined]

    session = _session()
    store.set_live_session(session)
    assert view._session_panel._end_button.isHidden() is False  # type: ignore[attr-defined]
    assert view._session_panel._end_button.isEnabled() is True  # type: ignore[attr-defined]

    ended = session.model_copy(update={"status": "ended", "ended_at_utc": _NOW})
    store.set_live_session(ended)
    assert view._session_panel._end_button.isHidden() is True  # type: ignore[attr-defined]

    calls: list[str] = []

    def fake_end_current_session() -> None:
        calls.append("end")

    monkeypatch.setattr(vm, "end_current_session", fake_end_current_session)
    store.set_live_session(session)
    view._session_panel._end_button.click()  # type: ignore[attr-defined]
    assert calls == ["end"]


def test_live_session_view_detail_pane_shows_reward_explanation() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                semantic_evaluation=SemanticEvaluationSummary(
                    reasoning="cross_encoder_high_match",
                    is_match=True,
                    confidence_score=0.91,
                    semantic_method="cross_encoder",
                    semantic_method_version="ce-v1",
                ),
                attribution=AttributionSummary(
                    finality="offline_final",
                    soft_reward_candidate=0.77,
                    au12_baseline_pre=0.10,
                    au12_lift_p90=0.55,
                    au12_lift_peak=0.70,
                    au12_peak_latency_ms=1250.0,
                    sync_peak_corr=0.401,
                    sync_peak_lag=3,
                    outcome_link_lag_s=15.0,
                ),
                transcription="hello welcome to the stream",
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert detail._p90_card._title.text() == "Strongest response signal"  # type: ignore[attr-defined]
    assert "0.420" in detail._p90_card._primary.text()  # type: ignore[attr-defined]
    assert "0.420" in detail._reward_card._primary.text()  # type: ignore[attr-defined]
    assert "150" in detail._frames_card._primary.text()  # type: ignore[attr-defined]
    assert "Response signal × stimulus confirmation" in detail._explanation.text()  # type: ignore[attr-defined]
    assert detail._transcription_title.text() == "Speech-to-text"  # type: ignore[attr-defined]
    assert "hello welcome" in detail._transcription_text.text()  # type: ignore[attr-defined]
    assert detail._semantic_title.text() == "Stimulus confirmation and follow-up signals"  # type: ignore[attr-defined]
    assert detail._semantic_empty.isHidden() is True  # type: ignore[attr-defined]
    assert "local stimulus confirmation checker" in detail._semantic_method_pill.text()  # type: ignore[attr-defined]
    assert detail._semantic_match_pill.text() == "match"  # type: ignore[attr-defined]
    assert "Stimulus was clearly confirmed" in detail._semantic_reason_label.text()  # type: ignore[attr-defined]
    assert detail._title.text() == "Stimulus-response detail"  # type: ignore[attr-defined]
    assert detail._confidence_card._title.text() == "Stimulus confirmation confidence"  # type: ignore[attr-defined]
    assert detail._confidence_card._primary.text() == "91%"  # type: ignore[attr-defined]
    assert "expected response" in detail._confidence_card._secondary.text()  # type: ignore[attr-defined]
    assert "offline final" in detail._attribution_finality_pill.text()  # type: ignore[attr-defined]
    assert detail._soft_reward_card._primary.text() == "possible follow-up reward 0.770"  # type: ignore[attr-defined]
    assert "strong response lift 0.550" in detail._au12_lifts_card._primary.text()  # type: ignore[attr-defined]
    assert detail._peak_latency_card._primary.text() == "1.25s"  # type: ignore[attr-defined]
    assert "movement together +0.401" in detail._synchrony_card._primary.text()  # type: ignore[attr-defined]
    assert detail._outcome_link_lag_card._primary.text() == "after 15.0s"  # type: ignore[attr-defined]
    assert "do not change the reward" in detail._semantic_observational_note.text()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_renders_acoustic_metrics_and_explanation() -> None:
    view, store, vm = _build_view()
    session = _session()
    acoustic = _valid_acoustic()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                observational_acoustic=acoustic,
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert vm.selected_acoustic() == acoustic
    assert "F0 windows" in vm.acoustic_explanation()
    assert detail._acoustic_title.text() == "Voice signal details"  # type: ignore[attr-defined]
    assert detail._acoustic_empty.isHidden() is True  # type: ignore[attr-defined]
    assert detail._acoustic_metrics_container.isHidden() is False  # type: ignore[attr-defined]
    assert detail._f0_validity_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]
    assert detail._perturbation_validity_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]
    assert "measure 220.0 Hz" in detail._f0_mean_card._primary.text()  # type: ignore[attr-defined]
    assert "baseline 200.0 Hz" in detail._f0_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert "+1.65 st" in detail._f0_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert "measure 0.0123" in detail._jitter_mean_card._primary.text()  # type: ignore[attr-defined]
    assert "baseline 0.0100" in detail._jitter_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert "+0.0023" in detail._jitter_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert "measure 0.0456" in detail._shimmer_mean_card._primary.text()  # type: ignore[attr-defined]
    assert "-0.0044" in detail._shimmer_mean_card._secondary.text()  # type: ignore[attr-defined]
    expected = build_acoustic_detail_display(acoustic)
    assert detail._f0_mean_card._primary.text() == expected.f0_mean.primary  # type: ignore[attr-defined]
    assert detail._voiced_coverage_label.text() == expected.voiced_coverage_text  # type: ignore[attr-defined]
    assert "measure 3.25s" in detail._voiced_coverage_label.text()  # type: ignore[attr-defined]
    assert "baseline 2.75s" in detail._voiced_coverage_label.text()  # type: ignore[attr-defined]
    assert "F0 Δ +1.65 st" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]
    assert "Jitter Δ +0.0023" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]


def test_live_session_view_renders_zero_acoustic_values_as_measured() -> None:
    view, store, vm = _build_view()
    session = _session()
    acoustic = _valid_acoustic(
        f0_mean_measure_hz=0.0,
        f0_mean_baseline_hz=0.0,
        f0_delta_semitones=0.0,
        jitter_mean_measure=0.0,
        jitter_mean_baseline=0.0,
        jitter_delta=0.0,
        shimmer_mean_measure=0.0,
        shimmer_mean_baseline=0.0,
        shimmer_delta=0.0,
    )
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                observational_acoustic=acoustic,
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert detail._acoustic_empty.isHidden() is True  # type: ignore[attr-defined]
    assert detail._f0_mean_card._primary.text() == "measure 0.0 Hz"  # type: ignore[attr-defined]
    assert "Δ +0.00 st" in detail._f0_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert detail._jitter_mean_card._primary.text() == "measure 0.0000"  # type: ignore[attr-defined]
    assert "Δ +0.0000" in detail._jitter_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert detail._shimmer_mean_card._primary.text() == "measure 0.0000"  # type: ignore[attr-defined]
    assert "F0 Δ +0.00 st" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]
    assert "Jitter Δ +0.0000" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]
    assert "Shimmer Δ +0.0000" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]


def test_live_session_view_acoustic_detail_uses_same_fallback_encounter_without_selection() -> None:
    view, store, _vm = _build_view()
    session = _session()
    acoustic_on_latest_completed = _valid_acoustic(
        f0_mean_measure_hz=333.0,
        f0_mean_baseline_hz=300.0,
        f0_delta_semitones=1.82,
    )
    acoustic_on_detail_row = _valid_acoustic(
        f0_mean_measure_hz=111.0,
        f0_mean_baseline_hz=100.0,
        f0_delta_semitones=1.80,
        jitter_delta=0.0011,
        shimmer_delta=-0.0011,
    )
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "latest-completed",
                session_id=session.session_id,
                observational_acoustic=acoustic_on_latest_completed,
                segment_timestamp_utc=_NOW + timedelta(seconds=1),
                p90=0.91,
                gated_reward=0.91,
            ),
            _encounter(
                "detail-row",
                session_id=session.session_id,
                observational_acoustic=acoustic_on_detail_row,
                segment_timestamp_utc=_NOW,
                p90=0.12,
                gated_reward=0.12,
            ),
        ]
    )

    detail = view._detail_panel
    expected = build_acoustic_detail_display(acoustic_on_latest_completed)
    assert "latest-completed" in detail._subtitle.text()  # type: ignore[attr-defined]
    assert "0.910" in detail._p90_card._primary.text()  # type: ignore[attr-defined]
    assert "0.910" in detail._explanation.text()  # type: ignore[attr-defined]
    assert "0.120" not in detail._explanation.text()  # type: ignore[attr-defined]
    assert detail._f0_mean_card._primary.text() == expected.f0_mean.primary  # type: ignore[attr-defined]
    assert detail._acoustic_explanation.text() == expected.explanation  # type: ignore[attr-defined]
    assert "333.0 Hz" in detail._acoustic_explanation.text()  # type: ignore[attr-defined]


def test_live_session_view_default_detail_uses_newest_acoustic_row() -> None:
    view, store, _vm = _build_view()
    session = _session()
    acoustic = _valid_acoustic(f0_mean_measure_hz=280.0)
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "new-with-voice",
                session_id=session.session_id,
                observational_acoustic=acoustic,
                segment_timestamp_utc=_NOW + timedelta(seconds=30),
            ),
            _encounter(
                "old-no-voice",
                session_id=session.session_id,
                observational_acoustic=None,
                segment_timestamp_utc=_NOW,
            ),
        ]
    )

    detail = view._detail_panel
    assert "new-with-voice" in detail._subtitle.text()  # type: ignore[attr-defined]
    assert detail._acoustic_empty.isHidden() is True  # type: ignore[attr-defined]
    assert "measure 280.0 Hz" in detail._f0_mean_card._primary.text()  # type: ignore[attr-defined]


def test_live_session_view_acoustic_validity_pills_update_independently() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                observational_acoustic=ObservationalAcousticSummary(
                    f0_valid_measure=False,
                    f0_valid_baseline=False,
                    perturbation_valid_measure=True,
                    perturbation_valid_baseline=True,
                ),
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert detail._f0_validity_pill.kind() == UiStatusKind.INFO  # type: ignore[attr-defined]
    assert "not measured" in detail._f0_validity_pill.text()  # type: ignore[attr-defined]
    assert "invalid" in detail._f0_validity_pill.text()  # type: ignore[attr-defined]
    assert detail._perturbation_validity_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]

    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                observational_acoustic=ObservationalAcousticSummary(
                    f0_valid_measure=True,
                    f0_valid_baseline=True,
                    perturbation_valid_measure=False,
                    perturbation_valid_baseline=True,
                ),
            )
        ]
    )

    assert detail._f0_validity_pill.kind() == UiStatusKind.OK  # type: ignore[attr-defined]
    assert detail._perturbation_validity_pill.kind() == UiStatusKind.INFO  # type: ignore[attr-defined]
    assert "measure perturbation window invalid" in detail._perturbation_validity_pill.text()  # type: ignore[attr-defined]


def test_live_session_view_acoustic_validity_pills_show_absent_state() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                observational_acoustic=ObservationalAcousticSummary(),
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert detail._f0_validity_pill.kind() == UiStatusKind.NEUTRAL  # type: ignore[attr-defined]
    assert detail._perturbation_validity_pill.kind() == UiStatusKind.NEUTRAL  # type: ignore[attr-defined]
    assert "absent" in detail._f0_validity_pill.text().lower()  # type: ignore[attr-defined]
    assert "absent" in detail._perturbation_validity_pill.text().lower()  # type: ignore[attr-defined]
    assert detail._f0_mean_card._primary.text() == "measure —"  # type: ignore[attr-defined]
    assert "baseline —" in detail._f0_mean_card._secondary.text()  # type: ignore[attr-defined]
    assert "measure —" in detail._voiced_coverage_label.text()  # type: ignore[attr-defined]


def test_live_session_view_acoustic_empty_state_without_error_banner() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                session_id=session.session_id,
                semantic_evaluation=SemanticEvaluationSummary(
                    reasoning="gray_band_llm_nonmatch",
                    is_match=False,
                    confidence_score=0.63,
                    semantic_method="llm_gray_band",
                    semantic_method_version="gray-v1",
                ),
                attribution=None,
            )
        ]
    )
    vm.select_encounter("e1")

    detail = view._detail_panel
    assert detail._acoustic_empty.text() == "No voice signal details for this segment"  # type: ignore[attr-defined]
    assert detail._acoustic_empty.isHidden() is False  # type: ignore[attr-defined]
    assert detail._acoustic_metrics_container.isHidden() is True  # type: ignore[attr-defined]
    assert view._error_banner.isHidden() is True  # type: ignore[attr-defined]
    assert detail._semantic_empty.text() == "No follow-up signal details for this encounter"  # type: ignore[attr-defined]
    assert detail._semantic_empty.isHidden() is False  # type: ignore[attr-defined]
    assert "backup stimulus confirmation checker" in detail._semantic_method_pill.text()  # type: ignore[attr-defined]
    assert detail._semantic_match_pill.text() == "non-match"  # type: ignore[attr-defined]
    assert detail._soft_reward_card.isHidden() is True  # type: ignore[attr-defined]
    assert "do not change the reward" in detail._semantic_observational_note.text()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_flags_zero_frames() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                state=EncounterState.REJECTED_NO_FRAMES,
                frames=0,
                p90=None,
                gated_reward=None,
                session_id=session.session_id,
            )
        ]
    )
    vm.select_encounter("e1")
    detail = view._detail_panel
    assert "No usable face frames" in detail._explanation.text()  # type: ignore[attr-defined]


def test_live_session_view_detail_pane_flags_gate_closed() -> None:
    view, store, vm = _build_view()
    session = _session()
    store.set_live_session(session)
    store.set_encounters(
        [
            _encounter(
                "e1",
                semantic_gate=0,
                gated_reward=0.0,
                session_id=session.session_id,
            )
        ]
    )
    vm.select_encounter("e1")
    detail = view._detail_panel
    assert "stimulus was not confirmed" in detail._explanation.text().lower()  # type: ignore[attr-defined]


def test_live_session_view_countdown_timer_activates_on_measuring() -> None:
    view, store, _vm = _build_view()
    session = _session()
    store.set_live_session(session)
    # Anchor the stimulus clock to wall-clock "now" so the result window
    # is in the future and the 1s tick does not auto-stop itself on the
    # zero-remaining boundary.
    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=datetime.now(UTC),
        )
    )
    assert view._countdown_timer.isActive() is True
    assert not hasattr(view._detail_panel, "_countdown_label")  # type: ignore[attr-defined]
    assert not hasattr(view._detail_panel, "_progress_label")  # type: ignore[attr-defined]


def test_live_session_view_countdown_timer_stops_when_not_measuring() -> None:
    view, store, _vm = _build_view()
    session = _session()
    store.set_live_session(session)
    # Start measuring, then transition to COMPLETED — timer must stop.
    store.set_stimulus_ui_context(
        StimulusUiContext(
            state=StimulusActionState.MEASURING,
            authoritative_stimulus_time_utc=_NOW,
        )
    )
    store.set_stimulus_ui_context(StimulusUiContext(state=StimulusActionState.COMPLETED))
    assert view._countdown_timer.isActive() is False


def test_live_session_view_on_activated_does_not_crash_without_session() -> None:
    view, _store, _vm = _build_view()
    # Page may be activated before a session is selected — must be a no-op
    # that leaves the empty state visible.
    view.on_activated()
    assert view._empty_state.isHidden() is False  # type: ignore[attr-defined]
