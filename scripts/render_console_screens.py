"""Render Operator Console views to PNG via offscreen Qt.

Headless screenshot harness for the running Operator Console. Boots a
real `MainWindow` with a real `OperatorStore`, seeds the store with
realistic DTOs, and grabs every route at three responsive widths so an
AI agent (or human reviewer) can compare the rendered visuals against
the design-system HTML reference.

Why this exists: Qt's QSS is a strict subset of CSS, and the design
system mockups are HTML/CSS. The two diverge in the small — text
cutoff, hover states, font metrics — and the only reliable way to
catch the divergence is to render the actual widgets.

Usage::

    uv run python scripts/render_console_screens.py
    # screenshots land in /tmp/console-screens/<route>-<width>.png

Set `LSIE_SCREENSHOT_DIR` to override the destination.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

# Force Qt offscreen *before* importing anything that builds a QApplication —
# the platform plugin is selected at QApplication construction time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Offscreen QPA on Windows registers zero system fonts by default and
# every glyph renders as a tofu box. Pointing QT_QPA_FONTDIR at the
# Windows font directory restores ~170+ families so the screenshots
# render legibly. Caller may override.
if sys.platform == "win32" and "QT_QPA_FONTDIR" not in os.environ:
    _windows_fontdir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if _windows_fontdir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(_windows_fontdir)

# Repo root on sys.path so `packages.*` and `services.*` resolve when the
# script runs directly (`uv run python scripts/render_console_screens.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload  # noqa: E402
from packages.schemas.operator_console import (  # noqa: E402
    AlertEvent,
    AlertKind,
    AlertSeverity,
    ArmSummary,
    AttributionSummary,
    CoModulationSummary,
    EncounterState,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthProbeState,
    HealthSnapshot,
    HealthState,
    HealthSubsystemProbe,
    HealthSubsystemStatus,
    LatestEncounterSummary,
    OverviewSnapshot,
    PhysiologyCurrentSnapshot,
    SemanticEvaluationSummary,
    SessionPhysiologySnapshot,
    SessionSummary,
)
from services.operator_console.api_client import ApiClient  # noqa: E402
from services.operator_console.config import OperatorConsoleConfig  # noqa: E402
from services.operator_console.design_system import (  # noqa: E402
    install_application_stylesheet,
)
from services.operator_console.polling import PollingCoordinator  # noqa: E402
from services.operator_console.state import AppRoute, OperatorStore  # noqa: E402
from services.operator_console.views.main_window import MainWindow  # noqa: E402

_NOW = datetime.now(UTC).replace(microsecond=0)
_SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
_WIDTHS: tuple[tuple[str, int, int], ...] = (
    ("narrow", 720, 720),
    ("medium", 1024, 768),
    ("wide", 1440, 900),
)
_ROUTES: tuple[AppRoute, ...] = (
    AppRoute.OVERVIEW,
    AppRoute.LIVE_SESSION,
    AppRoute.EXPERIMENTS,
    AppRoute.PHYSIOLOGY,
    AppRoute.HEALTH,
    AppRoute.SESSIONS,
)


def _stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(text=text),
        expected_stimulus_rule="Deliver the spoken greeting to the creator",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


def _config() -> OperatorConsoleConfig:
    return OperatorConsoleConfig(
        api_base_url="http://localhost:8000",
        api_timeout_seconds=2.0,
        environment_label="screens",
        overview_poll_ms=5_000,
        session_header_poll_ms=5_000,
        live_encounters_poll_ms=5_000,
        experiments_poll_ms=5_000,
        physiology_poll_ms=5_000,
        comodulation_poll_ms=5_000,
        health_poll_ms=5_000,
        alerts_poll_ms=5_000,
        sessions_poll_ms=5_000,
        default_experiment_id="exp-greeting",
    )


def _session() -> SessionSummary:
    return SessionSummary(
        session_id=_SESSION_ID,
        status="active",
        started_at_utc=_NOW - timedelta(minutes=12),
        active_arm="greeting_v1",
        expected_response_text="hei rakas",
        duration_s=720.0,
        is_calibrating=False,
        latest_reward=0.62,
        latest_semantic_gate=1,
    )


def _encounter(session_id: UUID, *, p90: float = 0.62, gate: int = 1) -> EncounterSummary:
    return EncounterSummary(
        encounter_id=f"e-{p90:.2f}",
        session_id=session_id,
        segment_timestamp_utc=_NOW - timedelta(seconds=8),
        state=EncounterState.COMPLETED,
        active_arm="greeting_v1",
        expected_response_text="hei rakas",
        stimulus_time_utc=_NOW - timedelta(seconds=12),
        semantic_gate=gate,
        semantic_confidence=0.91,
        observed_response_text="hei rakas",
        p90_intensity=p90,
        gated_reward=p90 * gate,
        n_frames_in_window=140,
        au12_baseline_pre=0.18,
        semantic_evaluation=SemanticEvaluationSummary(
            reasoning="cross_encoder_high_match",
            is_match=True,
            confidence_score=0.91,
            semantic_method="cross_encoder",
            semantic_method_version="ce-v1",
        ),
        attribution=AttributionSummary(
            finality="online_provisional",
            soft_reward_candidate=0.42,
            au12_baseline_pre=0.18,
            au12_lift_p90=0.41,
            au12_lift_peak=0.55,
            au12_peak_latency_ms=1240.0,
            sync_peak_corr=0.18,
            sync_peak_lag=2,
        ),
        physiology_attached=True,
        physiology_stale=False,
    )


def _overview(session: SessionSummary) -> OverviewSnapshot:
    return OverviewSnapshot(
        generated_at_utc=_NOW,
        active_session=session,
        latest_encounter=LatestEncounterSummary(
            encounter_id="e-latest",
            session_id=session.session_id,
            segment_timestamp_utc=_NOW - timedelta(seconds=8),
            state=EncounterState.COMPLETED,
            semantic_gate=1,
            p90_intensity=0.62,
            gated_reward=0.62,
            n_frames_in_window=140,
            semantic_evaluation=SemanticEvaluationSummary(
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=0.91,
                semantic_method="cross_encoder",
                semantic_method_version="ce-v1",
            ),
            attribution=AttributionSummary(finality="online_provisional"),
        ),
        experiment_summary=ExperimentSummary(
            experiment_id="exp-greeting",
            label="greeting line v1",
            active_arm_id="greeting_v1",
            arm_count=3,
            latest_reward=0.62,
        ),
        physiology=_physiology_snapshot(session.session_id, stale=False),
        health=_health_snapshot(),
    )


def _physiology_snapshot(session_id: UUID, *, stale: bool) -> SessionPhysiologySnapshot:
    return SessionPhysiologySnapshot(
        session_id=session_id,
        streamer=PhysiologyCurrentSnapshot(
            subject_role="streamer",
            rmssd_ms=46.0,
            heart_rate_bpm=78,
            is_stale=stale,
            freshness_s=12.0 if not stale else 180.0,
            source_timestamp_utc=_NOW - timedelta(seconds=12 if not stale else 180),
            provider="apple_watch",
        ),
        operator=PhysiologyCurrentSnapshot(
            subject_role="operator",
            rmssd_ms=52.0,
            heart_rate_bpm=68,
            is_stale=False,
            freshness_s=8.0,
            source_timestamp_utc=_NOW - timedelta(seconds=8),
            provider="polar_h10",
        ),
        comodulation=CoModulationSummary(
            session_id=session_id,
            co_modulation_index=0.42,
            n_paired_observations=18,
            coverage_ratio=0.82,
            window_start_utc=_NOW - timedelta(minutes=2),
            window_end_utc=_NOW,
        ),
        generated_at_utc=_NOW,
    )


def _health_snapshot() -> HealthSnapshot:
    return HealthSnapshot(
        generated_at_utc=_NOW,
        overall_state=HealthState.DEGRADED,
        degraded_count=1,
        recovering_count=1,
        error_count=0,
        subsystems=[
            HealthSubsystemStatus(
                subsystem_key="orchestrator",
                label="Orchestrator",
                state=HealthState.OK,
                detail="all probes healthy",
            ),
            HealthSubsystemStatus(
                subsystem_key="ml_backend",
                label="ML backend",
                state=HealthState.RECOVERING,
                recovery_mode="warmup",
                detail="reloading whisper model",
            ),
            HealthSubsystemStatus(
                subsystem_key="azure_openai",
                label="Azure OpenAI",
                state=HealthState.DEGRADED,
                detail="elevated retry rate",
                operator_action_hint="watch the next minute",
            ),
        ],
        subsystem_probes={
            "orchestrator": HealthSubsystemProbe(
                subsystem_key="orchestrator",
                label="Orchestrator",
                state=HealthProbeState.OK,
                latency_ms=14.0,
                detail="health endpoint responsive",
                checked_at_utc=_NOW,
            ),
            "ml_backend": HealthSubsystemProbe(
                subsystem_key="ml_backend",
                label="ML backend",
                state=HealthProbeState.OK,
                latency_ms=42.0,
                detail="model warmed up",
                checked_at_utc=_NOW,
            ),
            "azure_openai": HealthSubsystemProbe(
                subsystem_key="azure_openai",
                label="Azure OpenAI",
                state=HealthProbeState.NOT_CONFIGURED,
                latency_ms=None,
                detail="missing AZURE_OPENAI_ENDPOINT",
                checked_at_utc=_NOW,
            ),
        },
    )


def _alerts() -> list[AlertEvent]:
    return [
        AlertEvent(
            alert_id="a-1",
            severity=AlertSeverity.WARNING,
            kind=AlertKind.SUBSYSTEM_DEGRADED,
            message="Azure OpenAI retry rate elevated",
            emitted_at_utc=_NOW - timedelta(minutes=2),
        ),
        AlertEvent(
            alert_id="a-2",
            severity=AlertSeverity.INFO,
            kind=AlertKind.SUBSYSTEM_RECOVERING,
            message="ML backend warming up",
            emitted_at_utc=_NOW - timedelta(seconds=40),
        ),
    ]


def _experiment_detail() -> ExperimentDetail:
    return ExperimentDetail(
        experiment_id="exp-greeting",
        label="greeting line v1",
        active_arm_id="greeting_v1",
        arms=[
            ArmSummary(
                arm_id="greeting_v1",
                stimulus_definition=_stimulus_definition("hei rakas"),
                posterior_alpha=42.0,
                posterior_beta=12.0,
                evaluation_variance=0.018,
                selection_count=54,
                recent_reward_mean=0.62,
                recent_semantic_pass_rate=0.78,
                enabled=True,
            ),
            ArmSummary(
                arm_id="greeting_v2",
                stimulus_definition=_stimulus_definition("moi rakas"),
                posterior_alpha=18.0,
                posterior_beta=22.0,
                evaluation_variance=0.034,
                selection_count=40,
                recent_reward_mean=0.41,
                recent_semantic_pass_rate=0.55,
                enabled=True,
            ),
            ArmSummary(
                arm_id="greeting_v3",
                stimulus_definition=_stimulus_definition("hei kulta"),
                posterior_alpha=6.0,
                posterior_beta=4.0,
                evaluation_variance=0.07,
                selection_count=10,
                recent_reward_mean=None,
                recent_semantic_pass_rate=None,
                enabled=True,
            ),
        ],
        last_update_summary="arm greeting_v1 updated posterior to α=42, β=12",
        last_updated_utc=_NOW - timedelta(seconds=30),
    )


def _seed_store(store: OperatorStore) -> None:
    session = _session()
    store.set_selected_session_id(session.session_id)
    store.set_live_session(session)
    store.set_overview(_overview(session))
    store.set_sessions([session])
    store.set_encounters([_encounter(session.session_id, p90=p) for p in (0.62, 0.41, 0.18)])
    store.set_experiment(_experiment_detail())
    store.set_experiment_summaries(
        [
            ExperimentSummary(
                experiment_id="exp-greeting",
                label="greeting line v1",
                active_arm_id="greeting_v1",
                arm_count=3,
                latest_reward=0.62,
            ),
        ]
    )
    store.set_physiology(_physiology_snapshot(session.session_id, stale=False))
    store.set_health(_health_snapshot())
    store.set_alerts(_alerts())


def _output_dir() -> Path:
    raw = os.environ.get("LSIE_SCREENSHOT_DIR")
    if raw:
        return Path(raw)
    return Path("/tmp/console-screens")


def render_all() -> int:
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    app = cast(QApplication, QApplication.instance() or QApplication([]))
    app.setApplicationName("LSIE-MLF Operator Console")
    app.setOrganizationName("LSIE-MLF")
    # Offscreen QPA on Windows ships with a stub font that renders ASCII
    # as tofu boxes; pick a guaranteed-rendering family so the screenshots
    # are legible.
    fallback_family = next(
        (
            family
            for family in ("Segoe UI", "Arial", "DejaVu Sans", "Liberation Sans")
            if family in QFontDatabase.families()
        ),
        QFontDatabase.families()[0] if QFontDatabase.families() else "",
    )
    if fallback_family:
        app.setFont(QFont(fallback_family, 10))
    install_application_stylesheet(app)

    config = _config()
    store = OperatorStore()
    client = ApiClient(config.api_base_url, config.api_timeout_seconds)
    coordinator = PollingCoordinator(config, client, store)
    window = MainWindow(config, store, coordinator)
    _seed_store(store)

    rendered: list[Path] = []
    for route in _ROUTES:
        store.set_route(route)
        for label, width, height in _WIDTHS:
            window.resize(width, height)
            window.show()
            for _ in range(4):
                app.processEvents()
            pixmap = window.grab()
            target = out_dir / f"{route.value}-{label}-{width}x{height}.png"
            pixmap.save(str(target))
            rendered.append(target)

    # Drop the coordinator's threads (it never started, but `stop()` is
    # idempotent and tidy).
    coordinator.stop()
    window.close()
    app.processEvents()

    print(f"Rendered {len(rendered)} screens into {out_dir}")
    for path in rendered:
        print(f"  {path}")
    return 0


def main() -> int:
    return render_all()


if __name__ == "__main__":
    sys.exit(main())
