from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from automation.schemas.ux_plan import UxPlan, load_plan, main


def _valid_plan_payload() -> dict[str, object]:
    return {
        "spec_refs": ["§4.E.1"],
        "source_artifacts": ["docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md"],
        "page_route": "live_session",
        "shell": "SidebarStackShell",
        "viewmodel": "services.operator_console.viewmodels.live_session_vm.LiveSessionViewModel",
        "regions": [
            {
                "name": "header",
                "layout": "row",
                "components": [
                    {
                        "component": "SectionHeader",
                        "object_name": "LiveSessionHeader",
                        "props": {"title": "Live Session"},
                        "formatters": ["format_session_title"],
                        "a11y": {
                            "accessible_name": "Live session header",
                            "focusable": False,
                        },
                        "responsive": {
                            "narrow": "visible",
                            "medium": "visible",
                            "wide": "visible",
                        },
                    },
                    {
                        "component": "StatusPill",
                        "object_name": "LiveSessionReadinessPill",
                        "status_kind_source": "readiness_status_kind",
                        "a11y": {
                            "accessible_name": "Readiness status",
                            "accessible_description": "Current live-session readiness",
                        },
                    },
                ],
            }
        ],
        "target_files": [
            "services/operator_console/views/live_session_view.py",
            "services/operator_console/viewmodels/live_session_vm.py",
        ],
        "target_symbols": ["LiveSessionView", "LiveSessionViewModel"],
        "invariants": ["no inline setStyleSheet outside design_system/"],
    }


def test_ux_plan_accepts_valid_payload() -> None:
    plan = UxPlan.model_validate(_valid_plan_payload())

    assert plan.page_route == "live_session"
    assert plan.shell == "SidebarStackShell"
    assert [component.object_name for component in plan.regions[0].components] == [
        "LiveSessionHeader",
        "LiveSessionReadinessPill",
    ]


def test_ux_plan_rejects_duplicate_object_names() -> None:
    payload = _valid_plan_payload()
    regions = payload["regions"]
    assert isinstance(regions, list)
    first_region = regions[0]
    assert isinstance(first_region, dict)
    components = first_region["components"]
    assert isinstance(components, list)
    duplicate_component = {
        "component": "MetricCard",
        "object_name": "LiveSessionHeader",
        "a11y": {"accessible_name": "Duplicate card"},
    }
    components.append(duplicate_component)

    with pytest.raises(ValidationError, match="duplicate object_name"):
        UxPlan.model_validate(payload)


def test_load_plan_reads_json_file(tmp_path: Path) -> None:
    plan_path = tmp_path / "ux-plan.json"
    plan_path.write_text(json.dumps(_valid_plan_payload()), encoding="utf-8")

    plan = load_plan(plan_path)

    assert plan.target_symbols == ["LiveSessionView", "LiveSessionViewModel"]


def test_main_returns_zero_for_valid_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "ux-plan.json"
    plan_path.write_text(json.dumps(_valid_plan_payload()), encoding="utf-8")

    exit_code = main([str(plan_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "LiveSessionReadinessPill" in captured.out
