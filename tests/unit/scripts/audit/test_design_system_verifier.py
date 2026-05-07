from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.verifiers.design_system import (
    collect_design_system_issues,
    main,
    verify_design_system_artifacts,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_repo_fixture(repo_root: Path) -> Path:
    _write(
        repo_root / "services" / "operator_console" / "design_system" / "design_system.schema.json",
        "{}\n",
    )
    _write(
        repo_root / "services" / "operator_console" / "design_system" / "design_system.json",
        json.dumps(
            {
                "$schema": "./design_system.schema.json",
                "version": "0.1.0",
                "source_audit": "../../../docs/artifacts/OPERATOR_CONSOLE_UI_UX_AUDIT.md",
                "spec_refs": ["§4.E.1"],
                "tokens_file": "tokens.json",
                "shells": [],
                "primitives": [
                    {
                        "name": "SidebarHeader",
                        "object_names": ["SidebarTitle", "SidebarSubtitle"],
                        "description": "Sidebar labels.",
                    }
                ],
                "compounds": [],
                "selectors": [
                    {"object_name": "ContentSurface", "kind": "surface"},
                    {"object_name": "SidebarNav", "kind": "shell"},
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repo_root / "services" / "operator_console" / "design_system" / "tokens.json",
        json.dumps(
            {
                "$schema": "https://www.designtokens.org/TR/drafts/format/",
                "color": {"background": {"$value": "#0f1115", "$type": "color"}},
                "status": {"ok": {"$value": "#4ecb71", "$type": "color"}},
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repo_root / "services" / "operator_console" / "design_system" / "qss_builder.py",
        (
            'STYLESHEET = """\n'
            "QWidget#ContentSurface {}\n"
            "QFrame#SidebarNav {}\n"
            "QLabel#SidebarTitle {}\n"
            "QLabel#SidebarSubtitle {}\n"
            '"""\n'
        ),
    )
    _write(
        repo_root / "services" / "operator_console" / "views" / "example_view.py",
        "from __future__ import annotations\n",
    )
    _write(
        repo_root / "services" / "operator_console" / "app.py",
        "from __future__ import annotations\n",
    )
    _write(
        repo_root / "docs" / "artifacts" / "OPERATOR_CONSOLE_UI_UX_AUDIT.md",
        "# audit\n",
    )
    return repo_root


def test_collect_design_system_issues_passes_for_clean_fixture(tmp_path: Path) -> None:
    repo_root = _build_repo_fixture(tmp_path)

    issues = collect_design_system_issues(repo_root)

    assert issues == ()
    assert verify_design_system_artifacts(repo_root) is True


def test_collect_design_system_issues_reports_inline_qss_hex_api_import_and_unregistered_selector(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_fixture(tmp_path)
    _write(
        repo_root / "services" / "operator_console" / "app.py",
        "from __future__ import annotations\n\n"
        "def install() -> None:\n"
        '    widget.setStyleSheet("color: red;")\n',
    )
    _write(
        repo_root / "services" / "operator_console" / "views" / "example_view.py",
        "from __future__ import annotations\n"
        "from services.operator_console.api_client import ApiClient\n\n"
        'BORDER = "#ffffff"\n',
    )
    _write(
        repo_root / "services" / "operator_console" / "design_system" / "qss_builder.py",
        (
            'STYLESHEET = """\n'
            "QWidget#ContentSurface {}\n"
            "QLabel#SidebarTitle {}\n"
            "QLabel#UnregisteredThing {}\n"
            '"""\n'
        ),
    )

    issues = collect_design_system_issues(repo_root)

    assert any("setStyleSheet(...) is forbidden" in issue for issue in issues)
    assert any("hex literal is forbidden" in issue for issue in issues)
    assert any("must not import api_client modules" in issue for issue in issues)
    assert any("UnregisteredThing" in issue for issue in issues)
    assert verify_design_system_artifacts(repo_root) is False


def test_main_returns_non_zero_when_design_system_checks_fail(tmp_path: Path) -> None:
    repo_root = _build_repo_fixture(tmp_path)
    _write(
        repo_root / "services" / "operator_console" / "views" / "example_view.py",
        'from __future__ import annotations\nBORDER = "#ffffff"\n',
    )

    assert main(["--repo", str(repo_root)]) == 1
