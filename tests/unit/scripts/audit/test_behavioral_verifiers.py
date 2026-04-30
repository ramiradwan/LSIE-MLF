"""Tests for behavioral §13 audit verifiers."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent

import pytest

from scripts.audit import AuditContext, AuditRegistry, AuditResult, Section13Item
from scripts.audit.verifiers import behavioral

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _item(item_id: str = "13.20") -> Section13Item:
    return Section13Item(item_id=item_id, title=f"Item {item_id}", body="Synthetic body.")


def _context(repo_root: Path) -> AuditContext:
    return AuditContext(repo_root=repo_root, spec_content={})


CompletedPytestRun = subprocess.CompletedProcess[str]


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> CompletedPytestRun:
    return subprocess.CompletedProcess(
        args=[sys.executable, "-m", "pytest"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _write_audit_item_conftest(repo_root: Path) -> None:
    tests_root = repo_root / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    (tests_root / "conftest.py").write_text(
        dedent(
            """
            from __future__ import annotations

            from typing import Any


            def pytest_addoption(parser: Any) -> None:
                parser.addoption("--audit-item", action="store", default=None)


            def pytest_configure(config: Any) -> None:
                config.addinivalue_line(
                    "markers",
                    "audit_item(item_id): bind a test to a §13 audit checklist item",
                )


            def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
                requested_item_id = config.getoption("--audit-item")
                if requested_item_id is None:
                    return
                selected = []
                deselected = []
                for item in items:
                    marker_ids = []
                    for marker in item.iter_markers(name="audit_item"):
                        marker_ids.extend(str(arg) for arg in marker.args)
                        if "item_id" in marker.kwargs:
                            marker_ids.append(str(marker.kwargs["item_id"]))
                    if str(requested_item_id) in marker_ids:
                        selected.append(item)
                    else:
                        deselected.append(item)
                if deselected:
                    config.hook.pytest_deselected(items=deselected)
                items[:] = selected
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_behavioral_registry_is_complete_for_in_scope_items() -> None:
    expected_item_ids = (*(f"13.{index}" for index in range(16, 30)), "13.31")

    assert expected_item_ids == behavioral.BEHAVIORAL_AUDIT_ITEM_IDS
    assert tuple(behavioral.BEHAVIORAL_VERIFIERS) == behavioral.BEHAVIORAL_AUDIT_ITEM_IDS

    registry = AuditRegistry()
    behavioral.register_behavioral_verifiers(registry=registry)

    assert registry.item_ids == behavioral.BEHAVIORAL_AUDIT_ITEM_IDS


def test_all_in_scope_behavioral_items_have_markers_under_tests() -> None:
    discovered = behavioral.discover_audit_item_markers(_REPO_ROOT / "tests")

    assert set(behavioral.BEHAVIORAL_AUDIT_ITEM_IDS).issubset(discovered)
    for item_id in behavioral.BEHAVIORAL_AUDIT_ITEM_IDS:
        assert discovered[item_id], f"missing @pytest.mark.audit_item({item_id!r})"


def test_marker_discovery_is_keyed_by_audit_item_argument(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_marked.py").write_text(
        dedent(
            """
            import pytest

            @pytest.mark.audit_item("13.16")
            @pytest.mark.audit_item(item_id="13.21")
            def test_multiple_items():
                assert True

            @pytest.mark.audit_item("13.20")
            class TestMarkedClass:
                def test_inherits_class_marker(self):
                    assert True
            """
        ).lstrip(),
        encoding="utf-8",
    )

    discovered = behavioral.discover_audit_item_markers(tests_root)

    assert set(discovered) == {"13.16", "13.20", "13.21"}
    assert discovered["13.16"] == ("tests/test_marked.py:5:test_multiple_items",)
    assert discovered["13.21"] == ("tests/test_marked.py:5:test_multiple_items",)
    assert discovered["13.20"] == ("tests/test_marked.py:9:TestMarkedClass",)


def test_verifier_fails_when_requested_marker_is_missing(tmp_path: Path) -> None:
    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.18"))

    assert result.passed is False
    assert "No tests discovered" in result.evidence
    assert '@pytest.mark.audit_item("13.18")' in result.evidence
    assert result.follow_up is not None


def test_verifier_fails_when_requested_marker_collects_no_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, bool]] = []

    def fake_run(
        repo_root: Path,
        item_id: str,
        *,
        collect_only: bool,
        test_paths: Sequence[str] | None = None,
    ) -> CompletedPytestRun:
        assert repo_root == tmp_path
        observed.append((item_id, collect_only))
        return _completed(5, stdout="no tests collected")

    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda tests_root: {"13.18": ("tests/test_behavior.py:1:test_bound",)},
    )
    monkeypatch.setattr(behavioral, "_run_pytest_for_audit_item", fake_run)

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.18"))

    assert observed == [("13.18", True)]
    assert result.passed is False
    assert "No tests collected" in result.evidence
    assert '@pytest.mark.audit_item("13.18")' in result.evidence
    assert result.follow_up is not None


def test_verifier_fails_when_collection_output_has_no_matching_nodeids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        repo_root: Path,
        item_id: str,
        *,
        collect_only: bool,
        test_paths: Sequence[str] | None = None,
    ) -> CompletedPytestRun:
        assert repo_root == tmp_path
        assert item_id == "13.19"
        assert collect_only is True
        return _completed(0, stdout="2 tests deselected")

    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda tests_root: {"13.19": ("tests/test_behavior.py:1:test_bound",)},
    )
    monkeypatch.setattr(behavioral, "_run_pytest_for_audit_item", fake_run)

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.19"))

    assert result.passed is False
    assert "No tests collected" in result.evidence
    assert result.follow_up is not None


def test_verifier_passes_only_after_collecting_and_passing_selected_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, bool]] = []

    def fake_run(
        repo_root: Path,
        item_id: str,
        *,
        collect_only: bool,
        test_paths: Sequence[str] | None = None,
    ) -> CompletedPytestRun:
        assert repo_root == tmp_path
        assert item_id == "13.20"
        observed.append((item_id, collect_only))
        if collect_only:
            return _completed(0, stdout="tests/test_behavior.py::test_bound\n")
        return _completed(0, stdout=".\n1 passed")

    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda tests_root: {"13.20": ("tests/test_behavior.py:1:test_bound",)},
    )
    monkeypatch.setattr(behavioral, "_run_pytest_for_audit_item", fake_run)

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.20"))

    assert observed == [("13.20", True), ("13.20", False)]
    assert result == AuditResult(
        item_id="13.20",
        title="Item 13.20",
        passed=True,
        evidence=(
            '1 selected test(s) passed for @pytest.mark.audit_item("13.20"): '
            "tests/test_behavior.py::test_bound"
        ),
    )


def test_verifier_fails_when_any_selected_test_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        repo_root: Path,
        item_id: str,
        *,
        collect_only: bool,
        test_paths: Sequence[str] | None = None,
    ) -> CompletedPytestRun:
        assert repo_root == tmp_path
        assert item_id == "13.21"
        if collect_only:
            return _completed(
                0,
                stdout=(
                    "tests/test_behavior.py::test_selected_passes\n"
                    "tests/test_behavior.py::test_selected_fails\n"
                ),
            )
        return _completed(1, stdout=".F\nFAILED tests/test_behavior.py::test_selected_fails")

    monkeypatch.setattr(
        behavioral,
        "discover_audit_item_markers",
        lambda tests_root: {"13.21": ("tests/test_behavior.py:1:test_bound",)},
    )
    monkeypatch.setattr(behavioral, "_run_pytest_for_audit_item", fake_run)

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.21"))

    assert result.passed is False
    assert "Selected tests" in result.evidence
    assert "test_selected_fails" in result.evidence
    assert result.follow_up == 'All tests selected by @pytest.mark.audit_item("13.21") must pass.'


def test_verifier_selects_by_audit_item_argument_not_marker_name(tmp_path: Path) -> None:
    _write_audit_item_conftest(tmp_path)
    (tmp_path / "tests" / "test_argument_selection.py").write_text(
        dedent(
            """
            import pytest

            @pytest.mark.audit_item("13.16")
            def test_wrong_item_fails():
                assert False

            @pytest.mark.audit_item("13.20")
            def test_requested_item_passes():
                assert True
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.20"))

    assert result.passed is True
    assert "test_requested_item_passes" in result.evidence
    assert "test_wrong_item_fails" not in result.evidence


def test_argument_selected_verifier_fails_when_selected_test_fails(tmp_path: Path) -> None:
    _write_audit_item_conftest(tmp_path)
    (tmp_path / "tests" / "test_selected_failure.py").write_text(
        dedent(
            """
            import pytest

            @pytest.mark.audit_item("13.20")
            def test_selected_passes():
                assert True

            @pytest.mark.audit_item("13.20")
            def test_selected_fails():
                assert False
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = behavioral.verify_behavioral_item(_context(tmp_path), _item("13.20"))

    assert result.passed is False
    assert "test_selected_fails" in result.evidence
    assert result.follow_up is not None
