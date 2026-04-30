"""Unit coverage for structural audit verifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.audit import AuditContext, Section13Item
from scripts.audit.verifiers import structural
from scripts.audit.verifiers.structural import PytestRun


def _write_registry(repo_root: Path, relative_path: Path, payload: Any) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _context(repo_root: Path) -> AuditContext:
    return AuditContext(repo_root=repo_root, spec_content={})


def test_error_handling_verifier_reports_success_row_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = {
        "modules": [
            {"module": "A", "failures": [{"category": "Network"}, {"category": "Hardware"}]},
            {"module": "B", "failures": [{"category": "Network"}]},
        ]
    }
    _write_registry(tmp_path, structural.ERROR_HANDLING_REGISTRY_PATH, registry)
    observed: list[tuple[Path, Path]] = []

    def fake_run(repo_root: Path, test_path: Path) -> PytestRun:
        observed.append((repo_root, test_path))
        return PytestRun(0, "1 passed", "")

    monkeypatch.setattr(structural, "_run_pytest", fake_run)
    item = Section13Item("13.11", "Error handling", "All four failure categories implemented.")

    result = structural.verify_error_handling_registry(_context(tmp_path), item)

    assert result.passed is True
    assert result.item_id == "13.11"
    assert "rows=3" in result.evidence
    assert str(structural.ERROR_HANDLING_TEST_PATH) in result.evidence
    assert observed == [(tmp_path, structural.ERROR_HANDLING_TEST_PATH)]


def test_error_handling_verifier_surfaces_delegate_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = {"modules": [{"module": "F", "failures": [{"category": "Queue Overload"}]}]}
    _write_registry(tmp_path, structural.ERROR_HANDLING_REGISTRY_PATH, registry)

    monkeypatch.setattr(
        structural,
        "_run_pytest",
        lambda repo_root, test_path: PytestRun(
            1,
            "Module F / Queue Overload missing handler evidence",
            "",
        ),
    )
    item = Section13Item("13.11", "Error handling", "All four failure categories implemented.")

    result = structural.verify_error_handling_registry(_context(tmp_path), item)

    assert result.passed is False
    assert "rows=1" in result.evidence
    assert "Module F / Queue Overload missing handler evidence" in result.evidence
    assert result.follow_up is not None


def test_error_handling_verifier_reports_registry_parse_failure(tmp_path: Path) -> None:
    path = tmp_path / structural.ERROR_HANDLING_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not: [json", encoding="utf-8")
    item = Section13Item("13.11", "Error handling", "All four failure categories implemented.")

    result = structural.verify_error_handling_registry(_context(tmp_path), item)

    assert result.passed is False
    assert str(structural.ERROR_HANDLING_REGISTRY_PATH) in result.evidence
    assert "not parseable" in result.evidence


def test_variable_traceability_verifier_reports_success_row_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = {
        "variables": [
            {"variable": "Encoded Video Stream"},
            {"variable": "Raw Audio PCM"},
            {"variable": "External Context Metadata"},
        ]
    }
    _write_registry(tmp_path, structural.VARIABLE_TRACEABILITY_REGISTRY_PATH, registry)
    observed: list[tuple[Path, Path]] = []

    def fake_run(repo_root: Path, test_path: Path) -> PytestRun:
        observed.append((repo_root, test_path))
        return PytestRun(0, ".", "")

    monkeypatch.setattr(structural, "_run_pytest", fake_run)
    item = Section13Item("13.13", "Variable traceability", "Variables are produced.")

    result = structural.verify_variable_traceability_registry(_context(tmp_path), item)

    assert result.passed is True
    assert result.item_id == "13.13"
    assert "rows=3" in result.evidence
    assert str(structural.VARIABLE_TRACEABILITY_TEST_PATH) in result.evidence
    assert observed == [(tmp_path, structural.VARIABLE_TRACEABILITY_TEST_PATH)]


def test_variable_traceability_verifier_surfaces_delegate_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = {"variables": [{"variable": "External Context Metadata"}]}
    _write_registry(tmp_path, structural.VARIABLE_TRACEABILITY_REGISTRY_PATH, registry)

    monkeypatch.setattr(
        structural,
        "_run_pytest",
        lambda repo_root, test_path: PytestRun(
            1,
            "External Context Metadata / Module F missing public output schema",
            "pytest stderr",
        ),
    )
    item = Section13Item("13.13", "Variable traceability", "Variables are produced.")

    result = structural.verify_variable_traceability_registry(_context(tmp_path), item)

    assert result.passed is False
    assert "rows=1" in result.evidence
    assert "External Context Metadata / Module F missing public output schema" in result.evidence
    assert "pytest stderr" in result.evidence
    assert result.follow_up is not None


def test_variable_traceability_verifier_reports_missing_registry(tmp_path: Path) -> None:
    item = Section13Item("13.13", "Variable traceability", "Variables are produced.")

    result = structural.verify_variable_traceability_registry(_context(tmp_path), item)

    assert result.passed is False
    assert str(structural.VARIABLE_TRACEABILITY_REGISTRY_PATH) in result.evidence
    assert "Missing registry" in result.evidence
