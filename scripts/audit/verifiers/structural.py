"""Structural §13 verifiers backed by committed registries and pytest checks."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.audit.registry import AuditContext, register_audit_verifier
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item

ERROR_HANDLING_REGISTRY_PATH = Path("docs/registries/error_handling_matrix.yaml")
VARIABLE_TRACEABILITY_REGISTRY_PATH = Path("docs/registries/variable_traceability.yaml")
ERROR_HANDLING_TEST_PATH = Path("tests/integration/test_error_handling_matrix.py")
VARIABLE_TRACEABILITY_TEST_PATH = Path("tests/integration/test_variable_traceability.py")


@dataclass(frozen=True, slots=True)
class PytestRun:
    """Captured outcome from a delegated pytest invocation."""

    returncode: int
    stdout: str
    stderr: str


def _load_registry(repo_root: Path, relative_path: Path) -> dict[str, Any]:
    """Load a committed registry stored as a JSON-compatible YAML document."""
    path = repo_root / relative_path
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing registry {relative_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry {relative_path} is not parseable: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Registry {relative_path} root must be an object")
    return loaded


def _error_handling_row_count(registry: dict[str, Any]) -> int:
    modules = registry.get("modules")
    if not isinstance(modules, list):
        raise ValueError("error-handling registry missing modules list")
    count = 0
    for module in modules:
        if not isinstance(module, dict):
            raise ValueError("error-handling registry contains a non-object module row")
        failures = module.get("failures")
        if not isinstance(failures, list):
            raise ValueError(f"module {module.get('module', '?')} missing failures list")
        count += len(failures)
    return count


def _variable_traceability_row_count(registry: dict[str, Any]) -> int:
    variables = registry.get("variables")
    if not isinstance(variables, list):
        raise ValueError("variable-traceability registry missing variables list")
    return len(variables)


def _run_pytest(repo_root: Path, test_path: Path) -> PytestRun:
    """Delegate registry enforcement to the corresponding integration test."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return PytestRun(completed.returncode, completed.stdout, completed.stderr)


def _combined_output(run: PytestRun, *, max_chars: int = 4000) -> str:
    combined = "\n".join(part for part in (run.stdout.strip(), run.stderr.strip()) if part)
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


@register_audit_verifier("13.11")
def verify_error_handling_registry(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify §12 registry completeness by delegating to its integration test."""
    try:
        registry = _load_registry(context.repo_root, ERROR_HANDLING_REGISTRY_PATH)
        row_count = _error_handling_row_count(registry)
    except ValueError as exc:
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=False,
            evidence=f"{ERROR_HANDLING_REGISTRY_PATH}: {exc}",
            follow_up="Restore the committed §12 error-handling registry.",
        )

    run = _run_pytest(context.repo_root, ERROR_HANDLING_TEST_PATH)
    if run.returncode == 0:
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=True,
            evidence=(
                f"{ERROR_HANDLING_REGISTRY_PATH} rows={row_count}; delegated test "
                f"passed: {ERROR_HANDLING_TEST_PATH}"
            ),
        )

    return AuditResult(
        item_id=item.item_id,
        title=item.title,
        passed=False,
        evidence=(
            f"{ERROR_HANDLING_REGISTRY_PATH} rows={row_count}; delegated test "
            f"failed: {ERROR_HANDLING_TEST_PATH}\n{_combined_output(run)}"
        ),
        follow_up="Add the missing §12 handler evidence or repair the delegated integration test.",
    )


@register_audit_verifier("13.13")
def verify_variable_traceability_registry(
    context: AuditContext, item: Section13Item
) -> AuditResult:
    """Verify §11 registry completeness by delegating to its integration test."""
    try:
        registry = _load_registry(context.repo_root, VARIABLE_TRACEABILITY_REGISTRY_PATH)
        row_count = _variable_traceability_row_count(registry)
    except ValueError as exc:
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=False,
            evidence=f"{VARIABLE_TRACEABILITY_REGISTRY_PATH}: {exc}",
            follow_up="Restore the committed §11 variable-traceability registry.",
        )

    run = _run_pytest(context.repo_root, VARIABLE_TRACEABILITY_TEST_PATH)
    if run.returncode == 0:
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=True,
            evidence=(
                f"{VARIABLE_TRACEABILITY_REGISTRY_PATH} rows={row_count}; delegated test "
                f"passed: {VARIABLE_TRACEABILITY_TEST_PATH}"
            ),
        )

    return AuditResult(
        item_id=item.item_id,
        title=item.title,
        passed=False,
        evidence=(
            f"{VARIABLE_TRACEABILITY_REGISTRY_PATH} rows={row_count}; delegated test "
            f"failed: {VARIABLE_TRACEABILITY_TEST_PATH}\n{_combined_output(run)}"
        ),
        follow_up="Add the missing §11 producer evidence or repair the delegated integration test.",
    )
