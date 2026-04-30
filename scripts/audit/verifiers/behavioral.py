"""Behavioral §13 audit verifiers backed by pytest audit_item markers."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from scripts.audit.registry import (
    AuditContext,
    AuditRegistry,
    AuditVerifier,
    get_default_registry,
    register_audit_verifier,
)
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item

BEHAVIORAL_AUDIT_ITEM_IDS: tuple[str, ...] = (
    *(f"13.{index}" for index in range(16, 30)),
    "13.31",
)
_PYTEST_NO_TESTS_COLLECTED = 5
_PYTEST_TIMEOUT_S = 600
_MAX_OUTPUT_CHARS = 4_000


def _format_marker(item_id: str) -> str:
    return f'@pytest.mark.audit_item("{item_id}")'


def _is_audit_item_call(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "audit_item"
    )


def _audit_item_args(decorator: ast.expr) -> tuple[str, ...]:
    if not _is_audit_item_call(decorator):
        return ()
    assert isinstance(decorator, ast.Call)

    item_ids: list[str] = []
    for arg in decorator.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            item_ids.append(arg.value)
    keyword_item_id = next(
        (
            keyword.value
            for keyword in decorator.keywords
            if keyword.arg == "item_id"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ),
        None,
    )
    if isinstance(keyword_item_id, ast.Constant) and isinstance(keyword_item_id.value, str):
        item_ids.append(keyword_item_id.value)
    return tuple(item_ids)


def discover_audit_item_markers(tests_root: Path) -> Mapping[str, tuple[str, ...]]:
    """Return audit_item marker bindings discovered under ``tests_root``.

    Discovery is keyed by the marker argument, not by the marker name alone, so
    completeness checks fail when a specific §13.N value is missing.
    """
    discovered: dict[str, list[str]] = {}
    if not tests_root.exists():
        return {}

    for path in sorted(tests_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        relative_path = path.relative_to(tests_root.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                for item_id in _audit_item_args(decorator):
                    discovered.setdefault(item_id, []).append(
                        f"{relative_path}:{node.lineno}:{node.name}"
                    )

    return {item_id: tuple(locations) for item_id, locations in sorted(discovered.items())}


def _pytest_command(
    item_id: str,
    *,
    collect_only: bool,
    test_paths: Sequence[str] | None = None,
) -> list[str]:
    selected_paths = list(test_paths or ("tests",))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--strict-markers",
        *selected_paths,
        "--audit-item",
        item_id,
    ]
    if collect_only:
        command.extend(["--collect-only", "-q"])
    else:
        command.append("-q")
    return command


def _run_pytest_for_audit_item(
    repo_root: Path,
    item_id: str,
    *,
    collect_only: bool,
    test_paths: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath_entries = [str(repo_root), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(entry for entry in pythonpath_entries if entry)
    return subprocess.run(
        _pytest_command(item_id, collect_only=collect_only, test_paths=test_paths),
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=_PYTEST_TIMEOUT_S,
        env=env,
        check=False,
    )


def _collected_nodeids(stdout: str) -> tuple[str, ...]:
    nodeids: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and not stripped.startswith(("ERROR", "FAILED")):
            nodeids.append(stripped)
    return tuple(nodeids)


def _trimmed_pytest_output(result: subprocess.CompletedProcess[str]) -> str:
    combined = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    ).strip()
    if not combined:
        return "(pytest produced no output)"
    if len(combined) <= _MAX_OUTPUT_CHARS:
        return combined
    return "..." + combined[-_MAX_OUTPUT_CHARS:]


def verify_behavioral_item(context: AuditContext, item: Section13Item) -> AuditResult:
    """Run the pytest tests bound to one behavioral §13 audit item.

    §13.27 has both static semantic-registry/threshold evidence and pytest
    marker evidence.  Its verifier composes both surfaces in a deterministic
    order so registration order cannot drop either half of the audit item.
    """

    def marker_evidence_result() -> AuditResult:
        if item.item_id not in BEHAVIORAL_AUDIT_ITEM_IDS:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=False,
                evidence=f"§{item.item_id} is not an in-scope behavioral audit item.",
                follow_up=(
                    "Dispatch behavioral verifier only for registered behavioral items: "
                    + ", ".join(f"§{item_id}" for item_id in BEHAVIORAL_AUDIT_ITEM_IDS)
                    + "."
                ),
            )

        marker = _format_marker(item.item_id)
        discovered = discover_audit_item_markers(context.repo_root / "tests")
        binding_locations = discovered.get(item.item_id, ())
        if not binding_locations:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=False,
                evidence=f"No tests discovered for {marker} under tests/.",
                follow_up=f"Add at least one test under tests/ marked {marker}.",
            )

        test_paths = tuple(
            sorted({location.split(":", maxsplit=1)[0] for location in binding_locations})
        )
        collect_result = _run_pytest_for_audit_item(
            context.repo_root,
            item.item_id,
            collect_only=True,
            test_paths=test_paths,
        )
        collected_nodeids = _collected_nodeids(collect_result.stdout)
        if collect_result.returncode == _PYTEST_NO_TESTS_COLLECTED or not collected_nodeids:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=False,
                evidence=f"No tests collected for {marker} from {', '.join(test_paths)}.",
                follow_up=f"Add at least one test under tests/ marked {marker}.",
            )
        if collect_result.returncode != 0:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=False,
                evidence=(
                    f"pytest collection failed for {marker} with exit code "
                    f"{collect_result.returncode}: {_trimmed_pytest_output(collect_result)}"
                ),
                follow_up="Fix collection errors before running the behavioral audit verifier.",
            )

        run_result = _run_pytest_for_audit_item(
            context.repo_root,
            item.item_id,
            collect_only=False,
            test_paths=test_paths,
        )
        if run_result.returncode == 0:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=True,
                evidence=(
                    f"{len(collected_nodeids)} selected test(s) passed for {marker}: "
                    + ", ".join(collected_nodeids)
                ),
            )

        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=False,
            evidence=(
                f"Selected tests for {marker} failed with exit code {run_result.returncode}: "
                f"{_trimmed_pytest_output(run_result)}"
            ),
            follow_up=f"All tests selected by {marker} must pass.",
        )

    if item.item_id != "13.27":
        return marker_evidence_result()

    from scripts.audit.verifiers.mechanical import verify_semantic_reason_codes

    surface_results = (
        (
            "mechanical semantic registry/threshold surface",
            verify_semantic_reason_codes(context, item),
        ),
        ("behavioral marker/evidence surface", marker_evidence_result()),
    )
    passed = all(result.passed for _, result in surface_results)
    evidence = "\n\n".join(
        f"{'PASS' if result.passed else 'FAIL'} {surface_name}: {result.evidence}"
        for surface_name, result in surface_results
    )
    if passed:
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=True,
            evidence=evidence,
        )

    failing_followups = [
        f"{surface_name}: {result.follow_up or result.evidence}"
        for surface_name, result in surface_results
        if not result.passed
    ]
    return AuditResult(
        item_id=item.item_id,
        title=item.title,
        passed=False,
        evidence=evidence,
        follow_up=(
            "Resolve failing §13.27 verifier surface(s): "
            + " | ".join(failing_followups)
        ),
    )


BEHAVIORAL_VERIFIERS: Mapping[str, AuditVerifier] = {
    item_id: verify_behavioral_item for item_id in BEHAVIORAL_AUDIT_ITEM_IDS
}


def register_behavioral_verifiers(
    *,
    registry: AuditRegistry | None = None,
    item_ids: Iterable[str] | None = None,
) -> None:
    """Register behavioral verifiers for in-scope §13 items.

    ``item_ids`` lets the audit harness avoid adding registry entries for items
    not present in a synthetic or older checklist while still using the shared
    ``register_audit_verifier`` declaration surface for the default registry.
    """
    requested_item_ids = tuple(item_ids) if item_ids is not None else BEHAVIORAL_AUDIT_ITEM_IDS
    target_registry = registry if registry is not None else get_default_registry()

    for item_id in requested_item_ids:
        verifier = BEHAVIORAL_VERIFIERS.get(item_id)
        if verifier is None or target_registry.has_verifier(item_id):
            continue
        if registry is None:
            register_audit_verifier(item_id)(verifier)
        else:
            target_registry.register(item_id)(verifier)
