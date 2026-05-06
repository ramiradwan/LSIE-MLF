"""Structural enforcement for the committed §12 error-handling registry."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "registries" / "error_handling_matrix.yaml"
CATEGORIES = (
    "Network Disconnection",
    "Hardware Device Loss",
    "Worker Process Crash",
    "Queue Overload",
)
MODULE_LETTERS = ("A", "B", "C", "D", "E", "F")
EXPECTED_SECTIONS = {
    (module, category): f"§12.{module_index}.{category_index}"
    for module_index, module in enumerate(MODULE_LETTERS, start=1)
    for category_index, category in enumerate(CATEGORIES, start=1)
}


@pytest.mark.audit_item("13.11")
def test_error_handling_registry_rows_have_handler_evidence() -> None:
    """The §12 registry must exactly equal source-discovered handler relationships."""
    registry = _load_registry(REGISTRY_PATH)
    rows = list(_iter_failure_rows(registry))

    expected = int(registry.get("row_count_expected", 0))
    assert len(rows) == expected == 24

    evidence_errors: list[str] = []
    for row in rows:
        evidence_items = row.get("evidence")
        if not isinstance(evidence_items, list) or not evidence_items:
            evidence_errors.append(_row_id(row) + " has no evidence entries")
            continue
        for evidence in evidence_items:
            if not isinstance(evidence, Mapping):
                evidence_errors.append(_row_id(row) + f" invalid evidence object: {evidence!r}")
                continue
            ok, reason = _evidence_matches(evidence)
            if not ok:
                evidence_errors.append(_row_id(row) + " evidence mismatch: " + reason)

    assert not evidence_errors, "\n".join(evidence_errors)


def _load_registry(path: Path) -> dict[str, Any]:
    assert path.exists(), f"Missing registry: {path}"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # JSON is a valid YAML subset for this registry.
        raise AssertionError(f"Registry is not parseable JSON/YAML subset: {path}: {exc}") from exc
    assert isinstance(loaded, dict), "Registry root must be an object"
    assert loaded.get("registry_ref") == "§12 Error Handling and Recovery Specification"
    assert loaded.get("audit_item") == "§13.11 Error handling"
    return loaded


def _iter_failure_rows(registry: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    modules = registry.get("modules")
    assert isinstance(modules, list) and modules, "Registry must enumerate modules"
    seen: set[tuple[str, str]] = set()
    for module in modules:
        assert isinstance(module, Mapping), f"Invalid module row: {module!r}"
        letter = module.get("module")
        assert letter in set(MODULE_LETTERS)
        failures = module.get("failures")
        assert isinstance(failures, list), f"Module {letter} must list failures"
        assert len(failures) == 4, f"Module {letter} must enumerate all four §12 categories"
        for failure in failures:
            assert isinstance(failure, dict)
            category = failure.get("category")
            section = failure.get("section")
            assert isinstance(category, str) and category in CATEGORIES
            assert section == EXPECTED_SECTIONS[(str(letter), category)], _row_id(failure, letter)
            key = (str(letter), category)
            assert key not in seen, f"Duplicate row: {key}"
            seen.add(key)
            row = dict(failure)
            row["module"] = letter
            yield row


def _row_id(row: Mapping[str, Any], module: object | None = None) -> str:
    letter = module if module is not None else row.get("module", "?")
    return f"Module {letter} / {row.get('category', '?')} ({row.get('section', '?')})"


def _evidence_matches(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    mode = evidence.get("mode")
    if mode == "not_applicable":
        reason = evidence.get("reason")
        if not isinstance(reason, str) or not reason:
            return False, "not_applicable evidence requires a reason"
        return True, reason

    rel_path = evidence.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        return False, "evidence missing path"
    path = REPO_ROOT / rel_path
    if not path.exists():
        return False, f"{rel_path} does not exist"
    source = path.read_text(encoding="utf-8")
    patterns = evidence.get("patterns", [])
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        return False, f"{rel_path} patterns must be strings"

    if mode == "ast_try_except":
        if path.suffix != ".py":
            return False, f"{rel_path} requested AST inspection but is not Python"
        exceptions = evidence.get("exceptions", [])
        if not isinstance(exceptions, list) or not all(
            isinstance(item, str) for item in exceptions
        ):
            return False, f"{rel_path} exceptions must be strings"
        qualname = evidence.get("qualname")
        if not isinstance(qualname, str) or not qualname:
            return False, f"{rel_path} AST evidence requires a qualname locator"
        return _has_matching_try_except(
            source,
            qualname=qualname,
            expected_exceptions=set(exceptions),
            patterns=patterns,
            rel_path=rel_path,
        )
    if mode == "section_comment":
        missing_patterns = [pattern for pattern in patterns if pattern not in source]
        if missing_patterns:
            return False, f"{rel_path} missing co-located section patterns {missing_patterns!r}"
        if "§" not in source:
            return False, f"{rel_path} has no §-annotated comment/docstring"
    elif mode == "documented_degradation":
        missing_patterns = [pattern for pattern in patterns if pattern not in source]
        if missing_patterns:
            return False, f"{rel_path} missing documented degradation patterns {missing_patterns!r}"
    else:
        return False, f"{rel_path} has unsupported evidence mode {mode!r}"

    return True, f"{rel_path} matched"


def _has_matching_try_except(
    source: str,
    *,
    qualname: str,
    expected_exceptions: set[str],
    patterns: list[str],
    rel_path: str,
) -> tuple[bool, str]:
    tree = ast.parse(source, filename=rel_path)
    function = _qualname_node(tree, qualname)
    function_source = ast.get_source_segment(source, function) or ""
    decorator_source = "\n".join(
        ast.get_source_segment(source, decorator) or ""
        for decorator in getattr(function, "decorator_list", [])
    )
    localized_source = "\n".join((decorator_source, function_source))
    missing_patterns = [pattern for pattern in patterns if pattern not in localized_source]
    if missing_patterns:
        return False, f"{rel_path}:{qualname} missing patterns {missing_patterns!r}"

    handler_errors: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            names = _exception_names(handler.type)
            if names == expected_exceptions:
                return True, f"{rel_path}:{qualname} matched {sorted(names)!r}"
            handler_errors.append(f"handler exceptions {sorted(names)!r}")
    return (
        False,
        f"{rel_path}:{qualname} has no exact try/except for {sorted(expected_exceptions)!r}; "
        + ", ".join(handler_errors),
    )


def _qualname_node(tree: ast.Module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    parts = qualname.split(".")
    current: ast.AST = tree
    for part in parts:
        body = getattr(current, "body", [])
        for node in body:
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ):
                current = node
                break
        else:
            raise AssertionError(f"Cannot find AST qualname {qualname}")
    assert isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef))
    return current


def _exception_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return {"BaseException"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for item in node.elts:
            names.update(_exception_names(item))
        return names
    return {ast.unparse(node)}
