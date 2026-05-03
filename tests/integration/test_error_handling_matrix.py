"""Structural enforcement for the committed §12 error-handling registry."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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

DISCOVERY_ARTIFACTS = (
    "docs/SPEC_REFERENCE.md",
    "services/desktop_app/drift.py",
    "services/desktop_app/processes/capture_supervisor.py",
    "services/api/routes/physiology.py",
    "services/worker/pipeline/ground_truth.py",
    "services/worker/pipeline/orchestrator.py",
    "services/worker/pipeline/video_capture.py",
    "services/worker/pipeline/analytics.py",
    "services/worker/tasks/inference.py",
    "services/worker/tasks/enrichment.py",
    "packages/ml_core/semantic.py",
    "packages/ml_core/transcription.py",
)

CATEGORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "Network Disconnection": (
        "network disconnection",
        "network disconnect",
        "websocket disconnect",
        "backoff",
        "http 503",
        "db write failed",
        "db_retry_interval",
        "empty scrape result",
        "semantic_timeout",
        "semantic_error",
    ),
    "Hardware Device Loss": (
        "hardware device loss",
        "hardware loss",
        "device loss",
        "usb",
        "adb",
        "gpu",
        "cuda",
        "drift freeze",
        "freeze drift",
    ),
    "Worker Process Crash": (
        "worker process crash",
        "worker crash",
        "process crash",
        "browser crash",
        "pipeline break",
        "restart",
        "self.retry",
        "max_retries",
        "ffmpeg exited",
        "persist_metrics",
        "task failure",
    ),
    "Queue Overload": (
        "queue overload",
        "deque",
        "maxlen",
        "overflow",
        "csv fallback",
        "celery queue",
        "redis-server",
        "concurrency=1",
        "acks_late",
        "physio:events",
    ),
}


@dataclass(frozen=True, slots=True)
class HandlerEvidence:
    """Concrete source-derived evidence for one §12 module/category relationship."""

    module: str
    category: str
    path: str
    line: int
    mode: str
    detail: str

    @property
    def relationship(self) -> tuple[str, str]:
        return (self.module, self.category)

    def describe(self) -> str:
        return f"{self.path}:{self.line} [{self.mode}] {self.detail}"


@pytest.mark.audit_item("13.11")
def test_error_handling_registry_rows_have_handler_evidence() -> None:
    """The §12 registry must exactly equal source-discovered handler relationships."""
    registry = _load_registry(REGISTRY_PATH)
    rows = list(_iter_failure_rows(registry))

    expected = int(registry.get("row_count_expected", 0))
    assert len(rows) == expected == 24

    registry_relationships = {(row["module"], row["category"]) for row in rows}
    discovered_evidence, discovery_errors = _discover_handler_relationships()
    assert not discovery_errors, "\n".join(discovery_errors)
    discovered_relationships = set(discovered_evidence)

    rows_by_relationship = {(row["module"], row["category"]): row for row in rows}
    missing = sorted(registry_relationships - discovered_relationships)
    extra = sorted(discovered_relationships - registry_relationships)
    assert not missing and not extra, _format_relationship_diff(
        missing=missing,
        extra=extra,
        discovered_evidence=discovered_evidence,
        registry_rows_by_relationship=rows_by_relationship,
    )

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


def _format_relationship_diff(
    *,
    missing: Iterable[tuple[str, str]],
    extra: Iterable[tuple[str, str]],
    discovered_evidence: Mapping[tuple[str, str], list[HandlerEvidence]],
    registry_rows_by_relationship: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    parts: list[str] = []
    missing_list = list(missing)
    extra_list = list(extra)
    if missing_list:
        parts.append("Registry rows without matching source-discovered handler relationship:")
        for module, category in missing_list:
            row = registry_rows_by_relationship.get((module, category), {})
            evidence = row.get("evidence", [])
            parts.append(f"  - Module {module} / {category}: registry evidence={evidence!r}")
    if extra_list:
        parts.append(
            "Source-discovered implementation handler relationships missing from registry:"
        )
        for module, category in extra_list:
            details = "; ".join(
                item.describe() for item in discovered_evidence.get((module, category), [])[:3]
            )
            parts.append(f"  - Module {module} / {category}: {details or 'no evidence detail'}")
    return "\n".join(parts)


def _discover_handler_relationships() -> tuple[
    dict[tuple[str, str], list[HandlerEvidence]], list[str]
]:
    """Walk implementation artifacts and infer §12 relationships from source anchors.

    The registry is not used as a relationship catalog here.  The only expected
    structure baked into the test is the spec's module index (A-F) and four
    standardized failure categories.  Relationships are emitted from source
    comments/docstrings, AST-local try/except handlers, and documented
    module-owned degradation anchors with file/line evidence.
    """

    discovered: dict[tuple[str, str], list[HandlerEvidence]] = {}
    errors: list[str] = []
    for rel_path in DISCOVERY_ARTIFACTS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing discovery artifact: {rel_path}")
            continue
        source = path.read_text(encoding="utf-8")
        for evidence in _scan_documented_matrix(rel_path, source):
            _add_handler_evidence(discovered, evidence)
        for evidence in _scan_section_annotations(rel_path, source):
            _add_handler_evidence(discovered, evidence)
        if path.suffix == ".py":
            for evidence in _scan_ast_try_except_annotations(rel_path, source):
                _add_handler_evidence(discovered, evidence)
    return discovered, errors


def _add_handler_evidence(
    discovered: dict[tuple[str, str], list[HandlerEvidence]], evidence: HandlerEvidence
) -> None:
    if (evidence.module, evidence.category) in EXPECTED_SECTIONS:
        discovered.setdefault(evidence.relationship, []).append(evidence)


def _scan_documented_matrix(rel_path: str, source: str) -> Iterator[HandlerEvidence]:
    if rel_path != "docs/SPEC_REFERENCE.md":
        return
    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        category = _category_from_documented_matrix_line(stripped)
        if category is None:
            continue
        for match in re.finditer(r"Module\s+([A-F])\s*=\s*([^;]+)", stripped):
            module = match.group(1)
            detail = match.group(0).strip()
            yield HandlerEvidence(
                module=module,
                category=category,
                path=rel_path,
                line=line_no,
                mode="documented_degradation",
                detail=detail,
            )


def _category_from_documented_matrix_line(line: str) -> str | None:
    normalized = line.lower().lstrip("- ")
    for category in CATEGORIES:
        prefix = category.lower().replace("hardware device loss", "hardware device loss")
        if normalized.startswith(prefix + ":"):
            return category
    return None


def _scan_section_annotations(rel_path: str, source: str) -> Iterator[HandlerEvidence]:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "§12" not in line:
            continue
        window = "\n".join(lines[index : min(len(lines), index + 6)])
        for module, category, reason in _infer_relationships_from_text(window):
            yield HandlerEvidence(
                module=module,
                category=category,
                path=rel_path,
                line=index + 1,
                mode="section_annotation",
                detail=reason,
            )


def _scan_ast_try_except_annotations(rel_path: str, source: str) -> Iterator[HandlerEvidence]:
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        raise AssertionError(f"Cannot parse Python discovery artifact {rel_path}: {exc}") from exc
    for function in (
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        try_nodes = [node for node in ast.walk(function) if isinstance(node, ast.Try)]
        if not try_nodes:
            continue
        function_source = ast.get_source_segment(source, function) or ""
        if "§12" not in function_source:
            continue
        for module, category, reason in _infer_relationships_from_text(function_source):
            handler_names = sorted(
                {
                    name
                    for try_node in try_nodes
                    for handler in try_node.handlers
                    for name in _exception_names(handler.type)
                }
            )
            yield HandlerEvidence(
                module=module,
                category=category,
                path=rel_path,
                line=function.lineno,
                mode="ast_try_except",
                detail=f"{function.name} handlers={handler_names!r}; {reason}",
            )


def _infer_relationships_from_text(text: str) -> set[tuple[str, str, str]]:
    relationships: set[tuple[str, str, str]] = set()
    for section_match in re.finditer(r"§\s*12\.(\d)\.(\d)", text):
        module_index = int(section_match.group(1))
        category_index = int(section_match.group(2))
        if 1 <= module_index <= len(MODULE_LETTERS) and 1 <= category_index <= len(CATEGORIES):
            relationships.add(
                (
                    MODULE_LETTERS[module_index - 1],
                    CATEGORIES[category_index - 1],
                    f"explicit section {section_match.group(0)}",
                )
            )

    lowered = text.lower()
    modules = set(re.findall(r"Module\s+([A-F])\b", text))
    for alias_match in re.finditer(r"\b([A-F])\b", text):
        around = lowered[max(0, alias_match.start() - 40) : alias_match.end() + 40]
        has_category_alias = any(
            alias in around for aliases in CATEGORY_ALIASES.values() for alias in aliases
        )
        if "§12" in around or has_category_alias:
            modules.add(alias_match.group(1))

    categories = {
        category
        for category, aliases in CATEGORY_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    }
    if "§12.4" in lowered and "acoustic" in lowered:
        modules.add("D")
    if modules and categories:
        for module in modules:
            for category in categories:
                relationships.add((module, category, "module/category phrase in §12-local text"))
    return relationships


def _evidence_matches(evidence: Mapping[str, Any]) -> tuple[bool, str]:
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

    mode = evidence.get("mode")
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
