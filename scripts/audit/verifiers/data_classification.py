"""AST verifier for §5.2 data-tier annotations at persistence boundaries."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.audit.registry import AuditContext, AuditRegistry, AuditVerifier, get_default_registry
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item

_DATA_CLASSIFICATION_ITEM_ID = "13.14"
_INSERT_RE = re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE)
_MARKER_NAMES = {"mark_data_tier"}
_EXPECTED_INSERT_TIER = "PERMANENT"
_VALID_TIER_NAMES = {"TRANSIENT", "DEBUG", "PERMANENT"}

_IN_SCOPE_ROOTS: tuple[Path, ...] = (
    Path("services/api/routes"),
    Path("services/api/services"),
    Path("services/api/repos"),
    Path("services/worker/pipeline"),
)


@dataclass(frozen=True, slots=True)
class _Marker:
    tier: str | None
    spec_ref: str | None
    line: int


@dataclass(frozen=True, slots=True)
class _SqlBinding:
    sql: str
    line: int
    marker: _Marker | None


@dataclass(frozen=True, slots=True)
class _Finding:
    rel_path: str
    line: int
    message: str

    def render(self) -> str:
        return f"{self.rel_path}:{self.line}: {self.message}"


@dataclass(frozen=True, slots=True)
class DataClassificationScan:
    """Machine-readable scan output used by unit tests and AuditResult rendering."""

    insert_annotations: tuple[str, ...]
    inbound_annotations: tuple[str, ...]
    findings: tuple[_Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


def _result(
    item: Section13Item,
    scan: DataClassificationScan,
) -> AuditResult:
    if scan.passed:
        evidence = (
            f"PASS: §5.2 DataTier annotations verified for "
            f"{len(scan.insert_annotations)} persistence INSERT call(s) and "
            f"{len(scan.inbound_annotations)} inbound raw/transient boundary call(s)."
        )
        concrete = [*scan.insert_annotations[:8], *scan.inbound_annotations[:8]]
        if concrete:
            evidence += " Evidence: " + "; ".join(concrete)
        return AuditResult(
            item_id=item.item_id,
            title=item.title,
            passed=True,
            evidence=evidence,
        )

    return AuditResult(
        item_id=item.item_id,
        title=item.title,
        passed=False,
        evidence="FAIL: " + "\n".join(finding.render() for finding in scan.findings),
        follow_up=(
            "Annotate cited INSERT/inbound boundary call sites with "
            "packages.schemas.data_tiers.mark_data_tier(..., DataTier.*, spec_ref='§5.2.x') "
            "and normalize raw payload objects before Permanent Store writes."
        ),
    )


def _line(lines: Sequence[str], line_number: int) -> str:
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1].strip()
    return ""


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_marker_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (_call_name(node.func) in _MARKER_NAMES)


def _marker_from_call(node: ast.Call) -> _Marker:
    tier: str | None = None
    if len(node.args) >= 2:
        tier = _tier_name(node.args[1])
    for keyword in node.keywords:
        if keyword.arg == "tier":
            tier = _tier_name(keyword.value)
            break

    spec_ref: str | None = None
    for keyword in node.keywords:
        if keyword.arg == "spec_ref" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                spec_ref = keyword.value.value
    return _Marker(tier=tier, spec_ref=spec_ref, line=node.lineno)


def _tier_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "DataTier":
            return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        candidate = node.value.rsplit(".", maxsplit=1)[-1]
        if candidate in _VALID_TIER_NAMES:
            return candidate
    return None


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            else:
                return None
        return "".join(pieces)
    return None


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    else:
        targets.append(node.target)

    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return tuple(names)


def _collect_sql_bindings(tree: ast.AST) -> dict[str, _SqlBinding]:
    bindings: dict[str, _SqlBinding] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = _assigned_names(node)
        if not names:
            continue

        marker: _Marker | None = None
        value = node.value
        if value is None:
            continue
        sql_node = value
        if _is_marker_call(value):
            assert isinstance(value, ast.Call)
            marker = _marker_from_call(value)
            sql_node = value.args[0] if value.args else value
        sql = _string_literal(sql_node)
        if sql is None or not _INSERT_RE.search(sql):
            continue
        for name in names:
            bindings[name] = _SqlBinding(
                sql=sql,
                line=getattr(node, "lineno", 0),
                marker=marker,
            )
    return bindings


def _collect_return_markers(tree: ast.AST) -> dict[str, _Marker]:
    """Collect functions whose return value carries verifier-readable tier evidence."""

    def marker_from_data_tier_decorator(decorator: ast.expr) -> _Marker | None:
        if not isinstance(decorator, ast.Call):
            return None
        if _call_name(decorator.func) != "data_tier":
            return None

        tier: str | None = None
        if decorator.args:
            tier = _tier_name(decorator.args[0])
        for keyword in decorator.keywords:
            if keyword.arg == "tier":
                tier = _tier_name(keyword.value)
                break

        spec_ref: str | None = None
        for keyword in decorator.keywords:
            if keyword.arg == "spec_ref" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    spec_ref = keyword.value.value
        return _Marker(tier=tier, spec_ref=spec_ref, line=decorator.lineno)

    def visit_returns(statement: ast.AST, returns: list[ast.Return]) -> None:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(statement, ast.Return):
            returns.append(statement)
            return
        for child in ast.iter_child_nodes(statement):
            visit_returns(child, returns)

    markers: dict[str, _Marker] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        decorator_markers = [
            marker
            for decorator in node.decorator_list
            if (marker := marker_from_data_tier_decorator(decorator)) is not None
        ]
        if decorator_markers:
            markers[node.name] = decorator_markers[0]
            continue

        returns: list[ast.Return] = []
        for statement in node.body:
            visit_returns(statement, returns)

        return_markers: list[_Marker] = []
        has_unmarked_return = False
        for return_node in returns:
            if return_node.value is None:
                continue
            if _is_marker_call(return_node.value):
                assert isinstance(return_node.value, ast.Call)
                return_markers.append(_marker_from_call(return_node.value))
            else:
                has_unmarked_return = True
        if return_markers and not has_unmarked_return:
            markers[node.name] = return_markers[0]
    return markers


def _collect_value_markers(
    tree: ast.AST,
    return_markers: Mapping[str, _Marker],
) -> dict[str, tuple[_Marker, int]]:
    """Collect local/global names bound to marked values or marked normalizer output."""

    markers: dict[str, tuple[_Marker, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = _assigned_names(node)
        if not names or node.value is None:
            continue
        marker: _Marker | None = None
        if _is_marker_call(node.value):
            assert isinstance(node.value, ast.Call)
            marker = _marker_from_call(node.value)
        elif isinstance(node.value, ast.Call):
            call_name = _call_name(node.value.func)
            if call_name is not None and call_name in return_markers:
                marker = return_markers[call_name]
        if marker is None:
            continue
        for name in names:
            markers[name] = (marker, getattr(node, "lineno", marker.line))
    return markers


def _params_expr_from_execute_call(node: ast.Call) -> ast.AST | None:
    """Return the DB-API execute parameters expression, if one is present."""

    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg in {"params", "parameters", "vars"}:
            return keyword.value
    return None


def _params_marker_from_expr(
    node: ast.AST | None,
    value_markers: Mapping[str, tuple[_Marker, int]],
    return_markers: Mapping[str, _Marker],
) -> tuple[_Marker | None, int | None]:
    """Resolve normalization/tier evidence for execute params expressions."""

    if node is None:
        return None, None
    if _is_marker_call(node):
        assert isinstance(node, ast.Call)
        return _marker_from_call(node), node.lineno
    if isinstance(node, ast.Name) and node.id in value_markers:
        return value_markers[node.id]
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func)
        if call_name is not None and call_name in return_markers:
            return return_markers[call_name], node.lineno
    return None, getattr(node, "lineno", None)


def _sql_from_execute_arg(
    node: ast.AST,
    bindings: Mapping[str, _SqlBinding],
) -> tuple[str | None, _Marker | None, int | None]:
    if _is_marker_call(node):
        assert isinstance(node, ast.Call)
        marker = _marker_from_call(node)
        inner = node.args[0] if node.args else node
        if isinstance(inner, ast.Name) and inner.id in bindings:
            return bindings[inner.id].sql, marker, bindings[inner.id].line
        return _string_literal(inner), marker, None
    if isinstance(node, ast.Name) and node.id in bindings:
        binding = bindings[node.id]
        return binding.sql, binding.marker, binding.line
    return _string_literal(node), None, None


def _marker_valid(marker: _Marker | None, expected_tier: str) -> bool:
    return (
        marker is not None
        and marker.tier == expected_tier
        and marker.spec_ref is not None
        and marker.spec_ref.startswith("§5.2")
    )


def _is_execute_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute"


def _contains_call(node: ast.AST, predicate: Callable[[ast.Call], bool]) -> bool:
    return any(isinstance(child, ast.Call) and predicate(child) for child in ast.walk(node))


def _is_request_body_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "body"


def _is_json_raw_oura_decode(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "loads"):
        return False
    if not node.args:
        return False
    arg = node.args[0]
    return isinstance(arg, ast.Name) and arg.id in {"body", "raw_notification"}


def _is_oura_provider_fetch(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "get_json"


def _is_redis_physio_lpop(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "lpop"


def _is_raw_media_boundary(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "get_latest_frame":
        return True
    return (
        node.func.attr == "read"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "stdout"
    )


def _is_inbound_boundary_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Await):
        return _is_inbound_boundary_expression(node.value)
    predicates = (
        _is_request_body_call,
        _is_json_raw_oura_decode,
        _is_oura_provider_fetch,
        _is_redis_physio_lpop,
        _is_raw_media_boundary,
    )
    return any(_contains_call(node, predicate) for predicate in predicates)


def _scan_file(repo_root: Path, rel_path: Path) -> tuple[list[str], list[str], list[_Finding]]:
    path = repo_root / rel_path
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [], [], [_Finding(rel_path.as_posix(), exc.lineno or 0, f"syntax error: {exc.msg}")]

    insert_evidence: list[str] = []
    inbound_evidence: list[str] = []
    findings: list[_Finding] = []
    bindings = _collect_sql_bindings(tree)
    return_markers = _collect_return_markers(tree)
    value_markers = _collect_value_markers(tree, return_markers)

    for node in ast.walk(tree):
        if _is_execute_call(node):
            assert isinstance(node, ast.Call)
            if not node.args:
                continue
            sql, marker, binding_line = _sql_from_execute_arg(node.args[0], bindings)
            if sql is None or not _INSERT_RE.search(sql):
                continue
            sql_marker_valid = _marker_valid(marker, _EXPECTED_INSERT_TIER)
            if not sql_marker_valid:
                marker_line = f" marker_line={binding_line}" if binding_line else ""
                findings.append(
                    _Finding(
                        rel_path.as_posix(),
                        node.lineno,
                        (
                            "missing/invalid DataTier.PERMANENT annotation for INSERT "
                            f"call{marker_line}: {_line(lines, node.lineno)!r}"
                        ),
                    )
                )

            params_expr = _params_expr_from_execute_call(node)
            params_marker, params_line = _params_marker_from_expr(
                params_expr,
                value_markers,
                return_markers,
            )
            if not _marker_valid(params_marker, _EXPECTED_INSERT_TIER):
                params_line_text = f" params_line={params_line}" if params_line else ""
                findings.append(
                    _Finding(
                        rel_path.as_posix(),
                        node.lineno,
                        (
                            "missing/invalid DataTier.PERMANENT normalization evidence for "
                            f"INSERT params{params_line_text}: {_line(lines, node.lineno)!r}"
                        ),
                    )
                )
            elif sql_marker_valid:
                assert marker is not None
                assert params_marker is not None
                insert_evidence.append(
                    f"{rel_path.as_posix()}:{node.lineno} INSERT tier={marker.tier} "
                    f"{marker.spec_ref} params tier={params_marker.tier} {params_marker.spec_ref}"
                )
            continue

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        inbound_marker: _Marker | None = None
        inbound_node = value
        if _is_marker_call(value):
            assert isinstance(value, ast.Call)
            inbound_marker = _marker_from_call(value)
            inbound_node = value.args[0] if value.args else value
        if not _is_inbound_boundary_expression(inbound_node):
            continue
        if not _marker_valid(inbound_marker, "TRANSIENT"):
            findings.append(
                _Finding(
                    rel_path.as_posix(),
                    node.lineno,
                    (
                        "missing/invalid DataTier.TRANSIENT annotation for inbound raw/transient "
                        f"boundary: {_line(lines, node.lineno)!r}"
                    ),
                )
            )
        else:
            assert inbound_marker is not None
            inbound_evidence.append(
                f"{rel_path.as_posix()}:{node.lineno} inbound tier={inbound_marker.tier} {inbound_marker.spec_ref}"
            )

    return insert_evidence, inbound_evidence, findings


def _in_scope_python_files(repo_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in _IN_SCOPE_ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            files.append(path.relative_to(repo_root))
    return tuple(files)


def scan_data_classification(repo_root: Path) -> DataClassificationScan:
    """Scan in-scope API/pipeline source for §5.2 DataTier evidence."""

    insert_evidence: list[str] = []
    inbound_evidence: list[str] = []
    findings: list[_Finding] = []
    for rel_path in _in_scope_python_files(repo_root):
        file_insert, file_inbound, file_findings = _scan_file(repo_root, rel_path)
        insert_evidence.extend(file_insert)
        inbound_evidence.extend(file_inbound)
        findings.extend(file_findings)

    if not insert_evidence:
        findings.append(_Finding(".", 0, "no in-scope DataTier-annotated INSERT evidence found"))
    if not inbound_evidence:
        findings.append(_Finding(".", 0, "no in-scope DataTier-annotated inbound boundary evidence found"))

    return DataClassificationScan(
        insert_annotations=tuple(insert_evidence),
        inbound_annotations=tuple(inbound_evidence),
        findings=tuple(findings),
    )


def verify_data_classification(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify §13.14 data classification through concrete file:line evidence."""

    return _result(item, scan_data_classification(context.repo_root))


DATA_CLASSIFICATION_VERIFIERS: Mapping[str, AuditVerifier] = {
    _DATA_CLASSIFICATION_ITEM_ID: verify_data_classification,
}


def register_data_classification_verifiers(
    *,
    registry: AuditRegistry | None = None,
    item_ids: Iterable[str] | None = None,
) -> None:
    """Register the §13.14 data-classification verifier."""

    requested = set(DATA_CLASSIFICATION_VERIFIERS) if item_ids is None else set(item_ids)
    target = registry if registry is not None else get_default_registry()
    for item_id, verifier in DATA_CLASSIFICATION_VERIFIERS.items():
        if item_id not in requested or target.has_verifier(item_id):
            continue
        target.register(item_id)(verifier)


__all__ = [
    "DATA_CLASSIFICATION_VERIFIERS",
    "DataClassificationScan",
    "register_data_classification_verifiers",
    "scan_data_classification",
    "verify_data_classification",
]
