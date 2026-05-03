"""Structural enforcement for the committed §11 variable traceability registry."""

from __future__ import annotations

import ast
import json
import re
import shlex
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "registries" / "variable_traceability.yaml"
EXPECTED_COUNTS = {"A": 2, "B": 2, "C": 17, "D": 22, "E": 16, "F": 1}
EXPECTED_SECTIONS = {
    "Encoded Video Stream": "§11.1.1",
    "Raw Audio PCM": "§11.1.2",
    "Live Event Payloads": "§11.2.1",
    "Action Combo Trigger": "§11.2.2",
    "UTC Drift Offset": "§11.3.1",
    "Resampled Audio Chunks": "§11.3.2",
    "stimulus_time": "§11.3.3",
    "Segment ID": "§11.3.4",
    "Bandit Decision Snapshot": "§11.3.5",
    "3D Facial Landmarks": "§11.3.6",
    "AU12 Intensity Score": "§11.3.7",
    "RMSSD (Streamer)": "§11.3.8",
    "RMSSD (Operator)": "§11.3.9",
    "Heart Rate (Streamer/Operator)": "§11.3.10",
    "Physiological Validity Ratio": "§11.3.11",
    "Physiological Validity Flag": "§11.3.12",
    "Physiological Source Kind": "§11.3.13",
    "Physiological Derivation Method": "§11.3.14",
    "Physiological Window Length": "§11.3.15",
    "Physiological Freshness": "§11.3.16",
    "Physiological Staleness Flag": "§11.3.17",
    "Vocal Pitch F0": "§11.4.1",
    "Jitter": "§11.4.2",
    "Shimmer": "§11.4.3",
    "ASR Transcription": "§11.4.4",
    "Semantic Match": "§11.4.5",
    "F0 Validity (Measure Window)": "§11.4.6",
    "F0 Validity (Baseline Window)": "§11.4.7",
    "Perturbation Validity (Measure Window)": "§11.4.8",
    "Perturbation Validity (Baseline Window)": "§11.4.9",
    "Voiced Coverage (Measure Window)": "§11.4.10",
    "Voiced Coverage (Baseline Window)": "§11.4.11",
    "F0 Mean (Measure Window)": "§11.4.12",
    "F0 Mean (Baseline Window)": "§11.4.13",
    "F0 Delta (Semitones)": "§11.4.14",
    "Jitter Mean (Measure Window)": "§11.4.15",
    "Jitter Mean (Baseline Window)": "§11.4.16",
    "Jitter Delta": "§11.4.17",
    "Shimmer Mean (Measure Window)": "§11.4.18",
    "Shimmer Mean (Baseline Window)": "§11.4.19",
    "Shimmer Delta": "§11.4.20",
    "semantic_p_match": "§11.4.21",
    "semantic_reason_code": "§11.4.22",
    "Evaluation Variance": "§11.5.1",
    "gated_reward": "§11.5.2",
    "p90_intensity": "§11.5.3",
    "semantic_gate": "§11.5.4",
    "n_frames_in_window": "§11.5.5",
    "au12_baseline_pre": "§11.5.6",
    "Co-Modulation Index": "§11.5.7",
    "Physiological Coverage Ratio": "§11.5.8",
    "soft_reward_candidate": "§11.5.9",
    "au12_lift_p90": "§11.5.10",
    "au12_lift_peak": "§11.5.11",
    "au12_peak_latency_ms": "§11.5.12",
    "sync_peak_corr": "§11.5.13",
    "sync_peak_lag": "§11.5.14",
    "outcome_link_lag_s": "§11.5.15",
    "attribution_finality": "§11.5.16",
    "External Context Metadata": "§11.6.1",
}

MODULE_OWNED_SURFACES: tuple[tuple[str, str, str], ...] = (
    ("A", "services/desktop_app/processes/capture_supervisor.py", "public_output"),
    ("B", "services/worker/pipeline/ground_truth.py", "public_output"),
    ("C", "services/desktop_app/drift.py", "public_output"),
    ("C", "services/worker/pipeline/orchestrator.py", "public_output"),
    ("C", "packages/schemas/inference_handoff.py", "schema"),
    ("C", "packages/schemas/physiology.py", "schema"),
    ("D", "services/worker/tasks/inference.py", "public_output"),
    ("D", "packages/schemas/evaluation.py", "schema"),
    ("D", "packages/ml_core/acoustic.py", "schema"),
    ("E", "services/api/services/operator_read_service.py", "public_output"),
    ("E", "services/worker/pipeline/reward.py", "public_output"),
    ("E", "services/worker/pipeline/analytics.py", "persistence"),
    ("E", "packages/ml_core/attribution.py", "public_output"),
    ("E", "packages/schemas/attribution.py", "schema"),
    ("F", "services/worker/tasks/enrichment.py", "public_output"),
)


@dataclass(frozen=True, slots=True)
class VariableSurface:
    """Module-owned public/schema/persistence surface parsed from source."""

    module: str
    path: str
    surface: str
    source: str
    tokens: frozenset[str]

    def line_for(self, pattern: str) -> int:
        for line_no, line in enumerate(self.source.splitlines(), start=1):
            if pattern in line:
                return line_no
        for token in self.tokens:
            if pattern == token:
                return 1
        return 1


@dataclass(frozen=True, slots=True)
class VariableEvidence:
    """Concrete source-derived evidence for one §11 variable producer."""

    variable: str
    module: str
    path: str
    line: int
    surface: str
    detail: str

    @property
    def relationship(self) -> tuple[str, str]:
        return (self.variable, self.module)

    def describe(self) -> str:
        return f"{self.path}:{self.line} [{self.surface}] {self.detail}"


@pytest.mark.audit_item("13.13")
def test_variable_traceability_registry_rows_have_producer_emissions() -> None:
    """The §11 registry must exactly equal source-discovered variable producers."""
    registry = _load_registry(REGISTRY_PATH)
    variables = registry["variables"]

    expected = int(registry.get("row_count_expected", 0))
    assert len(variables) == expected == 60
    assert Counter(row["producer_module"] for row in variables) == EXPECTED_COUNTS

    registry_pairs = {(row["variable"], row["producer_module"]) for row in variables}
    discovered_evidence = _discover_variable_producers()
    discovered_pairs = set(discovered_evidence)

    rows_by_pair = {(row["variable"], row["producer_module"]): row for row in variables}
    missing = sorted(registry_pairs - discovered_pairs)
    extra = sorted(discovered_pairs - registry_pairs)
    assert not missing and not extra, _format_pair_diff(
        missing=missing,
        extra=extra,
        discovered_evidence=discovered_evidence,
        registry_rows_by_pair=rows_by_pair,
    )


@pytest.mark.audit_item("13.13")
def test_variable_traceability_registry_evidence_matches_sources() -> None:
    """Every registry evidence object must match its referenced source artifact."""
    registry = _load_registry(REGISTRY_PATH)
    variables = registry["variables"]

    evidence_errors: list[str] = []
    for row in variables:
        evidence_items = row.get("evidence")
        if not isinstance(evidence_items, list) or not evidence_items:
            evidence_errors.append(_row_id(row) + " has no evidence entries")
            continue
        for evidence in evidence_items:
            if not isinstance(evidence, Mapping):
                evidence_errors.append(_row_id(row) + f" invalid evidence object: {evidence!r}")
                continue
            rel_path = evidence.get("path")
            surface_kind = evidence.get("surface")
            patterns = evidence.get("patterns")
            if not isinstance(rel_path, str) or not rel_path:
                evidence_errors.append(_row_id(row) + f" invalid evidence path: {evidence!r}")
                continue
            if not isinstance(surface_kind, str) or not surface_kind:
                evidence_errors.append(_row_id(row) + f" invalid evidence surface: {evidence!r}")
                continue
            if not isinstance(patterns, list) or not all(
                isinstance(pattern, str) and pattern for pattern in patterns
            ):
                evidence_errors.append(_row_id(row) + f" invalid evidence patterns: {evidence!r}")
                continue
            try:
                source = _read_source(rel_path)
            except AssertionError as exc:
                evidence_errors.append(_row_id(row) + f" evidence path mismatch: {exc}")
                continue
            surface = VariableSurface(
                module=str(row["producer_module"]),
                path=rel_path,
                surface=surface_kind,
                source=source,
                tokens=frozenset(_extract_surface_tokens(rel_path, source)),
            )
            missing_patterns = [
                pattern for pattern in patterns if not _pattern_in_surface(pattern, surface)
            ]
            if missing_patterns:
                evidence_errors.append(
                    _row_id(row)
                    + f" evidence {rel_path} [{surface_kind}] missing patterns "
                    + repr(missing_patterns)
                )

    assert not evidence_errors, "\n".join(evidence_errors)


def _load_registry(path: Path) -> dict[str, Any]:
    assert path.exists(), f"Missing registry: {path}"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # JSON is a valid YAML subset for this registry.
        raise AssertionError(f"Registry is not parseable JSON/YAML subset: {path}: {exc}") from exc
    assert isinstance(loaded, dict), "Registry root must be an object"
    assert loaded.get("registry_ref") == "§11 Variable Extraction Matrix"
    assert loaded.get("audit_item") == "§13.13 Variable traceability"
    variables = loaded.get("variables")
    assert isinstance(variables, list) and variables, "Registry must contain variables"
    seen: set[str] = set()
    for row in variables:
        assert isinstance(row, dict), f"Invalid variable row: {row!r}"
        variable = row.get("variable")
        module = row.get("producer_module")
        section = row.get("section")
        assert isinstance(variable, str) and variable
        assert variable not in seen, f"Duplicate variable row: {variable}"
        seen.add(variable)
        assert module in EXPECTED_COUNTS, _row_id(row)
        assert section == EXPECTED_SECTIONS.get(variable), _row_id(row)
    assert set(seen) == set(EXPECTED_SECTIONS), "Registry variables differ from §11 artifact"
    return loaded


def _row_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('variable', '?')} / Module {row.get('producer_module', '?')} "
        f"({row.get('section', '?')})"
    )


def _format_pair_diff(
    *,
    missing: Iterable[tuple[str, str]],
    extra: Iterable[tuple[str, str]],
    discovered_evidence: Mapping[tuple[str, str], list[VariableEvidence]],
    registry_rows_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    parts: list[str] = []
    missing_list = list(missing)
    extra_list = list(extra)
    if missing_list:
        parts.append("Missing source-discovered implementation emissions registered in §11:")
        for variable, module in missing_list:
            row = registry_rows_by_pair.get((variable, module), {})
            parts.append(
                f"  - {variable} / Module {module}: registry evidence={row.get('evidence', [])!r}"
            )
    if extra_list:
        parts.append("Unregistered source-discovered implementation emissions:")
        for variable, module in extra_list:
            details = "; ".join(
                item.describe() for item in discovered_evidence.get((variable, module), [])[:3]
            )
            parts.append(f"  - {variable} / Module {module}: {details or 'no evidence detail'}")
    return "\n".join(parts)


def _read_source(rel_path: str) -> str:
    path = REPO_ROOT / rel_path
    assert path.exists(), f"Missing implementation artifact: {rel_path}"
    return path.read_text(encoding="utf-8")


def _parse(rel_path: str) -> ast.Module:
    return ast.parse(_read_source(rel_path), filename=rel_path)


def _literal_strings(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.add(child.value)
    return values


def _module_assign_strings(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
        if value is not None:
            return _literal_strings(value)
    return set()


def _class_field_names(tree: ast.Module, class_name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
            return fields
    return set()


def _function_node(tree: ast.Module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
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
        else:  # pragma: no cover - assertion message is what matters in CI.
            raise AssertionError(f"Cannot find AST qualname {qualname}")
    assert isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef))
    return current


def _dict_literal_key_sets(node: ast.AST) -> list[set[str]]:
    key_sets: list[set[str]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            keys = {
                key.value
                for key in child.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if keys:
                key_sets.append(keys)
    return key_sets


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _shell_commands(rel_path: str, executable: str) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] = []
    for raw_line in _read_source(rel_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("\\")
        line = line[:-1].strip() if continued else line
        if current:
            current.append(line)
            if not continued:
                joined = " ".join(current)
                current.clear()
                try:
                    tokens = shlex.split(joined)
                except ValueError:
                    continue
                if tokens and tokens[0] == executable:
                    commands.append(tokens)
        elif line.startswith(executable):
            if continued:
                current.append(line)
            else:
                try:
                    tokens = shlex.split(line)
                except ValueError:
                    continue
                if tokens and tokens[0] == executable:
                    commands.append(tokens)
    return commands


def _sql_table_columns(rel_path: str, table: str | None = None) -> set[str]:
    source = _read_source(rel_path)
    table_pattern = re.escape(table) if table is not None else r"[A-Za-z_][A-Za-z0-9_]*"
    columns: set[str] = set()
    for match in re.finditer(
        rf"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+{table_pattern}\s*\((.*?)\);",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip().rstrip(",")
            if not line or line.startswith("--"):
                continue
            name = line.split()[0].strip('"')
            if name.upper() not in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                columns.add(name)
    return columns


def _sql_insert_columns(source: str, table: str | None = None) -> set[str]:
    columns: set[str] = set()
    table_pattern = re.escape(table) if table is not None else r"[A-Za-z_][A-Za-z0-9_]*"
    for match in re.finditer(
        rf"INSERT\s+INTO\s+{table_pattern}\s*\((.*?)\)\s*VALUES",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        columns.update(col.strip().strip('"') for col in match.group(1).split(",") if col.strip())
    return columns


def _dataclass_fields(tree: ast.Module, class_name: str) -> set[str]:
    return _class_field_names(tree, class_name)


def _keyword_names_in_calls(node: ast.AST, call_name: str) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name == call_name:
            names.update(keyword.arg for keyword in child.keywords if keyword.arg is not None)
    return names


def _discover_variable_producers() -> dict[tuple[str, str], list[VariableEvidence]]:
    """Discover §11 variable producers from implementation-owned source surfaces.

    The committed registry is the expected declaration.  This discovery pass
    independently walks source-owned emission surfaces (public payload keys,
    schema/dataclass fields and aliases, and local persistence columns) and then
    normalizes implementation identifiers into canonical §11 names.  The
    normalization layer is identifier-only: it intentionally does not encode a
    variable→module relationship table.
    """

    def norm(value: str) -> str:
        return "_".join(re.findall(r"[a-z0-9]+", value.lower().strip("_")))

    def suffixes(key: str) -> tuple[str, ...]:
        parts = key.split("_")
        return tuple("_".join(parts[index:]) for index in range(len(parts)) if parts[index:])

    def func_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def target_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return target_name(node.value)
        return None

    def call_receiver(node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Attribute):
            return target_name(node.func.value)
        return None

    def is_dataclass_def(node: ast.ClassDef) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            or (isinstance(decorator, ast.Call) and func_name(decorator.func) == "dataclass")
            for decorator in node.decorator_list
        )

    def is_model_or_schema(node: ast.ClassDef, surface: VariableSurface) -> bool:
        return (
            surface.surface == "schema"
            or is_dataclass_def(node)
            or any(func_name(base) in {"BaseModel", "AttributionBaseModel"} for base in node.bases)
        )

    def field_alias(stmt: ast.AnnAssign) -> str | None:
        if not isinstance(stmt.value, ast.Call) or func_name(stmt.value.func) != "Field":
            return None
        for keyword in stmt.value.keywords:
            if keyword.arg == "alias" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value if isinstance(keyword.value.value, str) else None
        return None

    def string_collection(value: ast.AST) -> tuple[str, ...]:
        if not isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return ()
        strings: list[str] = []
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                strings.append(element.value)
            elif isinstance(element, ast.Starred):
                strings.extend(string_collection(element.value))
        return tuple(strings)

    def is_field_collection(name: str) -> bool:
        upper = name.upper()
        return (
            "FIELD" in upper and "FORWARD" not in upper and "METRICS" not in upper
        ) or "CANONICAL" in upper

    def contains_row_read(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in {"row", "enc_row", "active_row"}
            for child in ast.walk(node)
        )

    def enclosing_function(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> ast.AST | None:
        current: ast.AST | None = node
        while current is not None:
            current = parents.get(current)
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
        return None

    def row_derived_names(function: ast.AST | None) -> set[str]:
        if function is None:
            return set()
        names: set[str] = set()
        for child in ast.walk(function):
            value: ast.AST | None = None
            targets: list[ast.AST] = []
            if isinstance(child, ast.Assign):
                value = child.value
                targets = list(child.targets)
            elif isinstance(child, ast.AnnAssign):
                value = child.value
                targets = [child.target]
            if value is None or not contains_row_read(value):
                continue
            for target in targets:
                name = target_name(target)
                if name:
                    names.add(name)
        return names

    def sql_columns(source: str, *, ddl: bool) -> list[tuple[str, str, int]]:
        pattern = (
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);"
            if ddl
            else r"INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*VALUES"
        )
        columns: list[tuple[str, str, int]] = []
        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.DOTALL):
            table = match.group(1)
            start_line = source[: match.start()].count("\n") + 1
            if ddl:
                for offset, raw_line in enumerate(match.group(2).splitlines(), start=1):
                    line = raw_line.strip().rstrip(",")
                    if not line or line.startswith("--"):
                        continue
                    name = line.split()[0].strip('"')
                    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                        continue
                    if name.upper() not in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                        columns.append((table, name, start_line + offset))
            else:
                for column in match.group(2).split(","):
                    name = column.strip().strip('"')
                    if name:
                        columns.append((table, name, start_line))
        return columns

    def add_emission(
        emissions: list[dict[str, Any]],
        identifier: str,
        line: int,
        kind: str,
        detail: str,
        **extra: Any,
    ) -> None:
        if identifier:
            emissions.append(
                {"identifier": identifier, "line": line, "kind": kind, "detail": detail, **extra}
            )

    public_helper_evidence = {
        "scrape_context",
        "compute_soft_reward_candidate",
        "extract_landmarks",
    }

    def python_emissions(surface: VariableSurface) -> list[dict[str, Any]]:
        tree = ast.parse(surface.source, filename=surface.path)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        row_name_cache: dict[ast.AST | None, set[str]] = {}
        emissions: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and is_model_or_schema(node, surface):
                # Private/internal dataclasses such as _WindowSummary are helper
                # state, not public schema/dataclass emissions.
                if node.name.startswith("_"):
                    continue
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        identifier = f"{node.name}.{stmt.target.id}"
                        add_emission(
                            emissions,
                            identifier,
                            getattr(stmt, "lineno", surface.line_for(stmt.target.id)),
                            "schema/model field"
                            if surface.surface == "schema"
                            else "dataclass field",
                            f"{identifier} declared field",
                        )
                        alias = field_alias(stmt)
                        if alias:
                            add_emission(
                                emissions,
                                alias,
                                getattr(stmt, "lineno", surface.line_for(alias)),
                                "schema/model alias",
                                f"{identifier} declares alias {alias!r}",
                            )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in public_helper_evidence or "landmark" in node.name:
                    add_emission(
                        emissions,
                        node.name,
                        getattr(node, "lineno", surface.line_for(node.name)),
                        "public helper",
                        f"public helper {node.name}()",
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for target in targets:
                    name = target_name(target)
                    if not name:
                        continue
                    if (
                        name in {"FFMPEG_RESAMPLE_CMD", "drift_offset", "landmarks"}
                        or "landmark" in name
                    ):
                        add_emission(
                            emissions,
                            name,
                            getattr(node, "lineno", surface.line_for(name)),
                            "public output assignment",
                            f"assignment target {name}",
                        )
                    if isinstance(value, ast.Dict):
                        add_emission(
                            emissions,
                            name,
                            getattr(node, "lineno", surface.line_for(name)),
                            "public output dict",
                            f"dict payload {name}",
                        )
                    if value is not None and is_field_collection(name):
                        for field in string_collection(value):
                            add_emission(
                                emissions,
                                field,
                                getattr(node, "lineno", surface.line_for(field)),
                                "field declaration",
                                f"{name} declares {field!r}",
                            )
            elif isinstance(node, ast.Dict):
                dict_parent: ast.AST | None = parents.get(node)
                container = "dict"
                if isinstance(dict_parent, ast.Assign):
                    container = next(
                        (
                            name
                            for name in (target_name(target) for target in dict_parent.targets)
                            if name
                        ),
                        "dict",
                    )
                elif isinstance(dict_parent, ast.AnnAssign):
                    container = target_name(dict_parent.target) or "dict"
                elif isinstance(dict_parent, ast.Return):
                    container = "return"
                elif isinstance(dict_parent, ast.Call):
                    container = call_receiver(dict_parent) or func_name(dict_parent.func) or "dict"
                if container == "cur":
                    continue
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        identifier = (
                            f"{container}.{key.value}" if container != "dict" else key.value
                        )
                        add_emission(
                            emissions,
                            identifier,
                            getattr(key, "lineno", surface.line_for(key.value)),
                            "public output dict key",
                            f"{container} emits key {key.value!r}",
                        )
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    container = target_name(node.value) or "mapping"
                    add_emission(
                        emissions,
                        f"{container}.{node.slice.value}",
                        getattr(node, "lineno", surface.line_for(node.slice.value)),
                        "public output mapping key",
                        f"{container} assigns key {node.slice.value!r}",
                    )
            elif isinstance(node, ast.Call):
                name = func_name(node.func)
                if name in public_helper_evidence or "landmark" in name:
                    add_emission(
                        emissions,
                        name,
                        getattr(node, "lineno", surface.line_for(name)),
                        "public helper call",
                        f"public helper call {name}()",
                    )
                if name[:1].isupper() and surface.path.endswith("operator_read_service.py"):
                    function = enclosing_function(node, parents)
                    row_names = row_name_cache.setdefault(function, row_derived_names(function))
                    for keyword in node.keywords:
                        if keyword.arg is None:
                            continue
                        if isinstance(keyword.value, ast.Name) and keyword.value.id in row_names:
                            continue
                        add_emission(
                            emissions,
                            keyword.arg,
                            getattr(keyword, "lineno", surface.line_for(keyword.arg)),
                            "returned model payload key",
                            f"{name}(...) emits keyword {keyword.arg!r}",
                        )
                for keyword in node.keywords:
                    if (
                        keyword.arg == "attribution_method"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        add_emission(
                            emissions,
                            keyword.value.value,
                            getattr(keyword, "lineno", surface.line_for(str(keyword.value.value))),
                            "attribution score method",
                            f"attribution_method emits {keyword.value.value!r}",
                        )

        if surface.path.endswith("capture_supervisor.py"):
            if "--video-codec=h264" in surface.source and "--record-format=mkv" in surface.source:
                add_emission(
                    emissions,
                    "video_stream",
                    surface.line_for("--video-codec=h264"),
                    "scrcpy public stream",
                    "capture supervisor emits H.264 MKV video stream",
                )
            if "--audio-codec=raw" in surface.source and "--record-format=wav" in surface.source:
                add_emission(
                    emissions,
                    "audio_stream",
                    surface.line_for("--audio-codec=raw"),
                    "scrcpy public stream",
                    "capture supervisor emits raw WAV audio stream",
                )

        if surface.surface == "persistence":
            for table, column, line in sql_columns(surface.source, ddl=False):
                add_emission(
                    emissions,
                    f"{table}.{column}",
                    line,
                    "SQL insert column",
                    f"INSERT INTO {table} column {column!r}",
                    table=table,
                )
        return emissions

    def shell_emissions(surface: VariableSurface) -> list[dict[str, Any]]:
        emissions: list[dict[str, Any]] = []
        for line_no, line in enumerate(surface.source.splitlines(), start=1):
            match = re.match(r'\s*([A-Z_]*PIPE)="?([^"\n]+)"?', line)
            if match:
                add_emission(
                    emissions,
                    match.group(1),
                    line_no,
                    "shell stream declaration",
                    f"shell stream declaration {match.group(1)}={match.group(2)!r}",
                )
        for command in _shell_commands(surface.path, "scrcpy"):
            joined = " ".join(command)
            command_line = surface.line_for("scrcpy")
            if any(token.startswith("--video-codec") for token in command):
                add_emission(
                    emissions,
                    "video_stream",
                    command_line,
                    "shell public stream",
                    f"scrcpy video stream command {joined!r}",
                )
            if any(token.startswith("--audio-codec") for token in command):
                add_emission(
                    emissions,
                    "audio_stream",
                    command_line,
                    "shell public stream",
                    f"scrcpy audio stream command {joined!r}",
                )
        return emissions

    def ddl_emissions(surface: VariableSurface) -> list[dict[str, Any]]:
        emissions: list[dict[str, Any]] = []
        ddl_table_modules = {
            "events": {"B"},
            "encounter_log": {"E"},
            "comodulation_log": {"E"},
            "attribution_score": {"E"},
            "event_outcome_link": {"E"},
            "context": {"F"},
        }
        for table, column, line in sql_columns(surface.source, ddl=True):
            if surface.module not in ddl_table_modules.get(table, set()):
                continue
            add_emission(
                emissions,
                f"{table}.{column}",
                line,
                "SQL DDL column",
                f"CREATE TABLE {table} column {column!r}",
                table=table,
            )
        return emissions

    canonical_by_key: dict[str, tuple[str, ...]] = {
        norm(variable): (variable,) for variable in EXPECTED_SECTIONS
    }
    aliases: dict[str, tuple[str, ...]] = {}

    def add_alias(names: Iterable[str], *variables: str) -> None:
        for name in names:
            aliases[norm(name)] = tuple(variables)

    add_alias(("VIDEO_PIPE", "video_stream", "video_stream_mkv"), "Encoded Video Stream")
    add_alias(("AUDIO_PIPE", "audio_stream", "audio_stream_raw"), "Raw Audio PCM")
    add_alias(("live_event", "live_events", "tiktok_live_event"), "Live Event Payloads")
    add_alias(("Action_Combo", "combo_dict", "is_combo", "action_combo"), "Action Combo Trigger")
    add_alias(("drift_offset",), "UTC Drift Offset")
    add_alias(("FFMPEG_RESAMPLE_CMD", "read_chunk", "audio_resampler"), "Resampled Audio Chunks")
    add_alias(("extract_landmarks", "landmarks", "facial_landmarks"), "3D Facial Landmarks")
    add_alias(
        (
            "AU12Observation.intensity",
            "_au12_series",
            "au12_series",
            "au12_series.intensity",
            "au12_intensity",
        ),
        "AU12 Intensity Score",
    )
    add_alias(("rmssd_ms", "streamer_rmssd_ms"), "RMSSD (Streamer)")
    add_alias(("operator_rmssd_ms",), "RMSSD (Operator)")
    add_alias(("heart_rate_bpm",), "Heart Rate (Streamer/Operator)")
    add_alias(("validity_ratio",), "Physiological Validity Ratio")
    add_alias(("is_valid",), "Physiological Validity Flag")
    add_alias(("source_kind",), "Physiological Source Kind")
    add_alias(("derivation_method",), "Physiological Derivation Method")
    add_alias(("window_s",), "Physiological Window Length")
    add_alias(("freshness_s",), "Physiological Freshness")
    add_alias(("is_stale",), "Physiological Staleness Flag")
    add_alias(("transcription", "transcripts.text"), "ASR Transcription")
    add_alias(("is_match",), "Semantic Match")
    add_alias(("confidence_score", "evaluations.confidence"), "semantic_p_match")
    add_alias(("reasoning", "evaluations.reasoning"), "semantic_reason_code")
    add_alias(("f0_valid_measure",), "F0 Validity (Measure Window)")
    add_alias(("f0_valid_baseline",), "F0 Validity (Baseline Window)")
    add_alias(("perturbation_valid_measure",), "Perturbation Validity (Measure Window)")
    add_alias(("perturbation_valid_baseline",), "Perturbation Validity (Baseline Window)")
    add_alias(("voiced_coverage_measure_s",), "Voiced Coverage (Measure Window)")
    add_alias(("voiced_coverage_baseline_s",), "Voiced Coverage (Baseline Window)")
    add_alias(("f0_mean_measure_hz",), "Vocal Pitch F0", "F0 Mean (Measure Window)")
    add_alias(("f0_mean_baseline_hz",), "Vocal Pitch F0", "F0 Mean (Baseline Window)")
    add_alias(("f0_delta_semitones",), "F0 Delta (Semitones)")
    add_alias(("jitter_mean_measure",), "Jitter", "Jitter Mean (Measure Window)")
    add_alias(("jitter_mean_baseline",), "Jitter", "Jitter Mean (Baseline Window)")
    add_alias(("jitter_delta",), "Jitter", "Jitter Delta")
    add_alias(("shimmer_mean_measure",), "Shimmer", "Shimmer Mean (Measure Window)")
    add_alias(("shimmer_mean_baseline",), "Shimmer", "Shimmer Mean (Baseline Window)")
    add_alias(("shimmer_delta",), "Shimmer", "Shimmer Delta")
    add_alias(("coverage_ratio",), "Physiological Coverage Ratio")
    add_alias(("finality",), "attribution_finality")
    add_alias(("lag_s", "event_outcome_link.lag_s"), "outcome_link_lag_s")
    add_alias(("scrape_context", "context.data"), "External Context Metadata")

    # Tables that persist upstream-produced values are scanned but not treated as
    # producer declarations for this producer-registry test.  Locally derived
    # persistence tables remain eligible.
    passthrough_tables = {
        "metrics",
        "transcripts",
        "evaluations",
        "physiology_log",
        "attribution_event",
        "outcome_event",
    }
    producer_tables = {
        "events",
        "encounter_log",
        "comodulation_log",
        "attribution_score",
        "event_outcome_link",
        "context",
    }

    out_of_scope = {
        "id",
        "uniqueid",
        "unique_id",
        "session_id",
        "event_id",
        "outcome_id",
        "link_id",
        "score_id",
        "segment_window_start_utc",
        "segment_window_end_utc",
        "timestamp_utc",
        "timestamp_s",
        "ts_monotonic",
        "window_start_ts",
        "window_end_ts",
        "event_time_utc",
        "outcome_time_utc",
        "source_timestamp_utc",
        "window_start_utc",
        "window_end_utc",
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "end_dated_at",
        "scraped_at_utc",
        "stream_url",
        "source_url",
        "source_system",
        "source_event_ref",
        "scrape_type",
        "status",
        "success",
        "payload",
        "events",
        "data",
        "title",
        "text_content",
        "meta_tags",
        "status_code",
        "media_source",
        "codec",
        "resolution",
        "audio_bytes",
        "segments",
        "context",
        "streamer",
        "operator",
        "physiological_context",
        "physiology_stale",
        "null_reason",
        "attribution_outcome",
        "outcome_events",
        "creator_follow",
        "creator_follow_outcome",
        "event_type",
        "active_arm",
        "arm",
        "arm_id",
        "selected_arm_id",
        "candidate_arm_ids",
        "posterior_by_arm",
        "sampled_theta_by_arm",
        "posterior_alpha",
        "posterior_beta",
        "alpha_param",
        "beta_param",
        "experiment_id",
        "experiment_code",
        "policy_version",
        "selection_method",
        "selection_time_utc",
        "decision_context_hash",
        "expected_greeting",
        "expected_rule_text_hash",
        "reward_path_version",
        "method_version",
        "semantic_method",
        "semantic_method_version",
        "semantic",
        "schema_version",
        "evidence_flags",
        "eligibility_flags",
        "horizon_s",
        "link_rule_version",
        "attribution_method",
        "score_raw",
        "score_normalized",
        "confidence",
        "provider",
        "subject_role",
        "sample_interval_s",
        "valid_sample_count",
        "expected_sample_count",
        "ibi_ms_items",
        "rmssd_items_ms",
        "heart_rate_items_bpm",
        "motion_items",
        "n_paired_observations",
        "streamer_rmssd_mean",
        "operator_rmssd_mean",
        "window_minutes",
        "label",
        "greeting_text",
        "enabled",
        "recent_reward_mean",
        "recent_semantic_pass_rate",
        "selection_count",
        "duration_s",
        "last_segment_completed_at_utc",
        "latest_reward",
        "latest_semantic_gate",
        "au12_intensity",
        "semantic_gate",
        "p90_intensity",
        "gated_reward",
        "n_frames_in_window",
        "au12_baseline_pre",
        "stimulus_time",
        "co_modulation_index",
        "coverage_ratio",
        "finality",
        "lag_s",
        "semantic_p_match",
        "semantic_reason_code",
        "bandit_decision_snapshot",
        "f0_valid_measure",
        "f0_valid_baseline",
        "perturbation_valid_measure",
        "perturbation_valid_baseline",
        # Upstream-produced variables persisted by analytics.py through the
        # mark_data_tier(...) wrapper; analytics.py is a passthrough sink for
        # these (Module D produces is_match, Module C produces the physiology
        # fields), so the dict-literal emission inside the mark_data_tier call
        # would otherwise be reported as unmapped after the analytics surface
        # filter strips the matched canonical variables.
        "is_match",
        "source_kind",
        "window_s",
        "validity_ratio",
        "is_valid",
        "rmssd_ms",
        "heart_rate_bpm",
        "freshness_s",
        "is_stale",
        "is_calibrating",
        "calibration_frames_accumulated",
        "calibration_frames_required",
        "timestampedau12_intensity",
        "f0_hz",
        "periodic_peak_count",
        "jitter_local",
        "shimmer_local",
        "f0_valid",
        "perturbation_valid",
        "voiced_coverage_s",
        "f0_mean_hz",
        "jitter_mean",
        "shimmer_mean",
        "_audio_data",
        "_frame_data",
        "_experiment_code",
        "_active_arm",
        "_experiment_id",
        "_expected_greeting",
        "_physiological_context",
    }
    relevant_tokens = {
        "acoustic",
        "attribution",
        "au12",
        "baseline",
        "candidate",
        "combo",
        "confidence",
        "context",
        "coverage",
        "delta",
        "drift",
        "f0",
        "finality",
        "freshness",
        "gated",
        "gate",
        "heart",
        "intensity",
        "jitter",
        "lag",
        "landmark",
        "match",
        "metric",
        "modulation",
        "p90",
        "peak",
        "physiological",
        "pitch",
        "reason",
        "reward",
        "rmssd",
        "semantic",
        "shimmer",
        "source",
        "stale",
        "stimulus",
        "sync",
        "transcription",
        "valid",
        "validity",
        "variance",
        "voiced",
        "window",
    }

    def canonicalize(
        identifier: str, surface: VariableSurface, emission: Mapping[str, Any]
    ) -> tuple[str, ...]:
        key = norm(identifier)
        table = str(emission.get("table") or "")
        keys = (key, *suffixes(key))
        if surface.path.endswith("capture_supervisor.py") and key.endswith("drift_offset"):
            return ()
        if emission.get("kind") in {"SQL insert column", "SQL DDL column"}:
            if table in passthrough_tables:
                return ()
            if table and table not in producer_tables:
                return ()
            if any(part in out_of_scope for part in keys):
                return ()
        matched: list[str] = []
        for candidate in keys:
            for variable in canonical_by_key.get(candidate, ()) + aliases.get(candidate, ()):
                if variable not in matched:
                    matched.append(variable)
        if "rmssd_ms" in keys and "RMSSD (Operator)" not in matched:
            matched.append("RMSSD (Operator)")
        if (
            "confidence" in keys
            and (
                surface.path.endswith("packages/schemas/evaluation.py")
                or table == "evaluations"
                or "semantic" in key
            )
            and "semantic_p_match" not in matched
        ):
            matched.append("semantic_p_match")
        if (
            "reasoning" in keys
            and (
                surface.path.endswith("packages/schemas/evaluation.py")
                or table == "evaluations"
                or "semantic" in key
            )
            and "semantic_reason_code" not in matched
        ):
            matched.append("semantic_reason_code")

        def produced_in_section(variable: str, prefix: str) -> bool:
            return EXPECTED_SECTIONS.get(variable, "").startswith(prefix)

        filtered: list[str] = []
        for variable in matched:
            # Read-side DTOs and persistence adapters often re-emit upstream
            # columns.  They are scanned for unmapped/extra evidence, but known
            # pass-through readback surfaces are not producer declarations.
            if (
                surface.path.endswith("operator_read_service.py")
                and variable != "Evaluation Variance"
            ):
                continue
            if surface.path.endswith("services/worker/pipeline/analytics.py") and variable not in {
                "Co-Modulation Index",
                "Physiological Coverage Ratio",
            }:
                continue
            if surface.path.endswith(
                "services/worker/tasks/inference.py"
            ) and not produced_in_section(variable, "§11.4"):
                continue
            if surface.path.endswith("packages/schemas/attribution.py") and variable not in {
                "attribution_finality",
                "outcome_link_lag_s",
            }:
                continue
            if surface.path.endswith("packages/ml_core/attribution.py") and not produced_in_section(
                variable, "§11.5"
            ):
                continue
            filtered.append(variable)
        return tuple(filtered)

    def clear_out_of_scope(identifier: str, emission: Mapping[str, Any]) -> bool:
        key = norm(identifier)
        table = str(emission.get("table") or "")
        if table in passthrough_tables:
            return True
        if key in out_of_scope or any(part in out_of_scope for part in suffixes(key)):
            return True
        if key.endswith("drift_offset"):
            return True
        if key.startswith(("_", "test_")):
            return True
        if (
            key.endswith(("_id", "_uuid", "_url", "_utc", "_at", "_version"))
            and key != "segment_id"
        ):
            return True
        return not bool(set(key.split("_")) & relevant_tokens)

    discovered: dict[tuple[str, str], list[VariableEvidence]] = {}
    unmapped: list[str] = []
    for surface in _collect_module_owned_surfaces():
        if surface.path.endswith(".py"):
            emissions = python_emissions(surface)
        elif surface.path.endswith(".sh"):
            emissions = shell_emissions(surface)
        elif surface.path.endswith(".sql"):
            emissions = ddl_emissions(surface)
        else:
            emissions = []
        for emission in emissions:
            identifier = str(emission["identifier"])
            variables = canonicalize(identifier, surface, emission)
            if not variables:
                if not clear_out_of_scope(identifier, emission):
                    unmapped.append(
                        f"{surface.path}:{emission['line']} [{surface.surface}] "
                        f"{emission['kind']} {identifier!r}: {emission['detail']}"
                    )
                continue
            for variable in variables:
                evidence = VariableEvidence(
                    variable=variable,
                    module=surface.module,
                    path=surface.path,
                    line=int(emission["line"]),
                    surface=surface.surface,
                    detail=(
                        f"source-discovered {emission['kind']} {identifier!r} "
                        f"normalized to {variable!r}: {emission['detail']}"
                    ),
                )
                discovered.setdefault(evidence.relationship, []).append(evidence)

    if unmapped:
        preview = "\n".join(f"  - {item}" for item in unmapped[:40])
        extra = "" if len(unmapped) <= 40 else f"\n  ... {len(unmapped) - 40} more"
        raise AssertionError(
            "Relevant implementation emissions could not be normalized to a "
            "canonical §11 variable:\n" + preview + extra
        )
    return discovered


def _collect_module_owned_surfaces() -> list[VariableSurface]:
    surfaces: list[VariableSurface] = []
    surface_specs = (
        *MODULE_OWNED_SURFACES,
        ("B", "services/cloud_api/db/sql/01-schema.sql", "persistence"),
        ("D", "services/cloud_api/db/sql/01-schema.sql", "persistence"),
        ("E", "services/cloud_api/db/sql/01-schema.sql", "persistence"),
        ("E", "services/cloud_api/db/sql/03-physiology.sql", "persistence"),
        ("E", "services/cloud_api/db/sql/05-attribution.sql", "persistence"),
        ("F", "services/cloud_api/db/sql/01-schema.sql", "persistence"),
    )
    for module, rel_path, surface_kind in surface_specs:
        source = _read_source(rel_path)
        tokens = _extract_surface_tokens(rel_path, source)
        surfaces.append(
            VariableSurface(
                module=module,
                path=rel_path,
                surface=surface_kind,
                source=source,
                tokens=frozenset(tokens),
            )
        )
    return surfaces


def _extract_surface_tokens(rel_path: str, source: str) -> set[str]:
    tokens: set[str] = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))
    if rel_path.endswith(".py"):
        tree = ast.parse(source, filename=rel_path)
        tokens.update(_literal_strings(tree))
        tokens.update(_sql_insert_columns(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        tokens.add(stmt.target.id)
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        tokens.add(key.value)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                tokens.add(node.arg)
        if rel_path == "services/worker/pipeline/orchestrator.py":
            tokens.update(_module_assign_strings(tree, "FFMPEG_RESAMPLE_CMD"))
        if rel_path == "services/api/services/operator_read_service.py":
            with suppress(AssertionError):
                tokens.update(
                    _keyword_names_in_calls(
                        _function_node(tree, "OperatorReadService._build_arm_summary"),
                        "ArmSummary",
                    )
                )
    elif rel_path.endswith(".sh"):
        for command in _shell_commands(rel_path, "scrcpy"):
            tokens.update(command)
    elif rel_path.endswith(".sql"):
        tokens.update(_sql_table_columns(rel_path))
        tokens.update(_sql_insert_columns(source))
    return tokens


def _pattern_in_surface(pattern: str, surface: VariableSurface) -> bool:
    return pattern in surface.source or pattern in surface.tokens
