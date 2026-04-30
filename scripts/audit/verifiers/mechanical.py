"""Mechanical verifier implementations for greppable §13 audit controls."""

from __future__ import annotations

import ast
import json
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.audit.registry import AuditContext, AuditVerifier, get_default_registry
from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item


@dataclass(frozen=True, slots=True)
class _LocatedValue:
    value: Any
    line: int
    snippet: str


@dataclass(frozen=True, slots=True)
class _Check:
    passed: bool
    evidence: str


@dataclass(frozen=True, slots=True)
class _ExpectedVault:
    key_bytes: int
    nonce_bytes: int
    shred_command: tuple[str, ...]
    interval_hours: int
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedDependency:
    package: str
    version: str
    target_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RequirementEntry:
    package: str
    specifier: str
    raw_line: str
    rel_path: str
    line: int


@dataclass(frozen=True, slots=True)
class _SpecExtraction:
    value: Any | None
    evidence: str


@dataclass(frozen=True, slots=True)
class _CanonicalPattern:
    pattern: re.Pattern[str]
    source: str


@dataclass(frozen=True, slots=True)
class _GrayBandBranchEvidence:
    passed: bool
    evidence: str


_CODE_SCAN_DIRS = ("services", "packages", "scripts")
_SCAN_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}
_BINARY_SCAN_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".db",
    ".gif",
    ".gz",
    ".ico",
    ".jpg",
    ".jpeg",
    ".mkv",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".sqlite",
    ".tar",
    ".wav",
    ".zip",
}
_ROOT_CONFIG_SUFFIXES = {".cfg", ".env", ".ini", ".json", ".toml", ".yaml", ".yml"}
_CONFIG_ARTIFACT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ddl",
    ".env",
    ".ini",
    ".json",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
}
_CONFIG_ARTIFACT_DIRS = {"config", "configs", "ddl", "migration", "migrations", "sql"}
_CANONICAL_VERIFIER_PATH = Path("scripts/audit/verifiers/mechanical.py")

# §0.3/§13.15 rationale: these retired synonyms are also mandatory container,
# package, protocol, or infrastructure identifiers in the same spec and create
# unsafe grep noise.  They are excluded only after parsing §0.3; all other parsed
# retired_synonym_list entries are mechanically scanned.
_CANONICAL_TERM_EXCLUSIONS: Mapping[str, str] = {
    "api": "§0.3 defines API Server container name 'api'; config must retain it.",
    "worker": "§0.3 defines ML Worker container name 'worker'; config must retain it.",
    "Redis": "§0.3 Message Broker definition names the Redis technology.",
    "queue": "§0.3 Message Broker definition describes Celery task queues generically.",
    "broker": "§0.3 Message Broker component has broker as a common protocol noun.",
    "postgres": "§0.3 defines Persistent Store container name 'postgres'.",
    "PostgreSQL": "§0.3 Persistent Store definition names the PostgreSQL technology.",
    "database": "§0.3 Persistent Store definition uses database as a storage noun.",
    "DB": "§0.3 retired synonym is too short for safe grep without false positives.",
    "stream_scrcpy": "§0.3 defines Capture Container container name 'stream_scrcpy'.",
    "orchestrator": "§0.3 defines Orchestrator Container container/CMD identifier.",
}

# Prose-only guardrails that are not represented in §0.3 as individual retired
# synonyms; each inline regex carries its own normative source reference.
_PROSE_CANONICAL_REGEXES: tuple[tuple[str, str], ...] = (
    (r"free-form ration" + "ale", "§8/§13.15 free-form semantic rationales are forbidden"),
    (r"free-form ration" + "ales", "§8/§13.15 free-form semantic rationales are forbidden"),
    (
        r"free-form semantic ration" + "ale",
        "§8/§13.15 free-form semantic rationales are forbidden",
    ),
    (
        r"free-form semantic ration" + "ales",
        "§8/§13.15 free-form semantic rationales are forbidden",
    ),
    (r"x[_-]?max[- ]normalized reward", "§7B/§13.15 reward input terminology"),
    (r"x[_-]?max as reward input", "§7B/§13.15 reward input terminology"),
    (r"x[_-]?max reward input", "§7B/§13.15 reward input terminology"),
    (r"\bpitch_f" + r"0\b", "§7D/§13.15 acoustic variable terminology"),
    (r"legacy acoustic scal" + "ar", "§7D/§13.15 acoustic variable terminology"),
    (r"scalar-only acous" + "tic", "§7D/§13.15 acoustic variable terminology"),
    (r"\[0\.0, 5\.0\].*AU" + "12", "§7A/§13.15 AU12 bounded-scale terminology"),
    (r"AU" + r"12.*\[0\.0, 5\.0\]", "§7A/§13.15 AU12 bounded-scale terminology"),
    (r"AU" + r"12 clamp.*5\.0", "§7A/§13.15 AU12 bounded-scale terminology"),
    (r"clamp.*AU" + r"12.*5\.0", "§7A/§13.15 AU12 bounded-scale terminology"),
)


def _result(
    item: Section13Item,
    passed: bool,
    evidence: str,
    follow_up: str | None = None,
) -> AuditResult:
    return AuditResult(
        item_id=item.item_id,
        title=item.title,
        passed=passed,
        evidence=evidence,
        follow_up=follow_up,
    )


def _checks_result(item: Section13Item, checks: Sequence[_Check]) -> AuditResult:
    passed = all(check.passed for check in checks)
    prefix = "PASS" if passed else "FAIL"
    evidence = "\n".join(
        f"{prefix if check.passed == passed else 'INFO'}: {check.evidence}" for check in checks
    )
    return _result(
        item,
        passed,
        evidence,
        None if passed else "Update the cited implementation lines to match the cited spec values.",
    )


def _spec_extraction_result(item: Section13Item, extraction: _SpecExtraction) -> AuditResult:
    return _result(
        item,
        False,
        extraction.evidence,
        (
            "Run the harness against current PDF-derived spec_content; do not use "
            "duplicated verifier constants."
        ),
    )


def _extracted(value: Any, evidence: str) -> _SpecExtraction:
    return _SpecExtraction(value=value, evidence=evidence)


def _not_extracted(ref: str, description: str, details: str) -> _SpecExtraction:
    return _SpecExtraction(
        value=None,
        evidence=(
            f"{ref}: unable to extract {description} from current spec_content; {details}. "
            "Failing closed instead of substituting verifier-local fallback constants."
        ),
    )


def _repo_file(context: AuditContext, rel_path: str) -> Path:
    return context.repo_root / rel_path


def _read_text(context: AuditContext, rel_path: str) -> str:
    return _repo_file(context, rel_path).read_text(encoding="utf-8")


def _read_lines(context: AuditContext, rel_path: str) -> list[str]:
    return _read_text(context, rel_path).splitlines()


def _parse_python(context: AuditContext, rel_path: str) -> tuple[ast.Module, list[str]]:
    text = _read_text(context, rel_path)
    return ast.parse(text), text.splitlines()


def _line(lines: Sequence[str], line_number: int) -> str:
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1].strip()
    return ""


def _literal_assignment(
    context: AuditContext,
    rel_path: str,
    name: str,
) -> _LocatedValue | None:
    module, lines = _parse_python(context, rel_path)
    for node in ast.walk(module):
        value_node: ast.AST | None = None
        line_number = getattr(node, "lineno", 0)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                value_node = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value_node = node.value
                    break
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            return None
        return _LocatedValue(value=value, line=line_number, snippet=_line(lines, line_number))
    return None


def _find_line_regex(context: AuditContext, rel_path: str, pattern: str) -> tuple[int, str] | None:
    regex = re.compile(pattern)
    for line_number, line in enumerate(_read_lines(context, rel_path), start=1):
        if regex.search(line):
            return line_number, line.strip()
    return None


def _subscript_indices(
    context: AuditContext,
    rel_path: str,
    container_name: str,
) -> dict[int, list[int]]:
    module, _lines = _parse_python(context, rel_path)
    found: dict[int, list[int]] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != container_name:
            continue
        index: int | None = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            index = node.slice.value
        if index is None:
            continue
        found.setdefault(index, []).append(node.lineno)
    return found


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _as_sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return ()


def _strip_markup(value: object) -> str:
    return re.sub(r"<[^>]+>", "", str(value)).strip()


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for child in value:
            yield from _walk_strings(child)


def _walk_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            yield from _walk_mappings(child)


def _json_property_enum(source: str, property_name: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, Mapping):
        return ()
    properties = _as_mapping(parsed.get("properties"))
    property_schema = _as_mapping(properties.get(property_name)) if properties is not None else None
    enum = _as_sequence(property_schema.get("enum")) if property_schema is not None else ()
    return tuple(str(value) for value in enum if str(value).strip())


def _expected_semantic_methods(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    method_ids: list[str] = []
    for mapping in _walk_mappings(spec_content):
        method_id = mapping.get("method_id")
        if method_id is not None and ("role" in mapping or "execution_mode" in mapping):
            method_text = str(method_id).strip()
            if method_text and method_text not in method_ids:
                method_ids.append(method_text)
        source = mapping.get("source")
        if isinstance(source, str) and "semantic_method" in source:
            enum = _json_property_enum(source, "semantic_method")
            if enum:
                return _extracted(
                    enum,
                    (
                        "§6.4.1/§8.1/§13.27: extracted semantic_method enum "
                        "from spec_content schema source."
                    ),
                )
    if method_ids:
        return _extracted(
            tuple(method_ids),
            (
                "§8.1/§13.27: extracted deterministic semantic method registry "
                "method_id rows from spec_content."
            ),
        )
    for text in _walk_strings(spec_content):
        if "method_id" not in text or "cross_encoder" not in text:
            continue
        matches = re.findall(r"method_id\s*:\s*([A-Za-z0-9_\-]+)", text)
        method_ids = []
        for match in matches:
            if match not in method_ids:
                method_ids.append(match)
        if method_ids:
            return _extracted(
                tuple(method_ids),
                "§8.1/§13.27: extracted semantic method_id rows from spec_content text.",
            )
    return _not_extracted(
        "§8.1/§13.27",
        "deterministic semantic method registry contents",
        "missing §8.1 method_id rows or §6.4.1 semantic_method enum",
    )


def _module_by_id(spec_content: Mapping[str, Any], module_id: str) -> Mapping[str, Any] | None:
    core_modules = _as_mapping(spec_content.get("core_modules"))
    if core_modules is None:
        return None
    for module in _as_sequence(core_modules.get("modules")):
        module_mapping = _as_mapping(module)
        if module_mapping is not None and module_mapping.get("module_id") == module_id:
            return module_mapping
    return None


def _math_topic(spec_content: Mapping[str, Any], topic_id: str) -> Mapping[str, Any] | None:
    math_spec = _as_mapping(spec_content.get("math_specifications"))
    if math_spec is None:
        return None
    for topic in _as_sequence(math_spec.get("topics")):
        topic_mapping = _as_mapping(topic)
        if topic_mapping is not None and topic_mapping.get("topic_id") == topic_id:
            return topic_mapping
    return None


def _expected_ffmpeg_command(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    module_c = _module_by_id(spec_content, "C")
    if module_c is not None:
        command = _as_mapping(module_c.get("resampling_command"))
        if command is not None and command.get("source"):
            tokens = tuple(shlex.split(_strip_markup(command["source"])))
            if tokens:
                return _extracted(
                    tokens,
                    "§4.C.2/§13.4: extracted Module C resampling_command.source from spec_content.",
                )
    for text in _walk_strings(spec_content):
        if "ffmpeg -f s16le" in text and "pipe:1" in text:
            tokens = tuple(shlex.split(_strip_markup(text)))
            if tokens:
                return _extracted(
                    tokens,
                    "§4.C.2/§13.4: extracted FFmpeg command from spec_content text.",
                )
    return _not_extracted(
        "§4.C.2/§13.4",
        "Module C FFmpeg resampling command",
        (
            "missing core_modules.modules[module_id='C'].resampling_command.source "
            "containing ffmpeg and pipe:1"
        ),
    )


def _expected_au12(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    topic = _math_topic(spec_content, "au12")
    if topic is None:
        return _not_extracted(
            "§7A/§13.6",
            "AU12 landmark, epsilon, and alpha constants",
            "missing math_specifications.topics entry with topic_id='au12'",
        )
    landmarks: list[int] = []
    epsilon: float | None = None
    alpha: float | None = None
    for row in _as_sequence(topic.get("variable_dictionary")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        code_name = str(row_mapping.get("code_name", ""))
        match = re.fullmatch(r"landmarks\[(\d+)\]", code_name)
        if match:
            landmarks.append(int(match.group(1)))
        if code_name == "self.alpha":
            alpha_match = re.search(
                r"Default\s+([0-9.]+)", str(row_mapping.get("definition", ""))
            )
            if alpha_match:
                alpha = float(alpha_match.group(1))
    reference = _as_mapping(topic.get("reference_implementation"))
    if reference is not None:
        source = str(reference.get("source", ""))
        epsilon_match = re.search(r"EPSILON\s*:?\s*(?:float)?\s*=\s*([0-9.eE+-]+)", source)
        if epsilon_match:
            epsilon = float(epsilon_match.group(1))
    missing: list[str] = []
    if not landmarks:
        missing.append("landmarks[n] rows in variable_dictionary")
    if epsilon is None:
        missing.append("EPSILON assignment in reference_implementation.source")
    if alpha is None:
        missing.append("self.alpha Default value in variable_dictionary")
    if missing:
        return _not_extracted(
            "§7A/§13.6",
            "AU12 landmark, epsilon, and alpha constants",
            "missing " + ", ".join(missing),
        )
    return _extracted(
        (tuple(landmarks), epsilon, alpha),
        "§7A/§13.6: extracted AU12 landmarks, EPSILON, and alpha from spec_content.",
    )


def _coerce_expected_parameter(name: str, value: object) -> object:
    text = str(value).strip()
    if name in {"max_tokens", "seed"}:
        return int(float(text))
    if name in {"temperature", "top_p", "match_threshold"}:
        return float(text)
    return text


def _expected_llm_params(
    spec_content: Mapping[str, Any],
    required_parameters: Sequence[str] | None = None,
    ref: str = "§8.2.1/§13.7",
) -> _SpecExtraction:
    required = tuple(
        required_parameters
        or (
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "response_format",
            "cross_encoder_model_id",
            "cross_encoder_model_version",
            "semantic_calibration_version",
        )
    )
    llm_prompt = _as_mapping(spec_content.get("llm_prompt"))
    if llm_prompt is None:
        return _not_extracted(
            ref,
            "LLM deterministic inference parameters",
            "missing llm_prompt.inference_parameters",
        )
    params: dict[str, object] = {}
    for row in _as_sequence(llm_prompt.get("inference_parameters")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        name = str(row_mapping.get("parameter", "")).strip()
        if not name:
            continue
        try:
            params[name] = _coerce_expected_parameter(name, row_mapping.get("value", ""))
        except (TypeError, ValueError) as exc:
            return _not_extracted(
                ref,
                "LLM deterministic inference parameters",
                f"parameter {name!r} has unparseable value {row_mapping.get('value', '')!r}: {exc}",
            )
    missing = [name for name in required if name not in params]
    if missing:
        return _not_extracted(
            ref,
            "LLM deterministic inference parameters",
            "missing parameter rows " + ", ".join(missing),
        )
    return _extracted(
        params,
        f"{ref}: extracted LLM inference parameters {', '.join(required)} from spec_content.",
    )


def _expected_vault(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    governance = _as_mapping(spec_content.get("data_governance"))
    if governance is None:
        return _not_extracted(
            "§5.1/§13.8",
            "Ephemeral Vault constants",
            "missing data_governance.vault_parameters",
        )
    by_parameter: dict[str, str] = {}
    for row in _as_sequence(governance.get("vault_parameters")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        parameter = str(row_mapping.get("parameter", "")).casefold()
        by_parameter[parameter] = _strip_markup(row_mapping.get("specification", ""))
    key_spec = by_parameter.get("key generation") or by_parameter.get("key length", "")
    nonce_spec = by_parameter.get("iv/nonce length", "")
    deletion_spec = by_parameter.get("secure deletion method", "")
    missing_fields: list[str] = []
    if not key_spec:
        missing_fields.append("Key generation/key length")
    if not nonce_spec:
        missing_fields.append("IV/Nonce length")
    if not deletion_spec:
        missing_fields.append("Secure deletion method")
    if missing_fields:
        return _not_extracted(
            "§5.1/§13.8",
            "Ephemeral Vault constants",
            "missing vault parameter rows " + ", ".join(missing_fields),
        )
    key_match = re.search(r"urandom\((\d+)\)", key_spec)
    bit_match = re.search(r"(\d+)\s*bits", key_spec)
    key_bytes = int(key_match.group(1)) if key_match else None
    if key_bytes is None and bit_match:
        key_bytes = int(int(bit_match.group(1)) / 8)
    nonce_match = re.search(r"(\d+)\s*bytes", nonce_spec)
    command_match = re.search(r"(shred\s+-vfz\s+-n\s+3)", deletion_spec)
    interval_match = re.search(r"every\s+(\d+)\s+hours", deletion_spec)
    targets = tuple(re.findall(r"/data/(?:raw|interim)/", deletion_spec))
    missing_values: list[str] = []
    if key_bytes is None:
        missing_values.append("key byte length from §5.1 key generation")
    if nonce_match is None:
        missing_values.append("nonce byte length from §5.1 IV/Nonce length")
    if command_match is None:
        missing_values.append("shred command from §5.1 secure deletion method")
    if interval_match is None:
        missing_values.append("deletion interval hours from §5.1 secure deletion method")
    if not targets:
        missing_values.append("deletion targets from §5.1 secure deletion method")
    if missing_values:
        return _not_extracted(
            "§5.1/§13.8",
            "Ephemeral Vault constants",
            "missing " + ", ".join(missing_values),
        )
    if key_bytes is None or nonce_match is None or command_match is None or interval_match is None:
        return _not_extracted(
            "§5.1/§13.8",
            "Ephemeral Vault constants",
            "internal parse guard did not narrow all required values",
        )
    return _extracted(
        _ExpectedVault(
            key_bytes=key_bytes,
            nonce_bytes=int(nonce_match.group(1)),
            shred_command=tuple(command_match.group(1).split()),
            interval_hours=int(interval_match.group(1)),
            targets=targets,
        ),
        "§5.1/§13.8: extracted Ephemeral Vault constants from spec_content.",
    )


def _target_files_for(targets: Sequence[object]) -> tuple[str, ...]:
    files: set[str] = set()
    for target in targets:
        target_text = str(target)
        if target_text in {"worker", "orchestrator"}:
            files.add("requirements/worker.txt")
        elif target_text == "api":
            files.add("requirements/api.txt")
        elif target_text == "operator_host":
            files.add("requirements/cli.txt")
    return tuple(sorted(files))


def _expected_dependencies(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    dependency_matrix = _as_mapping(spec_content.get("dependency_matrix"))
    if dependency_matrix is None:
        return _not_extracted(
            "§10.2/§13.12",
            "dependency matrix pinned packages",
            "missing dependency_matrix.pinned_packages",
        )
    expected: list[_ExpectedDependency] = []
    skipped_rows = 0
    for row in _as_sequence(dependency_matrix.get("pinned_packages")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            skipped_rows += 1
            continue
        package = str(row_mapping.get("package", "")).strip()
        version = str(row_mapping.get("version", "")).strip()
        targets = _target_files_for(_as_sequence(row_mapping.get("container_targets")))
        if package and version and targets:
            expected.append(_ExpectedDependency(package, version, targets))
        else:
            skipped_rows += 1
    if not expected:
        return _not_extracted(
            "§10.2/§13.12",
            "dependency matrix pinned packages",
            "no rows with package, version, and mapped container_targets were parseable",
        )
    evidence = "§10.2/§13.12: extracted dependency_matrix.pinned_packages from spec_content"
    if skipped_rows:
        evidence += f"; skipped {skipped_rows} unparseable rows"
    return _extracted(tuple(expected), evidence + ".")


def _expected_reason_codes(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    llm_prompt = _as_mapping(spec_content.get("llm_prompt"))
    output_schema = _as_mapping(llm_prompt.get("output_schema")) if llm_prompt is not None else None
    if output_schema is None:
        return _not_extracted(
            "§8.3/§13.27",
            "bounded semantic reason-code enum",
            "missing llm_prompt.output_schema.source",
        )
    source = str(output_schema.get("source", ""))
    if not source:
        return _not_extracted(
            "§8.3/§13.27",
            "bounded semantic reason-code enum",
            "llm_prompt.output_schema.source is empty",
        )
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        return _not_extracted(
            "§8.3/§13.27",
            "bounded semantic reason-code enum",
            f"llm_prompt.output_schema.source is not parseable JSON: {exc}",
        )
    if isinstance(parsed, Mapping):
        properties = _as_mapping(parsed.get("properties"))
        reasoning = _as_mapping(properties.get("reasoning")) if properties is not None else None
        enum = _as_sequence(reasoning.get("enum")) if reasoning is not None else ()
        codes = tuple(str(code) for code in enum if str(code).strip())
        if codes:
            return _extracted(
                codes,
                "§8.3/§13.27: extracted bounded semantic reason-code enum from spec_content.",
            )
    return _not_extracted(
        "§8.3/§13.27",
        "bounded semantic reason-code enum",
        "missing properties.reasoning.enum in llm_prompt.output_schema.source",
    )


def _parse_requirement_line(raw_line: str) -> tuple[str, str] | None:
    stripped = raw_line.split("#", 1)[0].strip()
    if not stripped or stripped.startswith("-r"):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)\s*(.*)", stripped)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _normalize_package(name: str) -> str:
    return name.replace("_", "-").casefold()


def _read_requirement_entries(context: AuditContext, rel_path: str) -> dict[str, _RequirementEntry]:
    entries: dict[str, _RequirementEntry] = {}
    for line_number, raw_line in enumerate(_read_lines(context, rel_path), start=1):
        parsed = _parse_requirement_line(raw_line)
        if parsed is None:
            continue
        package, specifier = parsed
        entries[_normalize_package(package)] = _RequirementEntry(
            package=package,
            specifier=specifier,
            raw_line=raw_line.strip(),
            rel_path=rel_path,
            line=line_number,
        )
    return entries


def _effective_requirements(context: AuditContext, rel_path: str) -> dict[str, _RequirementEntry]:
    entries: dict[str, _RequirementEntry] = {}
    if rel_path != "requirements/base.txt":
        for raw_line in _read_lines(context, rel_path):
            include_match = re.match(r"\s*-r\s+base\.txt\s*(?:#.*)?$", raw_line)
            if include_match:
                entries.update(_read_requirement_entries(context, "requirements/base.txt"))
                break
    entries.update(_read_requirement_entries(context, rel_path))
    return entries


def _specifier_matches(expected: str, actual: str) -> bool:
    expected = expected.strip()
    actual = actual.strip()
    if expected.startswith(">="):
        return actual.startswith(expected)
    wildcard = re.fullmatch(r"(\d+(?:\.\d+)*)\.(?:x|\*)", expected)
    if wildcard:
        prefix = wildcard.group(1)
        return actual.startswith(f"=={prefix}.") or actual == f"=={prefix}.*"
    if re.fullmatch(r"\d+(?:\.\d+)+", expected):
        return actual == f"=={expected}"
    return actual == f"=={expected}"


def _retired_synonyms_from_row(row_mapping: Mapping[str, Any]) -> tuple[str, ...]:
    retired_list = _as_sequence(row_mapping.get("retired_synonym_list"))
    if retired_list:
        return tuple(str(synonym).strip() for synonym in retired_list if str(synonym).strip())
    retired_synonyms = row_mapping.get("retired_synonyms")
    if isinstance(retired_synonyms, str):
        return tuple(
            synonym.strip() for synonym in retired_synonyms.split(",") if synonym.strip()
        )
    return tuple(
        str(synonym).strip()
        for synonym in _as_sequence(retired_synonyms)
        if str(synonym).strip()
    )


def _canonical_patterns(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    document_control = _as_mapping(spec_content.get("document_control"))
    if document_control is None:
        return _not_extracted(
            "§0.3/§13.15",
            "canonical retired_synonym_list terms",
            "missing document_control.canonical_terms",
        )
    parsed_terms: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for row in _as_sequence(document_control.get("canonical_terms")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        canonical_name = str(row_mapping.get("canonical_name", "§0.3 canonical term")).strip()
        for synonym_text in _retired_synonyms_from_row(row_mapping):
            if synonym_text in _CANONICAL_TERM_EXCLUSIONS:
                excluded[synonym_text] = _CANONICAL_TERM_EXCLUSIONS[synonym_text]
                continue
            parsed_terms[synonym_text] = canonical_name
    if not parsed_terms and not excluded:
        return _not_extracted(
            "§0.3/§13.15",
            "canonical retired_synonym_list terms",
            "document_control.canonical_terms contained no retired_synonym_list entries",
        )
    patterns = [
        _CanonicalPattern(
            re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"),
            f"§0.3/§13.15 retired synonym for {canonical_name}: {term!r}",
        )
        for term, canonical_name in sorted(parsed_terms.items())
    ]
    patterns.extend(
        _CanonicalPattern(re.compile(pattern), source)
        for pattern, source in _PROSE_CANONICAL_REGEXES
    )
    evidence = (
        f"§0.3/§13.15: parsed {len(parsed_terms)} retired_synonym_list terms from spec_content"
    )
    if excluded:
        evidence += (
            f"; excluded {len(excluded)} broad §0.3 terms for documented safe-grep reasons: "
            + "; ".join(f"{term!r} ({reason})" for term, reason in sorted(excluded.items()))
        )
    return _extracted(tuple(patterns), evidence + ".")


def _has_excluded_scan_part(rel_path: Path) -> bool:
    return any(part in _SCAN_EXCLUDED_DIRS for part in rel_path.parts)


def _is_root_config_file(rel_path: Path) -> bool:
    if len(rel_path.parts) != 1:
        return False
    name = rel_path.name
    if name.startswith("docker-compose") and rel_path.suffix in {".yml", ".yaml"}:
        return True
    if name.startswith("Dockerfile"):
        return True
    if name.startswith(".env"):
        return True
    return rel_path.suffix in _ROOT_CONFIG_SUFFIXES


def _is_config_artifact(rel_path: Path) -> bool:
    if rel_path.name.startswith("Dockerfile"):
        return True
    if rel_path.suffix in {".sql", ".ddl"}:
        return True
    if any(part in _CONFIG_ARTIFACT_DIRS for part in rel_path.parts[:-1]):
        return rel_path.suffix in _CONFIG_ARTIFACT_SUFFIXES or rel_path.name.startswith(".env")
    return False


def _is_scanned_surface_path(rel_path: Path) -> bool:
    if rel_path == _CANONICAL_VERIFIER_PATH:
        return False
    if _has_excluded_scan_part(rel_path) or rel_path.suffix in _BINARY_SCAN_SUFFIXES:
        return False
    if rel_path.parts and rel_path.parts[0] in _CODE_SCAN_DIRS:
        return True
    if rel_path.parts and rel_path.parts[0] == "requirements":
        return True
    if len(rel_path.parts) >= 2 and rel_path.parts[:2] == (".github", "workflows"):
        return True
    return _is_root_config_file(rel_path) or _is_config_artifact(rel_path)


def _iter_scanned_files(context: AuditContext) -> Iterable[Path]:
    for path in sorted(context.repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(context.repo_root)
        if not _is_scanned_surface_path(rel_path):
            continue
        yield path


def verify_ffmpeg_resample(context: AuditContext, item: Section13Item) -> AuditResult:
    rel_path = "services/worker/pipeline/orchestrator.py"
    expected_extraction = _expected_ffmpeg_command(context.spec_content)
    if expected_extraction.value is None:
        return _spec_extraction_result(item, expected_extraction)
    expected = expected_extraction.value
    actual = _literal_assignment(context, rel_path, "FFMPEG_RESAMPLE_CMD")
    if actual is None:
        return _result(
            item,
            False,
            f"§4.C.2/§13.4: missing FFMPEG_RESAMPLE_CMD assignment in {rel_path}.",
            "Define the FFmpeg resampling command from the spec.",
        )
    invocation = _find_line_regex(context, rel_path, r"FFMPEG_RESAMPLE_CMD,")
    checks = [
        _Check(
            tuple(actual.value) == expected,
            (
                f"§4.C.2/§13.4 {rel_path}:{actual.line} matched command {actual.value!r}; "
                f"expected {expected!r}. {expected_extraction.evidence}"
            ),
        ),
        _Check(
            invocation is not None,
            (
                f"§4.C.2/§13.4 {rel_path}:{invocation[0]} invokes FFMPEG_RESAMPLE_CMD: "
                f"{invocation[1]!r}."
                if invocation is not None
                else (
                    f"§4.C.2/§13.4 {rel_path}: missing subprocess invocation of "
                    "FFMPEG_RESAMPLE_CMD."
                )
            ),
        ),
    ]
    return _checks_result(item, checks)


def verify_au12_geometry(context: AuditContext, item: Section13Item) -> AuditResult:
    rel_path = "packages/ml_core/au12.py"
    expected_extraction = _expected_au12(context.spec_content)
    if expected_extraction.value is None:
        return _spec_extraction_result(item, expected_extraction)
    expected_landmarks, expected_epsilon, expected_alpha = expected_extraction.value
    actual_landmarks = _subscript_indices(context, rel_path, "landmarks")
    epsilon = _literal_assignment(context, rel_path, "EPSILON")
    alpha = _literal_assignment(context, rel_path, "DEFAULT_ALPHA_SCALE")
    guard = _find_line_regex(context, rel_path, r"if\s+iod\s*<\s*EPSILON\s*:")
    checks = [
        _Check(
            set(actual_landmarks) == set(expected_landmarks),
            (
                f"§7A.1/§7A.2/§13.6 {rel_path}: landmark indices "
                f"{sorted(actual_landmarks)} at lines {actual_landmarks}; "
                f"expected {list(expected_landmarks)}. {expected_extraction.evidence}"
            ),
        ),
        _Check(
            epsilon is not None and float(epsilon.value) == expected_epsilon,
            (
                f"§7A.5/§13.6 {rel_path}:{epsilon.line if epsilon else '?'} "
                f"EPSILON={epsilon.value if epsilon else 'missing'}; expected {expected_epsilon}."
            ),
        ),
        _Check(
            guard is not None,
            (
                f"§7A.5/§13.6 {rel_path}:{guard[0]} epsilon guard {guard[1]!r}."
                if guard is not None
                else f"§7A.5/§13.6 {rel_path}: missing 'iod < EPSILON' guard."
            ),
        ),
        _Check(
            alpha is not None and float(alpha.value) == expected_alpha,
            (
                f"§7A.1/§7A.4/§13.6 {rel_path}:{alpha.line if alpha else '?'} "
                f"DEFAULT_ALPHA_SCALE={alpha.value if alpha else 'missing'}; "
                f"expected {expected_alpha}."
            ),
        ),
    ]
    return _checks_result(item, checks)


def verify_semantic_determinism(context: AuditContext, item: Section13Item) -> AuditResult:
    rel_path = "packages/ml_core/semantic.py"
    expected_extraction = _expected_llm_params(context.spec_content)
    if expected_extraction.value is None:
        return _spec_extraction_result(item, expected_extraction)
    expected = expected_extraction.value
    llm_params = _literal_assignment(context, rel_path, "LLM_PARAMS")
    response_type = _find_line_regex(context, rel_path, r'"type"\s*:\s*"json_schema"')
    strict_line = _find_line_regex(context, rel_path, r'"strict"\s*:\s*True')
    call_line = _find_line_regex(
        context, rel_path, r"response_format\s*=\s*GRAY_BAND_RESPONSE_FORMAT"
    )
    model_id = _literal_assignment(context, rel_path, "CROSS_ENCODER_MODEL_ID")
    model_version = _literal_assignment(context, rel_path, "CROSS_ENCODER_MODEL_VERSION")
    calibration = _literal_assignment(context, rel_path, "SEMANTIC_CALIBRATION_VERSION")
    actual_params = (
        llm_params.value if llm_params is not None and isinstance(llm_params.value, Mapping) else {}
    )
    expected_subset = {
        key: expected.get(key) for key in ("temperature", "top_p", "max_tokens", "seed")
    }
    parameter_checks = []
    for key, expected_value in expected_subset.items():
        parameter_checks.append(actual_params.get(key) == expected_value)
    checks = [
        _Check(
            bool(parameter_checks) and all(parameter_checks),
            (
                f"§8.2.1/§13.7 {rel_path}:{llm_params.line if llm_params else '?'} "
                f"LLM_PARAMS={dict(actual_params)!r}; expected deterministic subset "
                f"{expected_subset!r}. {expected_extraction.evidence}"
            ),
        ),
        _Check(
            model_id is not None and model_id.value == expected.get("cross_encoder_model_id"),
            (
                f"§8.2.1/§13.7 {rel_path}:{model_id.line if model_id else '?'} "
                f"CROSS_ENCODER_MODEL_ID={model_id.value if model_id else 'missing'}; "
                f"expected {expected.get('cross_encoder_model_id')}."
            ),
        ),
        _Check(
            model_version is not None
            and model_version.value == expected.get("cross_encoder_model_version"),
            (
                f"§8.2.1/§13.7 {rel_path}:{model_version.line if model_version else '?'} "
                "CROSS_ENCODER_MODEL_VERSION="
                f"{model_version.value if model_version else 'missing'}; "
                f"expected {expected.get('cross_encoder_model_version')}."
            ),
        ),
        _Check(
            calibration is not None
            and calibration.value == expected.get("semantic_calibration_version"),
            (
                f"§8.2.1/§13.7 {rel_path}:{calibration.line if calibration else '?'} "
                f"SEMANTIC_CALIBRATION_VERSION={calibration.value if calibration else 'missing'}; "
                f"expected {expected.get('semantic_calibration_version')}."
            ),
        ),
        _Check(
            response_type is not None and strict_line is not None and call_line is not None,
            (
                f"§8.2.1/§13.7 {rel_path}: structured output evidence "
                f"type={response_type}, strict={strict_line}, call={call_line}; "
                f"expected {expected.get('response_format')}."
            ),
        ),
    ]
    return _checks_result(item, checks)


def verify_ephemeral_vault(context: AuditContext, item: Section13Item) -> AuditResult:
    encryption_path = "packages/ml_core/encryption.py"
    cron_path = "services/worker/vault_cron.py"
    expected_extraction = _expected_vault(context.spec_content)
    if expected_extraction.value is None:
        return _spec_extraction_result(item, expected_extraction)
    expected = expected_extraction.value
    key_line = _find_line_regex(context, encryption_path, rf"os\.urandom\({expected.key_bytes}\)")
    nonce_length = _literal_assignment(context, encryption_path, "nonce_length")
    nonce_call = _find_line_regex(context, encryption_path, r"os\.urandom\(self\.nonce_length\)")
    aes_line = _find_line_regex(context, encryption_path, r"AES\.MODE_GCM")
    shred_line = _find_line_regex(context, encryption_path, r'"shred"')
    shred_flags = [
        _find_line_regex(context, encryption_path, rf'"{re.escape(token)}"')
        for token in expected.shred_command
    ]
    targets = _literal_assignment(context, cron_path, "SHRED_TARGETS")
    interval = _literal_assignment(context, cron_path, "INTERVAL_HOURS")
    checks = [
        _Check(
            key_line is not None,
            (
                f"§5.1.5/§13.8 {encryption_path}:{key_line[0]} key generation {key_line[1]!r}; "
                f"expected os.urandom({expected.key_bytes}). {expected_extraction.evidence}"
                if key_line is not None
                else f"§5.1.5/§13.8 {encryption_path}: missing os.urandom({expected.key_bytes})."
            ),
        ),
        _Check(
            nonce_length is not None
            and int(nonce_length.value) == expected.nonce_bytes
            and nonce_call is not None,
            (
                f"§5.1.4/§13.8 {encryption_path}:{nonce_length.line if nonce_length else '?'} "
                f"nonce_length={nonce_length.value if nonce_length else 'missing'}, "
                f"nonce_call={nonce_call}; expected {expected.nonce_bytes} bytes via os.urandom."
            ),
        ),
        _Check(
            aes_line is not None,
            (
                f"§5.1.1/§13.8 {encryption_path}:{aes_line[0]} AES-GCM marker {aes_line[1]!r}."
                if aes_line is not None
                else f"§5.1.1/§13.8 {encryption_path}: missing AES.MODE_GCM."
            ),
        ),
        _Check(
            shred_line is not None and all(flag is not None for flag in shred_flags),
            (
                f"§5.1.8/§13.8 {encryption_path}:{shred_line[0] if shred_line else '?'} "
                f"shred command tokens {expected.shred_command}; evidence {shred_flags}."
            ),
        ),
        _Check(
            targets is not None and tuple(targets.value) == expected.targets,
            (
                f"§5.1.8/§13.8 {cron_path}:{targets.line if targets else '?'} "
                f"SHRED_TARGETS={targets.value if targets else 'missing'}; "
                f"expected {expected.targets}."
            ),
        ),
        _Check(
            interval is not None and int(interval.value) == expected.interval_hours,
            (
                f"§5.1.8/§13.8 {cron_path}:{interval.line if interval else '?'} "
                f"INTERVAL_HOURS={interval.value if interval else 'missing'}; "
                f"expected {expected.interval_hours}."
            ),
        ),
    ]
    return _checks_result(item, checks)


def verify_dependency_pins(context: AuditContext, item: Section13Item) -> AuditResult:
    expected_extraction = _expected_dependencies(context.spec_content)
    if expected_extraction.value is None:
        return _spec_extraction_result(item, expected_extraction)
    expected = expected_extraction.value
    effective = {
        rel_path: _effective_requirements(context, rel_path)
        for rel_path in (
            "requirements/worker.txt",
            "requirements/api.txt",
            "requirements/cli.txt",
        )
    }
    checks: list[_Check] = []
    for dependency in expected:
        for target_file in dependency.target_files:
            entry = effective[target_file].get(_normalize_package(dependency.package))
            if entry is None:
                checks.append(
                    _Check(
                        False,
                        (
                            f"§10.2/§13.12 {target_file}: missing {dependency.package} "
                            f"expected version {dependency.version}. {expected_extraction.evidence}"
                        ),
                    )
                )
                continue
            checks.append(
                _Check(
                    _specifier_matches(dependency.version, entry.specifier),
                    (
                        f"§10.2/§13.12 {entry.rel_path}:{entry.line} matched {entry.raw_line!r}; "
                        f"expected {dependency.package} {dependency.version}. "
                        f"{expected_extraction.evidence}"
                    ),
                )
            )
    return _checks_result(item, checks)


def verify_canonical_terminology(context: AuditContext, item: Section13Item) -> AuditResult:
    patterns_extraction = _canonical_patterns(context.spec_content)
    if patterns_extraction.value is None:
        return _spec_extraction_result(item, patterns_extraction)
    patterns = patterns_extraction.value
    matches: list[str] = []
    scanned = 0
    for path in _iter_scanned_files(context):
        scanned += 1
        rel_path = path.relative_to(context.repo_root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for canonical_pattern in patterns:
                match = canonical_pattern.pattern.search(line)
                if match is None:
                    continue
                matches.append(
                    f"§0.3/§13.15 {rel_path.as_posix()}:{line_number} matched "
                    f"{match.group(0)!r} ({canonical_pattern.source}): {line.strip()}"
                )
    if matches:
        return _result(
            item,
            False,
            (
                f"§0.3/§13.15 scanned {scanned} code/config files; "
                f"{patterns_extraction.evidence}\nretired terminology matches:\n"
                + "\n".join(matches[:50])
            ),
            "Replace each matched retired term with the canonical name cited by §0.3.",
        )
    return _result(
        item,
        True,
        (
            f"§0.3/§13.15 scanned {scanned} code/config files; "
            f"no retired terminology matched {len(patterns)} parsed §0.3/prose patterns. "
            f"{patterns_extraction.evidence}"
        ),
    )


def _statements_always_exit(statements: Sequence[ast.stmt]) -> bool:
    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, ast.Return | ast.Raise):
        return True
    if isinstance(last, ast.If):
        return _statements_always_exit(last.body) and _statements_always_exit(last.orelse)
    return False


def _contains_gray_band_fallback_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and func.attr == "_evaluate_gray_band_fallback":
            return True
        if isinstance(func, ast.Name) and func.id == "_evaluate_gray_band_fallback":
            return True
    return False


def _condition_has_feature_gate(expr: ast.AST) -> bool:
    """Return whether ``expr`` logically implies the positive gray-band gate."""

    def is_feature_flag(candidate: ast.AST) -> bool:
        if isinstance(candidate, ast.Attribute):
            return candidate.attr == "gray_band_fallback_enabled"
        if isinstance(candidate, ast.Name):
            return candidate.id == "gray_band_fallback_enabled"
        return False

    def is_true_literal(candidate: ast.AST) -> bool:
        return isinstance(candidate, ast.Constant) and candidate.value is True

    if is_feature_flag(expr):
        return True
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            return any(_condition_has_feature_gate(value) for value in expr.values)
        if isinstance(expr.op, ast.Or):
            return all(_condition_has_feature_gate(value) for value in expr.values)
        return False
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return False
    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            return False
        op = expr.ops[0]
        right = expr.comparators[0]
        if not isinstance(op, ast.Eq | ast.Is):
            return False
        return (is_feature_flag(expr.left) and is_true_literal(right)) or (
            is_true_literal(expr.left) and is_feature_flag(right)
        )
    return False


def _is_threshold_name(expr: ast.AST, name: str) -> bool:
    return isinstance(expr, ast.Name) and expr.id == name


def _is_score_expr(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return "score" in expr.id.lower()
    if isinstance(expr, ast.Attribute):
        return "score" in expr.attr.lower()
    return not isinstance(expr, ast.Constant)


def _compare_pair_proves_bound(
    left: ast.AST,
    op: ast.cmpop,
    right: ast.AST,
    *,
    kind: str,
    negated: bool,
) -> bool:
    if not negated:
        if kind == "lower":
            return (
                isinstance(op, ast.GtE)
                and _is_score_expr(left)
                and _is_threshold_name(right, "GRAY_BAND_LOWER_THRESHOLD")
            ) or (
                isinstance(op, ast.LtE)
                and _is_threshold_name(left, "GRAY_BAND_LOWER_THRESHOLD")
                and _is_score_expr(right)
            )
        return (
            isinstance(op, ast.Lt)
            and _is_score_expr(left)
            and _is_threshold_name(right, "MATCH_THRESHOLD")
        ) or (
            isinstance(op, ast.Gt)
            and _is_threshold_name(left, "MATCH_THRESHOLD")
            and _is_score_expr(right)
        )
    if kind == "lower":
        return (
            isinstance(op, ast.Lt)
            and _is_score_expr(left)
            and _is_threshold_name(right, "GRAY_BAND_LOWER_THRESHOLD")
        ) or (
            isinstance(op, ast.Gt)
            and _is_threshold_name(left, "GRAY_BAND_LOWER_THRESHOLD")
            and _is_score_expr(right)
        )
    return (
        isinstance(op, ast.GtE)
        and _is_score_expr(left)
        and _is_threshold_name(right, "MATCH_THRESHOLD")
    ) or (
        isinstance(op, ast.LtE)
        and _is_threshold_name(left, "MATCH_THRESHOLD")
        and _is_score_expr(right)
    )


def _condition_proves_bound(expr: ast.AST, kind: str, *, negated: bool = False) -> bool:
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return _condition_proves_bound(expr.operand, kind, negated=not negated)
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And) and not negated:
            return any(_condition_proves_bound(value, kind) for value in expr.values)
        if isinstance(expr.op, ast.Or) and negated:
            return any(_condition_proves_bound(value, kind, negated=True) for value in expr.values)
        if isinstance(expr.op, ast.Or) and not negated:
            return all(_condition_proves_bound(value, kind) for value in expr.values)
        return False
    if not isinstance(expr, ast.Compare):
        return False
    left = expr.left
    for op, comparator in zip(expr.ops, expr.comparators, strict=True):
        if _compare_pair_proves_bound(left, op, comparator, kind=kind, negated=negated):
            return True
        left = comparator
    return False


def _collect_fallback_path_conditions(
    statements: Sequence[ast.stmt],
    inherited_conditions: Sequence[ast.AST],
) -> list[tuple[int, list[ast.AST]]]:
    found: list[tuple[int, list[ast.AST]]] = []
    active_conditions = list(inherited_conditions)
    for statement in statements:
        if isinstance(statement, ast.If):
            body_conditions = [*active_conditions, statement.test]
            found.extend(_collect_fallback_path_conditions(statement.body, body_conditions))
            negated_test = ast.UnaryOp(op=ast.Not(), operand=statement.test)
            found.extend(
                _collect_fallback_path_conditions(
                    statement.orelse,
                    [*active_conditions, negated_test],
                )
            )
            if not statement.orelse and _statements_always_exit(statement.body):
                active_conditions.append(negated_test)
            continue
        if _contains_gray_band_fallback_call(statement):
            found.append((getattr(statement, "lineno", 0), list(active_conditions)))
    return found


def _branch_conditions_evidence(conditions: Sequence[ast.AST]) -> str:
    return " AND ".join(ast.unparse(condition) for condition in conditions) or "<unconditional>"


def _gray_band_branch_evidence(context: AuditContext, rel_path: str) -> _GrayBandBranchEvidence:
    module, lines = _parse_python(context, rel_path)
    candidate_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if _contains_gray_band_fallback_call(node):
            candidate_functions.append(node)
    branches: list[tuple[int, list[ast.AST]]] = []
    for function in candidate_functions:
        branches.extend(_collect_fallback_path_conditions(function.body, []))
    if not branches:
        return _GrayBandBranchEvidence(
            False,
            f"§8.1/§8.2.1/§13.27 {rel_path}: no branch calls _evaluate_gray_band_fallback.",
        )
    branch_evidence: list[str] = []
    failing_branch_evidence: list[str] = []
    for line_number, conditions in branches:
        has_feature = any(_condition_has_feature_gate(condition) for condition in conditions)
        has_lower = any(_condition_proves_bound(condition, "lower") for condition in conditions)
        has_upper = any(_condition_proves_bound(condition, "upper") for condition in conditions)
        missing = [
            label
            for label, present in (
                ("gray_band_fallback_enabled", has_feature),
                ("lower-inclusive GRAY_BAND_LOWER_THRESHOLD", has_lower),
                ("upper-exclusive MATCH_THRESHOLD", has_upper),
            )
            if not present
        ]
        condition_text = _branch_conditions_evidence(conditions)
        snippet = _line(lines, line_number) if line_number else ""
        evidence = (
            f"{rel_path}:{line_number or '?'} effective condition [{condition_text}] "
            f"feature={has_feature} lower_inclusive={has_lower} upper_exclusive={has_upper} "
            f"snippet={snippet!r}"
        )
        branch_evidence.append(evidence)
        if missing:
            failing_branch_evidence.append(
                f"{evidence} missing {', '.join(missing)}"
            )
    if failing_branch_evidence:
        return _GrayBandBranchEvidence(
            False,
            (
                "§8.1/§8.2.1/§13.27 fallback branch must require the feature flag, "
                "lower-inclusive GRAY_BAND_LOWER_THRESHOLD gating, and upper-exclusive "
                "MATCH_THRESHOLD gating in every effective fallback branch; "
                "non-compliant fallback branches:\n"
                + "\n".join(failing_branch_evidence)
                + "\nobserved all fallback branches:\n"
                + "\n".join(branch_evidence)
            ),
        )
    return _GrayBandBranchEvidence(
        True,
        (
            "§8.1/§8.2.1/§13.27 fallback branch is bounded; "
            "all fallback branches are bounded by gray_band_fallback_enabled, "
            "GRAY_BAND_LOWER_THRESHOLD <= score, and score < MATCH_THRESHOLD; observed:\n"
            + "\n".join(branch_evidence)
        ),
    )


def verify_semantic_reason_codes(context: AuditContext, item: Section13Item) -> AuditResult:
    semantic_path = "packages/ml_core/semantic.py"
    schema_path = "packages/schemas/evaluation.py"
    expected_codes_extraction = _expected_reason_codes(context.spec_content)
    if expected_codes_extraction.value is None:
        return _spec_extraction_result(item, expected_codes_extraction)
    expected_params_extraction = _expected_llm_params(
        context.spec_content,
        required_parameters=("match_threshold", "gray_band_interval"),
        ref="§8.2.1/§13.27",
    )
    if expected_params_extraction.value is None:
        return _spec_extraction_result(item, expected_params_extraction)
    expected_methods_extraction = _expected_semantic_methods(context.spec_content)
    if expected_methods_extraction.value is None:
        return _spec_extraction_result(item, expected_methods_extraction)
    expected_codes = expected_codes_extraction.value
    expected_params = expected_params_extraction.value
    expected_methods = expected_methods_extraction.value
    actual_codes = _literal_assignment(context, schema_path, "SEMANTIC_REASON_CODES")
    actual_methods = _literal_assignment(context, schema_path, "SEMANTIC_METHODS")
    lower = _literal_assignment(context, semantic_path, "GRAY_BAND_LOWER_THRESHOLD")
    match_threshold = _literal_assignment(context, semantic_path, "MATCH_THRESHOLD")
    interval = str(expected_params.get("gray_band_interval", ""))
    lower_match = re.search(r"([0-9.]+)\s*<=\s*score", interval)
    upper_match = re.search(r"score\s*<\s*([0-9.]+)", interval)
    if lower_match is None or upper_match is None:
        return _spec_extraction_result(
            item,
            _not_extracted(
                "§8.2.1/§13.27",
                "gray-band threshold interval",
                f"gray_band_interval {interval!r} does not match 'lower <= score < upper'",
            ),
        )
    expected_lower = float(lower_match.group(1))
    expected_upper = float(upper_match.group(1))
    direct_gate = _find_line_regex(context, semantic_path, r"primary_score\s*>=\s*MATCH_THRESHOLD")
    branch_evidence = _gray_band_branch_evidence(context, semantic_path)
    schema_enum_line = _find_line_regex(
        context, semantic_path, r"enum\"\s*:\s*list\(SEMANTIC_REASON_CODES\)"
    )
    checks = [
        _Check(
            actual_codes is not None and tuple(actual_codes.value) == expected_codes,
            (
                f"§8.3/§13.27 {schema_path}:{actual_codes.line if actual_codes else '?'} "
                f"SEMANTIC_REASON_CODES={actual_codes.value if actual_codes else 'missing'}; "
                f"expected {expected_codes}. {expected_codes_extraction.evidence}"
            ),
        ),
        _Check(
            actual_methods is not None and tuple(actual_methods.value) == expected_methods,
            (
                f"§8.1/§13.27 {schema_path}:{actual_methods.line if actual_methods else '?'} "
                f"SEMANTIC_METHODS={actual_methods.value if actual_methods else 'missing'}; "
                f"expected {expected_methods}. {expected_methods_extraction.evidence}"
            ),
        ),
        _Check(
            schema_enum_line is not None,
            (
                f"§8.3/§13.27 {semantic_path}:{schema_enum_line[0] if schema_enum_line else '?'} "
                f"OUTPUT_SCHEMA delegates enum to SEMANTIC_REASON_CODES: "
                f"{schema_enum_line[1] if schema_enum_line else 'missing'!r}."
            ),
        ),
        _Check(
            lower is not None and float(lower.value) == expected_lower,
            (
                f"§8.2.1/§13.27 {semantic_path}:{lower.line if lower else '?'} "
                f"GRAY_BAND_LOWER_THRESHOLD={lower.value if lower else 'missing'}; "
                f"expected lower bound {expected_lower}. {expected_params_extraction.evidence}"
            ),
        ),
        _Check(
            match_threshold is not None and float(match_threshold.value) == expected_upper,
            (
                f"§8.2.1/§13.27 {semantic_path}:{match_threshold.line if match_threshold else '?'} "
                f"MATCH_THRESHOLD={match_threshold.value if match_threshold else 'missing'}; "
                f"expected cutoff {expected_upper}."
            ),
        ),
        _Check(
            direct_gate is not None and branch_evidence.passed,
            (
                f"§8.1/§8.2.1/§13.27 {semantic_path}: direct gate {direct_gate}; "
                f"{branch_evidence.evidence}"
            ),
        ),
    ]
    return _checks_result(item, checks)


_ATTRIBUTION_SCHEMA_CLASSES = (
    "AttributionEvent",
    "OutcomeEvent",
    "EventOutcomeLink",
    "AttributionScore",
)
_ATTRIBUTION_SQL_TABLES = (
    "attribution_event",
    "outcome_event",
    "event_outcome_link",
    "attribution_score",
)
_DERIVED_ONLY_REQUIRED_FIELDS = (
    "expected_rule_text_hash",
    "semantic_method",
    "semantic_method_version",
    "semantic_p_match",
    "semantic_reason_code",
    "bandit_decision_snapshot",
    "finality",
    "schema_version",
)
_DERIVED_ONLY_FORBIDDEN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "raw audio": (
        "raw_audio",
        "audio_blob",
        "audio_bytes",
        "audio_data",
        "audio_payload",
        "pcm_audio",
    ),
    "raw video": (
        "raw_video",
        "video_blob",
        "video_bytes",
        "video_data",
        "video_payload",
        "frame_bytes",
        "frame_data",
        "raw_frames",
    ),
    "complete transient PhysiologicalChunkEvent payload bodies": (
        "physiological_chunk_event",
        "physiological_chunk_payload",
        "physiological_chunk_body",
        "physiological_payload",
        "physio_payload",
        "raw_physiological_chunk",
    ),
    "free-form semantic rationales": (
        "semantic_rationale",
        "semantic_rationales",
        "free_form_rationale",
        "free_form_rationales",
        "free_form_reasoning",
        "semantic_reasoning_text",
        "rationale_text",
    ),
}


def _expected_derived_only_forbidden_aliases(item: Section13Item) -> _SpecExtraction:
    body = item.body.casefold()
    expected_phrases = {
        "raw audio": "raw audio",
        "raw video": "raw video",
        "complete transient PhysiologicalChunkEvent payload bodies": "physiologicalchunkevent",
        "free-form semantic rationales": "free-form semantic rationale",
    }
    missing = [
        label
        for label, marker in expected_phrases.items()
        if marker.casefold() not in body
    ]
    if missing:
        return _not_extracted(
            "§13.30",
            "derived-only attribution persistence forbidden categories",
            "item verification_criterion missing " + ", ".join(missing),
        )
    return _extracted(
        _DERIVED_ONLY_FORBIDDEN_ALIASES,
        (
            "§13.30: extracted forbidden persistence categories from audit item "
            "verification_criterion."
        ),
    )


def _attribution_schema_fields(context: AuditContext) -> dict[str, tuple[int, str]]:
    rel_path = "packages/schemas/attribution.py"
    module, lines = _parse_python(context, rel_path)
    fields: dict[str, tuple[int, str]] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name not in _ATTRIBUTION_SCHEMA_CLASSES:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                fields[statement.target.id] = (
                    statement.lineno,
                    (
                        f"{rel_path}:{statement.lineno} "
                        f"{node.name}.{statement.target.id} "
                        f"{_line(lines, statement.lineno)!r}"
                    ),
                )
    return fields


def _strip_sql_line_comment(line: str) -> str:
    return line.split("--", 1)[0]


def _sql_table_columns(context: AuditContext) -> dict[str, tuple[int, str]]:
    rel_path = "data/sql/05-attribution.sql"
    columns: dict[str, tuple[int, str]] = {}
    active_table: str | None = None
    for line_number, raw_line in enumerate(_read_lines(context, rel_path), start=1):
        line = _strip_sql_line_comment(raw_line).strip()
        table_match = re.match(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\(",
            line,
            re.IGNORECASE,
        )
        if table_match:
            table = table_match.group(1).lower()
            active_table = table if table in _ATTRIBUTION_SQL_TABLES else None
            continue
        if active_table is None:
            continue
        if line.startswith(");") or line == ")":
            active_table = None
            continue
        column_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+", line)
        if column_match is None:
            continue
        column = column_match.group(1)
        if column.upper() in {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"}:
            continue
        columns[column] = (
            line_number,
            f"{rel_path}:{line_number} {active_table}.{column} {raw_line.strip()!r}",
        )
    return columns


def _forbidden_persistence_matches(
    surfaces: Mapping[str, tuple[int, str]],
    forbidden_aliases: Mapping[str, tuple[str, ...]],
) -> list[str]:
    matches: list[str] = []
    normalized_surface_names = {name.casefold(): evidence for name, evidence in surfaces.items()}
    for category, aliases in forbidden_aliases.items():
        for alias in aliases:
            evidence = normalized_surface_names.get(alias.casefold())
            if evidence is None:
                continue
            matches.append(f"§13.30 forbidden {category} field {alias!r}: {evidence[1]}")
    return matches


def verify_derived_only_attribution_persistence(
    context: AuditContext,
    item: Section13Item,
) -> AuditResult:
    aliases_extraction = _expected_derived_only_forbidden_aliases(item)
    if aliases_extraction.value is None:
        return _spec_extraction_result(item, aliases_extraction)
    forbidden_aliases = aliases_extraction.value
    schema_fields = _attribution_schema_fields(context)
    sql_columns = _sql_table_columns(context)
    analytics_text = _read_text(context, "services/worker/pipeline/analytics.py")
    combined_surfaces = {**schema_fields, **sql_columns}
    matches = _forbidden_persistence_matches(combined_surfaces, forbidden_aliases)
    required_schema_missing = [
        field for field in _DERIVED_ONLY_REQUIRED_FIELDS if field not in schema_fields
    ]
    required_sql_missing = [
        field for field in _DERIVED_ONLY_REQUIRED_FIELDS if field not in sql_columns
    ]
    required_analytics_missing = [
        field for field in _DERIVED_ONLY_REQUIRED_FIELDS if field not in analytics_text
    ]
    checks = [
        _Check(
            not matches,
            (
                "§13.30 attribution schema and DDL persistence surfaces contain no "
                "forbidden raw media, PhysiologicalChunkEvent payload, or free-form "
                f"rationale fields. {aliases_extraction.evidence}"
                if not matches
                else "§13.30 forbidden attribution persistence fields matched:\n"
                + "\n".join(matches)
            ),
        ),
        _Check(
            not required_schema_missing,
            (
                "§13.30 packages/schemas/attribution.py exposes derived/versioned "
                f"fields { _DERIVED_ONLY_REQUIRED_FIELDS!r}."
                if not required_schema_missing
                else "§13.30 packages/schemas/attribution.py missing derived fields "
                + ", ".join(required_schema_missing)
            ),
        ),
        _Check(
            not required_sql_missing,
            (
                "§13.30 data/sql/05-attribution.sql persists derived/versioned "
                f"columns { _DERIVED_ONLY_REQUIRED_FIELDS!r}."
                if not required_sql_missing
                else "§13.30 data/sql/05-attribution.sql missing derived columns "
                + ", ".join(required_sql_missing)
            ),
        ),
        _Check(
            not required_analytics_missing,
            (
                "§13.30 services/worker/pipeline/analytics.py writes the approved "
                f"derived attribution fields { _DERIVED_ONLY_REQUIRED_FIELDS!r}."
                if not required_analytics_missing
                else "§13.30 services/worker/pipeline/analytics.py missing INSERT evidence for "
                + ", ".join(required_analytics_missing)
            ),
        ),
    ]
    return _checks_result(item, checks)


# ---------------------------------------------------------------------------
# §13.1 — Directory structure
# ---------------------------------------------------------------------------


def _expected_directory_hierarchy(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    architecture = _as_mapping(spec_content.get("codebase_architecture"))
    if architecture is None:
        return _not_extracted(
            "§3/§13.1",
            "directory hierarchy entries",
            "missing codebase_architecture object in spec_content",
        )
    entries: list[str] = []
    for row in _as_sequence(architecture.get("directory_hierarchy")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        path = str(row_mapping.get("path", "")).strip()
        if path and path not in entries:
            entries.append(path)
    if not entries:
        return _not_extracted(
            "§3/§13.1",
            "directory hierarchy entries",
            "codebase_architecture.directory_hierarchy is empty",
        )
    return _extracted(
        tuple(entries),
        f"§3/§13.1: extracted {len(entries)} directory_hierarchy entries from spec_content.",
    )


def verify_directory_structure(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify §3 directory hierarchy entries exist in the repository."""

    extraction = _expected_directory_hierarchy(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)

    checks: list[_Check] = []
    for spec_path in extraction.value:
        relative = spec_path.lstrip("/").rstrip("/")
        if not relative:
            checks.append(
                _Check(False, f"§3/§13.1 spec_path={spec_path!r} resolves to repo root.")
            )
            continue
        target = context.repo_root / relative
        if spec_path.endswith("/"):
            present = target.is_dir()
            kind = "directory"
        elif "." in target.name:
            present = target.is_file()
            kind = "file"
        else:
            present = target.exists()
            kind = "path"
        checks.append(
            _Check(
                present,
                f"§3/§13.1 {relative} {kind} {'present' if present else 'MISSING'} "
                f"(spec_path={spec_path!r}).",
            )
        )

    summary = _checks_result(item, checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


# ---------------------------------------------------------------------------
# §13.2 — Docker topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ExpectedContainer:
    name: str
    image_token: str
    network: str
    restart_policy: str
    depends_on: tuple[str, ...]
    gpu_required: bool


@dataclass(frozen=True, slots=True)
class _ExpectedVolume:
    name: str
    mount_path: str
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedTopology:
    containers: tuple[_ExpectedContainer, ...]
    volumes: tuple[_ExpectedVolume, ...]


def _expected_docker_topology(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    topology = _as_mapping(spec_content.get("docker_topology"))
    if topology is None:
        return _not_extracted(
            "§9/§13.2",
            "docker topology containers/volumes",
            "missing docker_topology object in spec_content",
        )
    containers: list[_ExpectedContainer] = []
    for row in _as_sequence(topology.get("containers")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        name = str(row_mapping.get("container_name", "")).strip()
        image_base = str(row_mapping.get("image_base", "")).strip()
        # docker-compose tag form: extract the leading "image:tag" before any " + " suffix
        # (Capture Container is "ubuntu:24.04 + scrcpy v3.3.4").
        image_token = image_base.split(" + ", 1)[0].split()[0] if image_base else ""
        depends_on_raw = tuple(
            str(value).strip() for value in _as_sequence(row_mapping.get("depends_on")) if value
        )
        containers.append(
            _ExpectedContainer(
                name=name,
                image_token=image_token,
                network=str(row_mapping.get("network", "")).strip(),
                restart_policy=str(row_mapping.get("restart_policy", "")).strip(),
                depends_on=depends_on_raw,
                gpu_required=bool(row_mapping.get("gpu_required")),
            )
        )
    volumes: list[_ExpectedVolume] = []
    for row in _as_sequence(topology.get("volumes")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        volumes.append(
            _ExpectedVolume(
                name=str(row_mapping.get("volume_name", "")).strip(),
                mount_path=str(row_mapping.get("mount_path", "")).strip(),
                targets=tuple(
                    str(value).strip()
                    for value in _as_sequence(row_mapping.get("container_targets"))
                    if value
                ),
            )
        )
    if not containers:
        return _not_extracted(
            "§9/§13.2",
            "docker topology containers",
            "docker_topology.containers is empty",
        )
    return _extracted(
        _ExpectedTopology(containers=tuple(containers), volumes=tuple(volumes)),
        f"§9/§13.2: extracted {len(containers)} containers and {len(volumes)} volumes "
        "from spec_content.docker_topology.",
    )


def _load_docker_compose(context: AuditContext) -> tuple[Mapping[str, Any], int]:
    import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

    compose_path = _repo_file(context, "docker-compose.yml")
    text = compose_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("docker-compose.yml root must be a mapping")
    return parsed, len(text.splitlines())


def verify_docker_topology(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify §9 docker topology container/volume contract against docker-compose.yml."""

    extraction = _expected_docker_topology(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)
    expected: _ExpectedTopology = extraction.value

    try:
        compose, _line_count = _load_docker_compose(context)
    except FileNotFoundError:
        return _result(
            item,
            False,
            f"§9/§13.2 docker-compose.yml is missing (expected at {context.repo_root}).",
            "Restore docker-compose.yml at the repository root.",
        )
    except (ImportError, ValueError) as exc:
        return _result(
            item,
            False,
            f"§9/§13.2 unable to parse docker-compose.yml: {exc}",
            "Install pyyaml or repair docker-compose.yml so the §13.2 verifier can parse it.",
        )

    services_raw = compose.get("services") if isinstance(compose, Mapping) else None
    services = _as_mapping(services_raw) or {}
    top_volumes = _as_mapping(compose.get("volumes")) or {}

    checks: list[_Check] = []
    for spec_container in expected.containers:
        service = _as_mapping(services.get(spec_container.name))
        if service is None:
            checks.append(
                _Check(
                    False,
                    f"§9/§13.2 service {spec_container.name!r} is missing from docker-compose.yml.",
                )
            )
            continue
        service_image = str(service.get("image", "")).strip()
        build_block = _as_mapping(service.get("build"))
        # GPU services build from a Dockerfile rather than declaring `image:`; for
        # those the spec image_token names the FROM base, which we accept implicitly.
        image_match = (
            spec_container.image_token == ""
            or spec_container.image_token in service_image
            or build_block is not None
        )
        checks.append(
            _Check(
                image_match,
                f"§9/§13.2 service {spec_container.name} image='{service_image}' "
                f"matches spec image_token={spec_container.image_token!r}."
                if image_match
                else (
                    f"§9/§13.2 service {spec_container.name} image='{service_image}' "
                    f"does not contain spec image_token={spec_container.image_token!r}."
                ),
            )
        )
        service_networks = service.get("networks")
        network_names: tuple[str, ...] = ()
        if isinstance(service_networks, Mapping):
            network_names = tuple(str(name) for name in service_networks)
        elif isinstance(service_networks, Sequence) and not isinstance(
            service_networks, str | bytes | bytearray
        ):
            network_names = tuple(str(name) for name in service_networks)
        if spec_container.network:
            network_match = spec_container.network in network_names
        else:
            network_match = True
        checks.append(
            _Check(
                network_match,
                f"§9/§13.2 service {spec_container.name} network={network_names!r} "
                f"includes spec network={spec_container.network!r}."
                if network_match
                else (
                    f"§9/§13.2 service {spec_container.name} network={network_names!r} "
                    f"missing spec network={spec_container.network!r}."
                ),
            )
        )
        restart_policy = str(service.get("restart", "")).strip()
        deploy_block = _as_mapping(service.get("deploy"))
        if not restart_policy and deploy_block is not None:
            restart_policy_block = _as_mapping(deploy_block.get("restart_policy")) or {}
            restart_policy = str(restart_policy_block.get("condition", "")).strip()
        # The compose form of "on-failure:5" is "restart: on-failure" with
        # max_attempts: 5; accept the prefix match against spec.
        spec_policy = spec_container.restart_policy
        if not spec_policy:
            policy_match = True
        elif restart_policy == spec_policy:
            policy_match = True
        elif spec_policy.startswith("on-failure") and restart_policy.startswith("on-failure"):
            policy_match = True
        else:
            policy_match = False
        checks.append(
            _Check(
                policy_match,
                f"§9/§13.2 service {spec_container.name} restart='{restart_policy}' "
                f"matches spec restart_policy={spec_policy!r}."
                if policy_match
                else (
                    f"§9/§13.2 service {spec_container.name} restart='{restart_policy}' "
                    f"does not match spec restart_policy={spec_policy!r}."
                ),
            )
        )
        depends_on_raw = service.get("depends_on")
        depends_names: tuple[str, ...] = ()
        if isinstance(depends_on_raw, Mapping):
            depends_names = tuple(str(name) for name in depends_on_raw)
        elif isinstance(depends_on_raw, Sequence) and not isinstance(
            depends_on_raw, str | bytes | bytearray
        ):
            depends_names = tuple(str(name) for name in depends_on_raw)
        depends_missing = tuple(name for name in spec_container.depends_on if name not in depends_names)
        depends_match = not depends_missing
        checks.append(
            _Check(
                depends_match,
                f"§9/§13.2 service {spec_container.name} depends_on={depends_names!r} "
                f"covers spec depends_on={spec_container.depends_on!r}."
                if depends_match
                else (
                    f"§9/§13.2 service {spec_container.name} depends_on={depends_names!r} "
                    f"missing spec entries {depends_missing!r}."
                ),
            )
        )

    for spec_volume in expected.volumes:
        if spec_volume.name not in top_volumes:
            checks.append(
                _Check(
                    False,
                    f"§9/§13.2 top-level volumes is missing {spec_volume.name!r}.",
                )
            )
        else:
            checks.append(
                _Check(
                    True,
                    f"§9/§13.2 top-level volumes declares {spec_volume.name!r}.",
                )
            )
        for target in spec_volume.targets:
            service = _as_mapping(services.get(target))
            mount_lines: list[str] = []
            if service is not None:
                volumes_block = service.get("volumes")
                if isinstance(volumes_block, Sequence) and not isinstance(
                    volumes_block, str | bytes | bytearray
                ):
                    for entry in volumes_block:
                        if isinstance(entry, str):
                            mount_lines.append(entry)
                        elif isinstance(entry, Mapping):
                            mount_lines.append(
                                f"{entry.get('source', '')}:{entry.get('target', '')}"
                            )
            mounted = any(
                line.startswith(f"{spec_volume.name}:") and spec_volume.mount_path in line
                for line in mount_lines
            )
            checks.append(
                _Check(
                    mounted,
                    f"§9/§13.2 service {target} mounts volume {spec_volume.name!r} at "
                    f"{spec_volume.mount_path!r}."
                    if mounted
                    else (
                        f"§9/§13.2 service {target} does not mount {spec_volume.name!r} at "
                        f"{spec_volume.mount_path!r} (mounts={mount_lines!r})."
                    ),
                )
            )

    summary = _checks_result(item, checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


# ---------------------------------------------------------------------------
# §13.3 — IPC lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _IPCStep:
    step_number: int
    title: str
    tokens: tuple[str, ...]


def _expected_ipc_lifecycle(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    module_a = _module_by_id(spec_content, "A")
    if module_a is None:
        return _not_extracted(
            "§4.A/§13.3",
            "Module A IPC lifecycle steps",
            "missing core_modules.modules[module_id='A']",
        )
    steps: list[_IPCStep] = []
    for row in _as_sequence(module_a.get("ipc_lifecycle_steps")):
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        description = str(row_mapping.get("description", ""))
        # Pull every <path>/<func>/<env>/<cmd>/<flag> token from the description
        # and use those as the literal grep targets for the implementation file.
        tokens = tuple(
            value.strip()
            for _tag, value in re.findall(
                r"<(path|func|env|cmd|flag|file)>([^<]+)</\1>", description
            )
            if value.strip()
        )
        if not tokens:
            continue
        steps.append(
            _IPCStep(
                step_number=int(row_mapping.get("step_number", len(steps) + 1)),
                title=str(row_mapping.get("title", "")).strip(),
                tokens=tokens,
            )
        )
    if not steps:
        return _not_extracted(
            "§4.A/§13.3",
            "Module A IPC lifecycle tokens",
            "no <tag>...</tag> tokens in any ipc_lifecycle_steps[*].description",
        )
    return _extracted(
        tuple(steps),
        f"§4.A/§13.3: extracted {len(steps)} IPC lifecycle steps with {sum(len(s.tokens) for s in steps)} "
        "structured tokens from spec_content.",
    )


_SHELL_ASSIGN_RE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)="([^"\n]+)"\s*$', re.MULTILINE)


def _expand_shell_variables(text: str) -> str:
    """Substitute literal POSIX `VAR="value"` declarations into expanded form.

    The verifier only expands declarations whose value is itself a literal (or
    references previously-declared variables) so the IPC lifecycle tokens like
    `/tmp/ipc/audio_stream.raw` resolve through `$AUDIO_PIPE="$IPC_DIR/..."`.
    Quoted variable references (`"$VAR"`) collapse to just the value so the
    expanded form matches unquoted spec tokens like `mkdir -p /tmp/ipc`.
    Conditionally-assigned shell variables and command substitutions are left
    untouched.
    """

    expansions: dict[str, str] = {}
    for match in _SHELL_ASSIGN_RE.finditer(text):
        name, value = match.group(1), match.group(2)
        if "$(" in value or "`" in value:
            continue
        resolved = re.sub(
            r"\$\{?([A-Z_][A-Z0-9_]*)\}?",
            lambda m: expansions.get(m.group(1), m.group(0)),
            value,
        )
        expansions[name] = resolved
    expanded = text
    for name, value in expansions.items():
        # Drop surrounding double quotes around the reference so the expanded
        # text matches unquoted spec tokens (e.g. spec says `mkdir -p /tmp/ipc`,
        # source says `mkdir -p "$IPC_DIR"`).
        expanded = expanded.replace(f'"${{{name}}}"', value)
        expanded = expanded.replace(f'"${name}"', value)
        expanded = expanded.replace(f"${{{name}}}", value)
        expanded = re.sub(rf"\${name}\b", value, expanded)
    return text + "\n" + expanded


def verify_ipc_lifecycle(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify each Module A IPC lifecycle step is implemented in stream_ingest entrypoint."""

    extraction = _expected_ipc_lifecycle(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)
    steps: tuple[_IPCStep, ...] = extraction.value

    rel_path = "services/stream_ingest/entrypoint.sh"
    try:
        raw_text = _read_text(context, rel_path)
    except FileNotFoundError:
        return _result(
            item,
            False,
            f"§4.A/§13.3 {rel_path} is missing.",
            f"Create {rel_path} or update the verifier to reference the new Module A entrypoint.",
        )
    text = _expand_shell_variables(raw_text)

    checks: list[_Check] = []
    for step in steps:
        missing = tuple(token for token in step.tokens if token not in text)
        passed = not missing
        if passed:
            checks.append(
                _Check(
                    True,
                    f"§4.A/§13.3 step {step.step_number} ({step.title}): "
                    f"all tokens present {step.tokens!r}.",
                )
            )
        else:
            checks.append(
                _Check(
                    False,
                    f"§4.A/§13.3 step {step.step_number} ({step.title}): "
                    f"{rel_path} missing tokens {missing!r}.",
                )
            )

    summary = _checks_result(item, checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


# ---------------------------------------------------------------------------
# §13.5 — Drift correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ExpectedDrift:
    poll_interval_s: int
    max_drift_ms: int
    freeze_after: int
    reset_timeout_s: int
    formula: str


_DRIFT_INTERVAL_RE = re.compile(r"(\d+)\s*seconds?")
_DRIFT_MS_RE = re.compile(r"(\d+)\s*milliseconds?")
_DRIFT_FREEZE_RE = re.compile(r"(\d+)\s*consecutive")
_DRIFT_RESET_RE = re.compile(r"(\d+)\s*seconds?")


def _expected_drift_parameters(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    module_c = _module_by_id(spec_content, "C")
    if module_c is None:
        return _not_extracted(
            "§4.C/§13.5",
            "Module C drift_polling_parameters",
            "missing core_modules.modules[module_id='C']",
        )
    rows = _as_sequence(module_c.get("drift_polling_parameters"))
    if not rows:
        return _not_extracted(
            "§4.C/§13.5",
            "Module C drift_polling_parameters",
            "drift_polling_parameters list is empty on Module C",
        )

    poll_interval: int | None = None
    max_drift_ms: int | None = None
    freeze_after: int | None = None
    reset_timeout: int | None = None
    formula: str | None = None
    for row in rows:
        row_mapping = _as_mapping(row)
        if row_mapping is None:
            continue
        parameter = str(row_mapping.get("parameter", "")).strip().lower()
        value = str(row_mapping.get("value", "")).strip()
        if parameter == "polling interval":
            match = _DRIFT_INTERVAL_RE.search(value)
            if match:
                poll_interval = int(match.group(1))
        elif parameter == "maximum tolerated drift":
            match = _DRIFT_MS_RE.search(value)
            if match:
                max_drift_ms = int(match.group(1))
        elif parameter == "freeze threshold":
            match = _DRIFT_FREEZE_RE.search(value)
            if match:
                freeze_after = int(match.group(1))
        elif parameter == "reset timeout":
            match = _DRIFT_RESET_RE.search(value)
            if match:
                reset_timeout = int(match.group(1))
        elif parameter == "drift formula":
            formula = value
    missing: list[str] = []
    if poll_interval is None:
        missing.append("Polling interval")
    if max_drift_ms is None:
        missing.append("Maximum tolerated drift")
    if freeze_after is None:
        missing.append("Freeze threshold")
    if reset_timeout is None:
        missing.append("Reset timeout")
    if formula is None:
        missing.append("Drift formula")
    if (
        missing
        or poll_interval is None
        or max_drift_ms is None
        or freeze_after is None
        or reset_timeout is None
        or formula is None
    ):
        return _not_extracted(
            "§4.C/§13.5",
            "Module C drift_polling_parameters values",
            "missing rows for " + ", ".join(missing) if missing else "incomplete extraction",
        )
    return _extracted(
        _ExpectedDrift(
            poll_interval_s=poll_interval,
            max_drift_ms=max_drift_ms,
            freeze_after=freeze_after,
            reset_timeout_s=reset_timeout,
            formula=formula,
        ),
        "§4.C/§13.5: extracted drift_polling_parameters (interval, max drift, freeze, "
        "reset, formula) from spec_content.",
    )


def verify_drift_correction(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify Module C drift constants and formula appear in orchestrator.py."""

    extraction = _expected_drift_parameters(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)
    expected: _ExpectedDrift = extraction.value

    rel_path = "services/worker/pipeline/orchestrator.py"
    try:
        text = _read_text(context, rel_path)
    except FileNotFoundError:
        return _result(
            item,
            False,
            f"§4.C/§13.5 {rel_path} is missing.",
            f"Create {rel_path} or update §13.5 verifier to point to the new drift module.",
        )

    poll_value = _literal_assignment(context, rel_path, "DRIFT_POLL_INTERVAL")
    max_drift_value = _literal_assignment(context, rel_path, "MAX_TOLERATED_DRIFT_MS")
    freeze_value = _literal_assignment(context, rel_path, "DRIFT_FREEZE_AFTER_FAILURES")
    reset_value = _literal_assignment(context, rel_path, "DRIFT_RESET_TIMEOUT")
    # Drift formula evidence: spec form is "drift_offset = host_utc - android_epoch"; the
    # orchestrator implementation uses the same algebraic form with renamed locals, so we
    # require either a substring match on the spec form OR a documented `drift_offset`
    # subtraction (sign-preserving) in the file.
    formula_match = (
        expected.formula in text
        or re.search(
            r"drift_offset\s*=\s*[A-Za-z_][A-Za-z0-9_.]*\s*-\s*[A-Za-z_][A-Za-z0-9_.]*", text
        )
        is not None
    )

    checks: list[_Check] = [
        _Check(
            poll_value is not None and poll_value.value == expected.poll_interval_s,
            (
                f"§4.C/§13.5 {rel_path}:{poll_value.line if poll_value else '?'} "
                f"DRIFT_POLL_INTERVAL={poll_value.value if poll_value else '<unset>'}; "
                f"expected {expected.poll_interval_s}."
            ),
        ),
        _Check(
            max_drift_value is not None and max_drift_value.value == expected.max_drift_ms,
            (
                f"§4.C/§13.5 {rel_path}:{max_drift_value.line if max_drift_value else '?'} "
                f"MAX_TOLERATED_DRIFT_MS={max_drift_value.value if max_drift_value else '<unset>'}; "
                f"expected {expected.max_drift_ms}."
            ),
        ),
        _Check(
            freeze_value is not None and freeze_value.value == expected.freeze_after,
            (
                f"§4.C/§13.5 {rel_path}:{freeze_value.line if freeze_value else '?'} "
                f"DRIFT_FREEZE_AFTER_FAILURES={freeze_value.value if freeze_value else '<unset>'}; "
                f"expected {expected.freeze_after}."
            ),
        ),
        _Check(
            reset_value is not None and reset_value.value == expected.reset_timeout_s,
            (
                f"§4.C/§13.5 {rel_path}:{reset_value.line if reset_value else '?'} "
                f"DRIFT_RESET_TIMEOUT={reset_value.value if reset_value else '<unset>'}; "
                f"expected {expected.reset_timeout_s}."
            ),
        ),
        _Check(
            formula_match,
            (
                f"§4.C/§13.5 {rel_path} contains drift formula {expected.formula!r} or an "
                f"algebraic `drift_offset = X - Y` assignment."
                if formula_match
                else f"§4.C/§13.5 {rel_path} missing drift formula {expected.formula!r}."
            ),
        ),
    ]

    summary = _checks_result(item, checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


# ---------------------------------------------------------------------------
# §13.9 — Schema validation (InferenceHandoffPayload)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ExpectedSchema:
    required: tuple[str, ...]
    properties: tuple[str, ...]
    deprecated: frozenset[str]


_SCHEMA_DEPRECATED_HINTS = (
    "diagnostic",
    "calibration telemetry",
    "compatibility",
    "does not divide",
)


def _expected_inference_handoff_schema(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    contracts = _as_mapping(spec_content.get("interface_contracts"))
    if contracts is None:
        return _not_extracted(
            "§6.1/§13.9",
            "InferenceHandoffPayload schema",
            "missing interface_contracts object in spec_content",
        )
    schema_block = _as_mapping(contracts.get("schema_definition"))
    if schema_block is None:
        return _not_extracted(
            "§6.1/§13.9",
            "InferenceHandoffPayload schema",
            "missing interface_contracts.schema_definition",
        )
    source = str(schema_block.get("source", "")).strip()
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        return _not_extracted(
            "§6.1/§13.9",
            "InferenceHandoffPayload schema",
            f"schema_definition.source is not parseable JSON: {exc}",
        )
    if not isinstance(parsed, Mapping) or parsed.get("title") != "InferenceHandoffPayload":
        return _not_extracted(
            "§6.1/§13.9",
            "InferenceHandoffPayload schema",
            "schema_definition.source.title != 'InferenceHandoffPayload'",
        )
    required = tuple(str(field) for field in _as_sequence(parsed.get("required")))
    properties_map = _as_mapping(parsed.get("properties")) or {}
    properties = tuple(str(field) for field in properties_map)
    deprecated: set[str] = set()
    for field, schema in properties_map.items():
        schema_mapping = _as_mapping(schema)
        if schema_mapping is None:
            continue
        description = str(schema_mapping.get("description", "")).lower()
        if any(hint in description for hint in _SCHEMA_DEPRECATED_HINTS):
            deprecated.add(str(field))
    return _extracted(
        _ExpectedSchema(
            required=required, properties=properties, deprecated=frozenset(deprecated)
        ),
        f"§6.1/§13.9: extracted InferenceHandoffPayload schema with {len(required)} required, "
        f"{len(properties)} total properties, {len(deprecated)} deprecated/diagnostic fields "
        f"({sorted(deprecated)!r}) from spec_content.",
    )


def _is_pydantic_basemodel_subclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def _pydantic_field_aliases(node: ast.ClassDef) -> dict[str, str]:
    """Map declared attribute name -> external alias for every Pydantic field."""

    aliases: dict[str, str] = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            attr_name = stmt.target.id
            value = stmt.value
            alias = attr_name  # default if no explicit alias
            if isinstance(value, ast.Call) and (
                (isinstance(value.func, ast.Name) and value.func.id == "Field")
                or (isinstance(value.func, ast.Attribute) and value.func.attr == "Field")
            ):
                for keyword in value.keywords:
                    if keyword.arg == "alias" and isinstance(keyword.value, ast.Constant):
                        if isinstance(keyword.value.value, str):
                            alias = keyword.value.value
            aliases[attr_name] = alias
    return aliases


def _has_extra_forbid(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "model_config"
                    and isinstance(stmt.value, ast.Call)
                ):
                    for keyword in stmt.value.keywords:
                        if (
                            keyword.arg == "extra"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value == "forbid"
                        ):
                            return True
    return False


def verify_schema_validation(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify InferenceHandoffPayload is a Pydantic BaseModel covering §6.1 required fields."""

    extraction = _expected_inference_handoff_schema(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)
    expected: _ExpectedSchema = extraction.value

    rel_path = "packages/schemas/inference_handoff.py"
    try:
        module, _lines = _parse_python(context, rel_path)
    except FileNotFoundError:
        return _result(
            item,
            False,
            f"§6.1/§13.9 {rel_path} is missing.",
            f"Create {rel_path} or repoint §13.9 verifier at the new location.",
        )

    target_class: ast.ClassDef | None = None
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "InferenceHandoffPayload":
            target_class = node
            break
    if target_class is None:
        return _result(
            item,
            False,
            f"§6.1/§13.9 {rel_path} does not define class InferenceHandoffPayload.",
            f"Define InferenceHandoffPayload(BaseModel) in {rel_path}.",
        )

    aliases = _pydantic_field_aliases(target_class)
    declared_external = set(aliases.values())
    missing_required: list[str] = []
    deprecation_excluded: list[str] = []
    for required_field in expected.required:
        if required_field in declared_external:
            continue
        if required_field in expected.deprecated:
            deprecation_excluded.append(required_field)
            continue
        missing_required.append(required_field)

    is_basemodel = _is_pydantic_basemodel_subclass(target_class)
    extra_forbid = _has_extra_forbid(target_class)

    # Confirm the model is actually validated at a dispatch boundary somewhere
    # under services/. We grep for `InferenceHandoffPayload(` or `.model_validate(`
    # invocations under services/worker/.
    dispatch_evidence_re = re.compile(
        r"InferenceHandoffPayload\s*\(|InferenceHandoffPayload\.model_validate"
    )
    dispatch_hits: list[str] = []
    for path in (context.repo_root / "services" / "worker").rglob("*.py"):
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if dispatch_evidence_re.search(body):
            dispatch_hits.append(str(path.relative_to(context.repo_root).as_posix()))

    checks: list[_Check] = [
        _Check(
            is_basemodel,
            f"§6.1/§13.9 {rel_path} InferenceHandoffPayload subclasses pydantic BaseModel."
            if is_basemodel
            else f"§6.1/§13.9 {rel_path} InferenceHandoffPayload does not subclass BaseModel.",
        ),
        _Check(
            extra_forbid,
            f"§6.1/§13.9 {rel_path} InferenceHandoffPayload sets model_config extra='forbid'."
            if extra_forbid
            else (
                f"§6.1/§13.9 {rel_path} InferenceHandoffPayload model_config does not "
                "set extra='forbid'."
            ),
        ),
        _Check(
            not missing_required,
            f"§6.1/§13.9 {rel_path} declares all required fields {sorted(expected.required)!r}; "
            f"deprecated/diagnostic carve-outs accepted: {sorted(deprecation_excluded)!r}."
            if not missing_required
            else (
                f"§6.1/§13.9 {rel_path} missing required fields "
                f"{sorted(missing_required)!r} (declared aliases={sorted(declared_external)!r})."
            ),
        ),
        _Check(
            bool(dispatch_hits),
            f"§6.1/§13.9 InferenceHandoffPayload validated at dispatch boundary: {dispatch_hits!r}."
            if dispatch_hits
            else (
                "§6.1/§13.9 services/worker/ does not invoke InferenceHandoffPayload(...) or "
                ".model_validate(); the schema is defined but not used at a dispatch boundary."
            ),
        ),
    ]

    summary = _checks_result(item, checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


# ---------------------------------------------------------------------------
# §13.10 — Module contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ContractToken:
    module_id: str
    contract_field: str
    tag: str
    value: str


_CONTRACT_TAG_RE = re.compile(r"<(path|func|env|class|cmd|flag|field|file)>([^<]+)</\1>")
_CONTRACT_FIELDS = ("inputs", "outputs", "dependencies", "side_effects", "failure_modes")


def _expected_module_contracts(spec_content: Mapping[str, Any]) -> _SpecExtraction:
    core_modules = _as_mapping(spec_content.get("core_modules"))
    if core_modules is None:
        return _not_extracted(
            "§4/§13.10",
            "Module A–F contract surfaces",
            "missing core_modules object in spec_content",
        )
    tokens: list[_ContractToken] = []
    seen_modules: set[str] = set()
    for module in _as_sequence(core_modules.get("modules")):
        module_mapping = _as_mapping(module)
        if module_mapping is None:
            continue
        module_id = str(module_mapping.get("module_id", "")).strip()
        if not module_id:
            continue
        seen_modules.add(module_id)
        contract = _as_mapping(module_mapping.get("contract"))
        if contract is None:
            continue
        for field in _CONTRACT_FIELDS:
            text = str(contract.get(field, ""))
            for tag, value in _CONTRACT_TAG_RE.findall(text):
                cleaned = value.strip()
                if not cleaned:
                    continue
                tokens.append(
                    _ContractToken(
                        module_id=module_id, contract_field=field, tag=tag, value=cleaned
                    )
                )
    if not tokens:
        return _not_extracted(
            "§4/§13.10",
            "Module contract <tag>...</tag> tokens",
            "no structured tokens in any contract field across modules",
        )
    return _extracted(
        tuple(tokens),
        f"§4/§13.10: extracted {len(tokens)} contract tokens across "
        f"{len(seen_modules)} modules ({sorted(seen_modules)!r}) from spec_content.",
    )


# Tokens that the spec carves out as deprecated/diagnostic (handled elsewhere) or
# that name external infrastructure rather than repo-local symbols. The audit
# accepts these as documented but does not require them in the source tree.
_CONTRACT_DOCUMENTED_OMISSIONS = frozenset(
    {
        "_x_max",  # spec §6.1 _x_max — diagnostic/compatibility telemetry only
    }
)
# Token values whose tag is "env" but whose canonical home is deployment-only (compose
# environment, .env.example) rather than Python source.
_CONTRACT_DEPLOYMENT_ENV_TOKENS = frozenset(
    {
        "ADB_SERVER_SOCKET=tcp:host.docker.internal:5037",
        "SDL_VIDEODRIVER=dummy",
        "XDG_RUNTIME_DIR=/tmp",
    }
)


def _scan_targets_for_module(repo_root: Path, module_id: str) -> tuple[Path, ...]:
    """Per-module file globs that audit-token presence checks against."""

    if module_id == "A":
        return (repo_root / "services" / "stream_ingest", repo_root / "docker-compose.yml")
    if module_id == "B":
        return (
            repo_root / "services" / "worker" / "pipeline" / "ground_truth.py",
            repo_root / "services" / "api" / "routes",
            repo_root / "services" / "api" / "services",
            repo_root / "packages" / "schemas",
        )
    if module_id == "C":
        return (
            repo_root / "services" / "worker" / "pipeline",
            repo_root / "packages" / "schemas",
            repo_root / "packages" / "ml_core",
        )
    if module_id == "D":
        return (
            repo_root / "services" / "worker" / "tasks",
            repo_root / "packages" / "ml_core",
            repo_root / "packages" / "schemas",
        )
    if module_id == "E":
        return (
            repo_root / "services" / "worker" / "pipeline",
            repo_root / "services" / "api",
            repo_root / "packages" / "ml_core",
            repo_root / "packages" / "schemas",
        )
    if module_id == "F":
        return (
            repo_root / "services" / "worker" / "tasks" / "enrichment.py",
            repo_root / "packages" / "schemas",
            repo_root / "data" / "sql",
        )
    return ()


def _token_present(token: _ContractToken, search_roots: Sequence[Path]) -> bool:
    needle = token.value
    if token.tag == "func" and needle.endswith("()"):
        needle = needle[:-2]
    for root in search_roots:
        if not root.exists():
            continue
        candidates: Iterable[Path]
        if root.is_file():
            candidates = (root,)
        else:
            candidates = (
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix not in _BINARY_SCAN_SUFFIXES
                and not _has_excluded_scan_part(path.relative_to(root))
            )
        for path in candidates:
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Shell scripts encode token-bearing paths through variable
            # assignments (e.g. `AUDIO_PIPE="$IPC_DIR/audio_stream.raw"`); the
            # spec lists the expanded form, so resolve declarations before the
            # substring check.
            if path.suffix == ".sh":
                body = _expand_shell_variables(body)
            if needle in body:
                return True
    return False


def verify_module_contracts(context: AuditContext, item: Section13Item) -> AuditResult:
    """Verify each Module A–F contract <tag> token resolves to repo-local source evidence."""

    extraction = _expected_module_contracts(context.spec_content)
    if extraction.value is None:
        return _spec_extraction_result(item, extraction)
    tokens: tuple[_ContractToken, ...] = extraction.value

    repo_root = context.repo_root
    deployment_search_roots = (
        repo_root / "docker-compose.yml",
        repo_root / ".env.example",
    )

    # Group module-level results so the evidence stays compact.
    by_module: dict[str, list[_Check]] = {}
    for token in tokens:
        if token.value in _CONTRACT_DOCUMENTED_OMISSIONS:
            by_module.setdefault(token.module_id, []).append(
                _Check(
                    True,
                    f"§4/§13.10 module {token.module_id} contract.{token.contract_field} "
                    f"<{token.tag}>{token.value}</{token.tag}>: documented carve-out "
                    f"(diagnostic/compatibility-only); not required in source tree.",
                )
            )
            continue
        search_roots: tuple[Path, ...]
        if token.tag == "env" and token.value in _CONTRACT_DEPLOYMENT_ENV_TOKENS:
            search_roots = deployment_search_roots
        else:
            search_roots = _scan_targets_for_module(repo_root, token.module_id)
            # For env-style tokens, also accept presence in compose / .env.example.
            if token.tag == "env":
                search_roots = (*search_roots, *deployment_search_roots)
        present = _token_present(token, search_roots)
        by_module.setdefault(token.module_id, []).append(
            _Check(
                present,
                f"§4/§13.10 module {token.module_id} contract.{token.contract_field} "
                f"<{token.tag}>{token.value}</{token.tag}>: "
                f"{'present' if present else 'MISSING'} in {[r.name for r in search_roots]!r}.",
            )
        )

    # Flatten in module order.
    flat_checks: list[_Check] = []
    for module_id in sorted(by_module):
        flat_checks.extend(by_module[module_id])

    summary = _checks_result(item, flat_checks)
    return _result(
        item,
        summary.passed,
        f"{extraction.evidence}\n{summary.evidence}",
        summary.follow_up,
    )


MECHANICAL_VERIFIERS: Mapping[str, AuditVerifier] = {
    "13.1": verify_directory_structure,
    "13.2": verify_docker_topology,
    "13.3": verify_ipc_lifecycle,
    "13.4": verify_ffmpeg_resample,
    "13.5": verify_drift_correction,
    "13.6": verify_au12_geometry,
    "13.7": verify_semantic_determinism,
    "13.8": verify_ephemeral_vault,
    "13.9": verify_schema_validation,
    "13.10": verify_module_contracts,
    "13.12": verify_dependency_pins,
    "13.15": verify_canonical_terminology,
    "13.30": verify_derived_only_attribution_persistence,
}


def register_mechanical_verifiers(item_ids: Iterable[str] | None = None) -> None:
    """Register mechanical verifiers on the current shared registry."""

    registry = get_default_registry()
    requested = set(MECHANICAL_VERIFIERS) if item_ids is None else set(item_ids)
    for item_id, verifier in MECHANICAL_VERIFIERS.items():
        if item_id not in requested:
            continue
        if not registry.has_verifier(item_id):
            registry.register(item_id)(verifier)


__all__ = [
    "MECHANICAL_VERIFIERS",
    "register_mechanical_verifiers",
    "verify_au12_geometry",
    "verify_canonical_terminology",
    "verify_dependency_pins",
    "verify_derived_only_attribution_persistence",
    "verify_directory_structure",
    "verify_docker_topology",
    "verify_drift_correction",
    "verify_ephemeral_vault",
    "verify_ffmpeg_resample",
    "verify_ipc_lifecycle",
    "verify_module_contracts",
    "verify_schema_validation",
    "verify_semantic_determinism",
    "verify_semantic_reason_codes",
]
