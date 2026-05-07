"""Static verifier for Operator Console design-system invariants."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

HEX_LITERAL_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
OBJECT_NAME_RE = re.compile(r"#([A-Za-z_][A-Za-z0-9_]*)")
REQUIRED_MANIFEST_KEYS = (
    "$schema",
    "version",
    "source_audit",
    "spec_refs",
    "tokens_file",
    "shells",
    "primitives",
    "compounds",
    "selectors",
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _operator_console_root(repo_root: Path) -> Path:
    return repo_root / "services" / "operator_console"


def _design_system_root(repo_root: Path) -> Path:
    return _operator_console_root(repo_root) / "design_system"


def _manifest_path(repo_root: Path) -> Path:
    return _design_system_root(repo_root) / "design_system.json"


def _schema_path(repo_root: Path) -> Path:
    return _design_system_root(repo_root) / "design_system.schema.json"


def _tokens_path(repo_root: Path) -> Path:
    return _design_system_root(repo_root) / "tokens.json"


def _qss_builder_path(repo_root: Path) -> Path:
    return _design_system_root(repo_root) / "qss_builder.py"


def _relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def _resolve_targets(repo_root: Path, raw_paths: Sequence[str] | None) -> tuple[Path, ...]:
    if not raw_paths:
        return (_operator_console_root(repo_root).resolve(),)
    resolved: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {candidate}")
        resolved.append(candidate)
    return tuple(resolved)


def _iter_python_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    discovered: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            discovered.add(path.resolve())
            continue
        if path.is_dir():
            for child in path.rglob("*.py"):
                discovered.add(child.resolve())
    return tuple(sorted(discovered))


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _registered_object_names(manifest: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    for key in ("primitives", "compounds", "selectors"):
        entries = manifest.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            object_name = entry.get("object_name")
            if isinstance(object_name, str):
                names.add(object_name)
            object_names = entry.get("object_names")
            if isinstance(object_names, list):
                names.update(name for name in object_names if isinstance(name, str))
    return names


def _stylesheet_object_names(qss_builder_source: str) -> set[str]:
    return {match.group(1) for match in OBJECT_NAME_RE.finditer(qss_builder_source)}


def _collect_manifest_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    manifest_path = _manifest_path(repo_root)
    schema_path = _schema_path(repo_root)
    qss_builder_path = _qss_builder_path(repo_root)

    if not schema_path.exists():
        issues.append(f"{_relative(repo_root, schema_path)} is missing")
    if not manifest_path.exists():
        issues.append(f"{_relative(repo_root, manifest_path)} is missing")
        return issues
    if not qss_builder_path.exists():
        issues.append(f"{_relative(repo_root, qss_builder_path)} is missing")
        return issues

    manifest = _read_json(manifest_path)
    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_keys:
        issues.append(
            f"{_relative(repo_root, manifest_path)} is missing keys: {', '.join(missing_keys)}"
        )

    source_audit = manifest.get("source_audit")
    if isinstance(source_audit, str):
        audit_path = (manifest_path.parent / source_audit).resolve()
        if not audit_path.exists():
            issues.append(
                f"{_relative(repo_root, manifest_path)} points to missing "
                f"source audit: {source_audit}"
            )
    else:
        issues.append(f"{_relative(repo_root, manifest_path)} must define string source_audit")

    tokens_file = manifest.get("tokens_file")
    if isinstance(tokens_file, str):
        tokens_path = (manifest_path.parent / tokens_file).resolve()
        if not tokens_path.exists():
            issues.append(
                f"{_relative(repo_root, manifest_path)} points to missing "
                f"tokens file: {tokens_file}"
            )
    else:
        issues.append(f"{_relative(repo_root, manifest_path)} must define string tokens_file")

    for key in ("shells", "primitives", "compounds", "selectors"):
        value = manifest.get(key)
        if not isinstance(value, list):
            issues.append(f"{_relative(repo_root, manifest_path)} key {key} must be a list")

    selectors = manifest.get("selectors")
    if isinstance(selectors, list):
        for index, entry in enumerate(selectors):
            if not isinstance(entry, Mapping):
                issues.append(
                    f"{_relative(repo_root, manifest_path)} selectors[{index}] must be an object"
                )
                continue
            if not isinstance(entry.get("object_name"), str):
                issues.append(
                    f"{_relative(repo_root, manifest_path)} selectors[{index}] "
                    "is missing object_name"
                )
            if not isinstance(entry.get("kind"), str):
                issues.append(
                    f"{_relative(repo_root, manifest_path)} selectors[{index}] is missing kind"
                )

    qss_builder_source = qss_builder_path.read_text(encoding="utf-8")
    referenced_names = _stylesheet_object_names(qss_builder_source)
    registered_names = _registered_object_names(manifest)
    missing_names = sorted(referenced_names - registered_names)
    for name in missing_names:
        issues.append(
            f"{_relative(repo_root, qss_builder_path)} references unregistered selector #{name}"
        )

    return issues


def _collect_tokens_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    tokens_path = _tokens_path(repo_root)
    if not tokens_path.exists():
        issues.append(f"{_relative(repo_root, tokens_path)} is missing")
        return issues

    tokens = _read_json(tokens_path)
    if "$schema" not in tokens:
        issues.append(f"{_relative(repo_root, tokens_path)} is missing top-level $schema")
    for key in ("color", "status"):
        if key not in tokens:
            issues.append(f"{_relative(repo_root, tokens_path)} is missing top-level {key}")
    return issues


def _collect_inline_stylesheet_issues(repo_root: Path, targets: Sequence[Path]) -> list[str]:
    issues: list[str] = []
    design_system_root = _design_system_root(repo_root).resolve()
    for path in _iter_python_files(targets):
        if path.is_relative_to(design_system_root):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "setStyleSheet(" in line:
                issues.append(
                    f"{_relative(repo_root, path)}:{line_number}: "
                    "setStyleSheet(...) is forbidden outside design_system/"
                )
    return issues


def _collect_hex_literal_issues(repo_root: Path, targets: Sequence[Path]) -> list[str]:
    issues: list[str] = []
    allowed_path = (_design_system_root(repo_root) / "tokens.py").resolve()
    for path in _iter_python_files(targets):
        if path == allowed_path:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if HEX_LITERAL_RE.search(line):
                issues.append(
                    f"{_relative(repo_root, path)}:{line_number}: hex literal is "
                    "forbidden outside design_system/tokens.py"
                )
    return issues


def _collect_api_client_import_issues(repo_root: Path, targets: Sequence[Path]) -> list[str]:
    issues: list[str] = []
    for path in _iter_python_files(targets):
        if not any(part in {"views", "widgets"} for part in path.parts):
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                if imported_module.endswith("api_client") or ".api_client" in imported_module:
                    issues.append(
                        f"{_relative(repo_root, path)}:{node.lineno}: views/widgets "
                        "must not import api_client modules"
                    )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("api_client") or ".api_client" in alias.name:
                        issues.append(
                            f"{_relative(repo_root, path)}:{node.lineno}: views/widgets "
                            "must not import api_client modules"
                        )
    return issues


def collect_design_system_issues(
    repo_root: Path,
    paths: Sequence[Path] | None = None,
) -> tuple[str, ...]:
    resolved_root = repo_root.resolve()
    targets = tuple(paths) if paths is not None else (_operator_console_root(resolved_root),)
    issues: list[str] = []
    issues.extend(_collect_manifest_issues(resolved_root))
    issues.extend(_collect_tokens_issues(resolved_root))
    issues.extend(_collect_inline_stylesheet_issues(resolved_root, targets))
    issues.extend(_collect_hex_literal_issues(resolved_root, targets))
    issues.extend(_collect_api_client_import_issues(resolved_root, targets))
    return tuple(issues)


def verify_design_system_artifacts(
    repo_root: Path,
    paths: Sequence[Path] | None = None,
) -> bool:
    return not collect_design_system_issues(repo_root, paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Operator Console design-system checks.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=_default_repo_root(),
        help="Repository root (default: inferred from this script path).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional files/directories to scan for code-surface checks.",
    )
    args = parser.parse_args(argv)

    try:
        targets = _resolve_targets(args.repo.resolve(), args.paths)
        issues = collect_design_system_issues(args.repo.resolve(), targets)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1

    print("design_system: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
