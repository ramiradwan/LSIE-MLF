"""Generate Operator Console PySide token artifacts from a verified designer export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    REPO_ROOT / "services/operator_console/design_system/designer_export_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "services/operator_console/design_system"
GENERATED_HEADER = "# AUTO-GENERATED FROM DESIGNER EXPORT. DO NOT EDIT."
TOKEN_SCHEMA = "https://www.designtokens.org/TR/drafts/format/"
_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]+")


@dataclass(frozen=True)
class TokenField:
    section: str
    name: str
    value: str

    @property
    def python_name(self) -> str:
        if self.section == "status":
            return f"status_{_python_identifier(self.name)}"
        return _python_identifier(self.name)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _python_identifier(raw: str) -> str:
    normalized = _IDENTIFIER_RE.sub("_", raw.replace("-", "_")).strip("_").lower()
    if not normalized:
        raise ValueError(f"Token name {raw!r} does not normalize to a Python identifier")
    if normalized[0].isdigit():
        normalized = f"token_{normalized}"
    return normalized


def _export_tokens_path(export_dir: Path) -> Path:
    contract_path = export_dir / "contract" / "tokens.json"
    if contract_path.exists():
        return contract_path
    tokens_path = export_dir / "tokens.json"
    if tokens_path.exists():
        return tokens_path
    raise FileNotFoundError(
        f"Could not find contract/tokens.json or tokens.json under {export_dir}"
    )


def _expected_contract_hashes(manifest: Mapping[str, object]) -> Mapping[str, object]:
    hashes = manifest.get("contract_hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("designer export manifest must define contract_hashes")
    return hashes


def _contract_file_path(export_dir: Path, relative_path: str) -> Path:
    path = export_dir / relative_path
    if path.exists():
        return path
    if relative_path.startswith("contract/"):
        fallback = export_dir / relative_path.removeprefix("contract/")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find {relative_path} under {export_dir}")


def _verify_contract_hashes(export_dir: Path, manifest: Mapping[str, object]) -> None:
    for relative_path, expected_hash in _expected_contract_hashes(manifest).items():
        if not isinstance(relative_path, str):
            raise ValueError("designer export manifest contract hash keys must be strings")
        if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256:"):
            raise ValueError(
                f"designer export manifest must define {relative_path} as a sha256 value"
            )
        contract_file = _contract_file_path(export_dir, relative_path)
        actual_hash = _sha256(contract_file)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{contract_file} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )


def _verify_manifest(
    export_dir: Path,
    manifest_path: Path,
    *,
    verify_contract_hashes: bool,
) -> tuple[dict[str, object], Path]:
    manifest = _read_json(manifest_path)
    if verify_contract_hashes:
        _verify_contract_hashes(export_dir, manifest)
    return manifest, _export_tokens_path(export_dir)


def _extract_tokens(tokens: Mapping[str, object]) -> tuple[TokenField, ...]:
    fields: list[TokenField] = []
    for section_name in ("color", "status"):
        section = tokens.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"tokens.json must define object section {section_name}")
        for token_name, token_payload in section.items():
            if not isinstance(token_name, str) or not isinstance(token_payload, Mapping):
                raise ValueError(f"Invalid token entry in {section_name}")
            value = token_payload.get("$value")
            token_type = token_payload.get("$type")
            if not isinstance(value, str):
                raise ValueError(f"Token {section_name}.{token_name} must define string $value")
            if token_type != "color":
                raise ValueError(f"Token {section_name}.{token_name} must use $type=color")
            fields.append(TokenField(section_name, token_name, value))
    python_names = [field.python_name for field in fields]
    duplicates = sorted({name for name in python_names if python_names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate generated token field names: " + ", ".join(duplicates))
    return tuple(fields)


def render_tokens_py(tokens: Mapping[str, object]) -> str:
    fields = _extract_tokens(tokens)
    lines = [
        GENERATED_HEADER,
        '"""Canonical design tokens for the Operator Console."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class Palette:",
    ]
    for field in fields:
        lines.append(f'    {field.python_name}: str = "{field.value}"')
    lines.extend(
        [
            "",
            "",
            "PALETTE = Palette()",
            "",
            "",
            "def token_manifest(palette: Palette = PALETTE) -> dict[str, object]:",
            "    return {",
            f'        "$schema": "{TOKEN_SCHEMA}",',
        ]
    )
    for section_name in ("color", "status"):
        section_fields = [field for field in fields if field.section == section_name]
        lines.append(f'        "{section_name}": {{')
        for field in section_fields:
            lines.append(
                f'            "{field.name}": '
                f'{{"$value": palette.{field.python_name}, "$type": "color"}},'
            )
        lines.append("        },")
    lines.extend(["    }", ""])
    return "\n".join(lines)


def render_tokens_json(tokens: Mapping[str, object]) -> str:
    normalized = {
        "$schema": tokens.get("$schema", TOKEN_SCHEMA),
        "color": tokens["color"],
        "status": tokens["status"],
    }
    return json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"


def _write_outputs(tokens: Mapping[str, object], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tokens.py").write_text(render_tokens_py(tokens), encoding="utf-8")
    (output_root / "tokens.json").write_text(render_tokens_json(tokens), encoding="utf-8")


def _check_outputs(tokens: Mapping[str, object], output_root: Path) -> list[str]:
    expected = {
        output_root / "tokens.py": render_tokens_py(tokens),
        output_root / "tokens.json": render_tokens_json(tokens),
    }
    mismatches: list[str] = []
    for path, expected_content in expected.items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            mismatches.append(f"{path} is missing")
            continue
        if actual != expected_content:
            mismatches.append(f"{path} is out of date")
    return mismatches


def run_generation(
    export_dir: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    check: bool,
) -> int:
    verify_contract_hashes = export_dir.resolve() != output_root.resolve()
    _, tokens_path = _verify_manifest(
        export_dir,
        manifest_path,
        verify_contract_hashes=verify_contract_hashes,
    )
    tokens = _read_json(tokens_path)
    _extract_tokens(tokens)
    if check:
        mismatches = _check_outputs(tokens, output_root)
        if mismatches:
            for mismatch in mismatches:
                print(f"ERROR: {mismatch}", file=sys.stderr)
            print(
                "Run uv run python scripts/generate_pyside_tokens.py "
                f"--export-dir {export_dir} --manifest {manifest_path}",
                file=sys.stderr,
            )
            return 1
        print("generate_pyside_tokens: OK")
        return 0
    _write_outputs(tokens, output_root)
    print(f"Generated PySide token artifacts from {tokens_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Operator Console PySide tokens from a verified designer export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Designer export directory containing contract/tokens.json or tokens.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Consumed designer export manifest with expected token hashes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where tokens.py and tokens.json are written or checked.",
    )
    parser.add_argument("--check", action="store_true", help="Fail if generated outputs differ.")
    args = parser.parse_args(argv)

    try:
        return run_generation(
            args.export_dir.resolve(),
            args.manifest.resolve(),
            args.output_root.resolve(),
            check=args.check,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
