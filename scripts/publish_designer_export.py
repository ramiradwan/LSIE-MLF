"""Package immutable Operator Console designer-reference exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

EXPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_CONTRACTS = (
    "tokens.json",
    "reference_capture_manifest.json",
    "reference_to_qt_mapping.json",
)
REFERENCE_SUFFIXES = frozenset((".html", ".css", ".md"))
ROBOTS_META = '<meta name="robots" content="noindex, nofollow">'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _inject_noindex(html_path: Path) -> None:
    content = html_path.read_text(encoding="utf-8")
    if ROBOTS_META in content:
        return
    head_match = re.search(r"<head(\s[^>]*)?>", content, flags=re.IGNORECASE)
    if head_match is None:
        raise ValueError(f"{html_path} must contain a <head> element")
    insert_at = head_match.end()
    content = f"{content[:insert_at]}\n  {ROBOTS_META}{content[insert_at:]}"
    html_path.write_text(content, encoding="utf-8")


def _copy_reference_files(source: Path, target: Path) -> int:
    copied = 0
    for item in sorted(source.iterdir()):
        if item.is_file() and item.suffix.lower() in REFERENCE_SUFFIXES:
            destination = target / item.name
            shutil.copy2(item, destination)
            if destination.suffix.lower() == ".html":
                _inject_noindex(destination)
            copied += 1
    uploads = source / "uploads"
    if uploads.is_dir():
        uploads_target = target / "uploads"
        uploads_target.mkdir(exist_ok=True)
        for image in sorted(uploads.glob("*.png")):
            shutil.copy2(image, uploads_target / image.name)
            copied += 1
    return copied


def _contract_source_path(source: Path, filename: str) -> Path:
    candidates = (source / "contract" / filename, source / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required contract file: {filename}")


def _copy_contracts(source: Path, target: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in REQUIRED_CONTRACTS:
        source_file = _contract_source_path(source, filename)
        _read_json(source_file)
        destination = target / filename
        shutil.copy2(source_file, destination)
        hashes[f"contract/{filename}"] = _sha256(destination)
    return hashes


def _validate_args(source: Path, export_id: str, spec_checkout: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not EXPORT_ID_RE.fullmatch(export_id):
        raise ValueError("export-id must use only letters, numbers, dots, underscores, and dashes")
    if any(part in {"", ".", ".."} for part in Path(export_id).parts):
        raise ValueError("export-id must be a single relative export name")
    if not spec_checkout.is_dir():
        raise FileNotFoundError(f"design-system-spec checkout does not exist: {spec_checkout}")


def publish_export(source: Path, export_id: str, spec_checkout: Path) -> Path:
    source = source.resolve()
    spec_checkout = spec_checkout.resolve()
    _validate_args(source, export_id, spec_checkout)

    target_root = spec_checkout / "exports" / export_id
    reference_target = target_root / "reference"
    contract_target = target_root / "contract"
    if target_root.exists():
        raise FileExistsError(
            f"Export already exists: {target_root}. Choose a new export-id; exports are immutable."
        )

    try:
        reference_target.mkdir(parents=True)
        contract_target.mkdir()
        reference_count = _copy_reference_files(source, reference_target)
        if reference_count == 0:
            raise ValueError(
                f"No reference .html, .css, .md, or uploads/*.png files found in {source}"
            )
        contract_hashes = _copy_contracts(source, contract_target)
        manifest = {
            "$schema": "https://lsie-mlf.local/operator-console/designer-export-manifest.schema.json",
            "version": 1,
            "export_id": export_id,
            "created_at": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "contract_hashes": contract_hashes,
        }
        (target_root / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if target_root.exists():
            shutil.rmtree(target_root)
        raise

    return target_root


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a new immutable Operator Console designer-reference export.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the designer's local reference folder.",
    )
    parser.add_argument(
        "--export-id",
        required=True,
        help="Unique export id, for example refresh-2026-06.",
    )
    parser.add_argument(
        "--spec-checkout",
        type=Path,
        required=True,
        help="Path to a local checkout of the design-system-spec branch.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        target_root = publish_export(args.source, args.export_id, args.spec_checkout)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    spec_checkout = args.spec_checkout.resolve()
    relative_target = target_root.relative_to(spec_checkout).as_posix()
    print(f"Packaged designer export: {relative_target}")
    print("Next maintainer steps:")
    print(f"  git add {relative_target}")
    print(f'  git commit -m "Publish designer export {args.export_id}"')
    print("  git push origin design-system-spec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
