from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.generate_pyside_tokens import main, render_tokens_py


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _token_payload() -> dict[str, object]:
    return {
        "$schema": "https://www.designtokens.org/TR/drafts/format/",
        "color": {
            "surface-raised": {"$value": "#101820", "$type": "color"},
            "text_primary": {"$value": "#f8fafc", "$type": "color"},
        },
        "status": {
            "ok": {"$value": "#22c55e", "$type": "color"},
            "bad-state": {"$value": "#ef4444", "$type": "color"},
        },
    }


def _build_export(tmp_path: Path, tokens: dict[str, object] | None = None) -> tuple[Path, Path]:
    export_dir = tmp_path / "export"
    token_payload = tokens or _token_payload()
    tokens_path = export_dir / "contract" / "tokens.json"
    _write_json(tokens_path, token_payload)
    manifest_path = tmp_path / "designer_export_manifest.json"
    capture_manifest_path = export_dir / "contract" / "reference_capture_manifest.json"
    mapping_path = export_dir / "contract" / "reference_to_qt_mapping.json"
    _write_json(capture_manifest_path, {"version": 1, "captures": []})
    _write_json(mapping_path, {"version": 1, "routes": [], "primitives": []})
    _write_json(
        manifest_path,
        {
            "$schema": "https://lsie-mlf.local/operator-console/designer-export-manifest.schema.json",
            "version": 1,
            "export_id": "unit-test-export",
            "export_ref": "unit-test",
            "artifact_uri": None,
            "contract_hashes": {
                "contract/tokens.json": _sha256(tokens_path),
                "contract/reference_capture_manifest.json": _sha256(capture_manifest_path),
                "contract/reference_to_qt_mapping.json": _sha256(mapping_path),
            },
            "generated_files": {
                "tokens.py": "services/operator_console/design_system/tokens.py",
                "tokens.json": "services/operator_console/design_system/tokens.json",
            },
        },
    )
    return export_dir, manifest_path


def test_render_tokens_py_normalizes_names() -> None:
    rendered = render_tokens_py(_token_payload())

    assert "# AUTO-GENERATED FROM DESIGNER EXPORT. DO NOT EDIT." in rendered
    assert 'surface_raised: str = "#101820"' in rendered
    assert 'status_bad_state: str = "#ef4444"' in rendered
    assert "palette.status_bad_state" in rendered


def test_main_generates_and_checks_outputs(tmp_path: Path) -> None:
    export_dir, manifest_path = _build_export(tmp_path)
    output_root = tmp_path / "generated"

    assert (
        main(
            [
                "--export-dir",
                str(export_dir),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    assert (output_root / "tokens.py").is_file()
    assert (output_root / "tokens.json").is_file()

    assert (
        main(
            [
                "--export-dir",
                str(export_dir),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(output_root),
                "--check",
            ]
        )
        == 0
    )


def test_main_check_fails_when_outputs_are_stale(tmp_path: Path) -> None:
    export_dir, manifest_path = _build_export(tmp_path)
    output_root = tmp_path / "generated"
    output_root.mkdir()
    (output_root / "tokens.py").write_text("stale\n", encoding="utf-8")
    (output_root / "tokens.json").write_text("{}\n", encoding="utf-8")

    assert (
        main(
            [
                "--export-dir",
                str(export_dir),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(output_root),
                "--check",
            ]
        )
        == 1
    )


def test_main_rejects_hash_mismatch(tmp_path: Path) -> None:
    export_dir, manifest_path = _build_export(tmp_path)
    tokens_path = export_dir / "contract" / "tokens.json"
    _write_json(tokens_path, _token_payload() | {"extra": "changed"})

    assert (
        main(
            [
                "--export-dir",
                str(export_dir),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(tmp_path / "generated"),
            ]
        )
        == 1
    )
