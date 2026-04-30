"""Tests for version-agnostic spec reference tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import spec_ref_check


def test_load_content_discovers_single_committed_spec_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_path = docs_dir / f"tech-spec-v{'9.9'}.pdf"
    pdf_path.write_bytes(b"fake-pdf")

    def fake_extract(path: Path) -> dict[str, Any]:
        return {"loaded_from": path.name}

    monkeypatch.setattr(spec_ref_check, "extract_content_from_pdf", fake_extract)

    assert spec_ref_check.load_content(repo_root=tmp_path) == {
        "loaded_from": f"tech-spec-v{'9.9'}.pdf"
    }


def test_load_content_requires_exactly_one_default_spec_pdf(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / f"tech-spec-v{'1.0'}.pdf").write_bytes(b"one")
    (docs_dir / f"tech-spec-v{'2.0'}.pdf").write_bytes(b"two")

    with pytest.raises(FileExistsError, match=r"Expected exactly one docs/tech-spec-v\*\.pdf"):
        spec_ref_check.load_content(repo_root=tmp_path)


def test_load_content_reports_missing_default_spec_pdf(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()

    with pytest.raises(FileNotFoundError, match=r"Expected exactly one match"):
        spec_ref_check.load_content(repo_root=tmp_path)


def test_explicit_content_path_does_not_require_committed_pdf(tmp_path: Path) -> None:
    content_path = tmp_path / "content.json"
    content_path.write_text(json.dumps({"document_control": {"section_title": "Doc"}}))

    assert spec_ref_check.load_content(content_path=content_path, repo_root=tmp_path) == {
        "document_control": {"section_title": "Doc"}
    }


def test_expand_ref_range_supports_abbreviated_same_prefix_end() -> None:
    assert spec_ref_check.expand_ref_range("11.5.13", "14") == ["11.5.13", "11.5.14"]
