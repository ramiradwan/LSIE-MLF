"""Tests for the executable §13 audit harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.audit.registry as audit_registry_module
from scripts import run_audit
from scripts.audit import (
    AuditContext,
    AuditRegistry,
    AuditResult,
    register_audit_verifier,
    spec_items,
)
from scripts.audit.spec_items import Section13Item


@pytest.fixture(autouse=True)
def fresh_default_audit_registry(monkeypatch: pytest.MonkeyPatch) -> AuditRegistry:
    """Isolate tests from the process-global audit registry."""
    registry = AuditRegistry()
    monkeypatch.setattr(audit_registry_module, "_DEFAULT_REGISTRY", registry)
    return registry


def _write_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-pdf")
    return path


def test_discover_spec_pdf_reports_zero_matches(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()

    with pytest.raises(FileNotFoundError, match=r"Expected exactly one match"):
        run_audit.discover_spec_pdf(tmp_path)


def test_discover_spec_pdf_returns_single_match(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "docs" / "tech-spec-vsynthetic.pdf")

    assert run_audit.discover_spec_pdf(tmp_path) == pdf_path


def test_discover_spec_pdf_reports_many_matches(tmp_path: Path) -> None:
    _write_pdf(tmp_path / "docs" / "tech-spec-va.pdf")
    _write_pdf(tmp_path / "docs" / "tech-spec-vb.pdf")

    with pytest.raises(FileExistsError, match=r"Expected exactly one docs/tech-spec-v\*\.pdf"):
        run_audit.discover_spec_pdf(tmp_path)


def test_enumerate_section13_items_from_supplied_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied_pdf = tmp_path / "custom-name.pdf"
    observed_paths: list[Path] = []

    def fake_extract(path: Path) -> dict[str, Any]:
        observed_paths.append(path)
        return {
            "audit_checklist": {
                "items": [
                    {
                        "item_number": 7,
                        "audit_item": "Runtime enumerated item",
                        "verification_criterion": "Body text from embedded content.",
                    }
                ]
            }
        }

    monkeypatch.setattr(spec_items, "extract_content_from_pdf", fake_extract)

    assert spec_items.enumerate_section13_items_from_pdf(supplied_pdf) == [
        Section13Item("13.7", "Runtime enumerated item", "Body text from embedded content.")
    ]
    assert observed_paths == [supplied_pdf]


def test_run_audit_prints_placeholder_report_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Use synthetic item numbers (98, 99) that no concrete verifier registers
    # against, so the harness must fall back to the placeholder mechanism. The
    # real §13.1 / §13.2 ids are now backed by concrete mechanical verifiers.
    pdf_path = _write_pdf(tmp_path / "docs" / "tech-spec-vsynthetic.pdf")
    spec_content = {
        "audit_checklist": {
            "items": [
                {
                    "item_number": 98,
                    "audit_item": "Synthetic placeholder A",
                    "verification_criterion": "Body text exercising the placeholder mechanism.",
                },
                {
                    "item_number": 99,
                    "audit_item": "Synthetic placeholder B",
                    "verification_criterion": "Second body text for placeholder mechanism.",
                },
            ]
        }
    }
    observed_paths: list[Path] = []

    def fake_extract(path: Path) -> dict[str, Any]:
        observed_paths.append(path)
        return spec_content

    monkeypatch.setattr(run_audit, "extract_content_from_pdf", fake_extract)

    exit_code = run_audit.run_audit(tmp_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert observed_paths == [pdf_path]
    assert "| 13.98 | Synthetic placeholder A | FAIL |" in captured.out
    assert "| 13.99 | Synthetic placeholder B | FAIL |" in captured.out
    assert "not yet implemented" in captured.out
    assert "Body text exercising the placeholder mechanism." in captured.out


def test_run_audit_honors_pre_registered_shared_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fresh_default_audit_registry: AuditRegistry,
) -> None:
    # Use synthetic item numbers 98/99 — see comment on the placeholder test.
    pdf_path = _write_pdf(tmp_path / "docs" / "tech-spec-vsynthetic.pdf")
    spec_content = {
        "audit_checklist": {
            "items": [
                {
                    "item_number": 98,
                    "audit_item": "Synthetic shared-verifier item",
                    "verification_criterion": "Body text routed to a pre-registered verifier.",
                },
                {
                    "item_number": 99,
                    "audit_item": "Synthetic placeholder fallback",
                    "verification_criterion": "Body text routed to the placeholder mechanism.",
                },
            ]
        }
    }
    observed_paths: list[Path] = []
    observed_verifier_calls: list[tuple[Path, Any, Section13Item]] = []

    @register_audit_verifier("13.98")
    def verify_synthetic(ctx: AuditContext, current_item: Section13Item) -> AuditResult:
        observed_verifier_calls.append((ctx.repo_root, ctx.spec_content, current_item))
        return AuditResult(
            current_item.item_id,
            current_item.title,
            True,
            "real shared verifier executed",
        )

    def fake_extract(path: Path) -> dict[str, Any]:
        observed_paths.append(path)
        return spec_content

    monkeypatch.setattr(run_audit, "extract_content_from_pdf", fake_extract)

    exit_code = run_audit.run_audit(tmp_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert observed_paths == [pdf_path]
    assert len(observed_verifier_calls) == 1
    observed_repo_root, observed_spec_content, observed_item = observed_verifier_calls[0]
    assert observed_repo_root == tmp_path.resolve()
    assert observed_spec_content is spec_content
    assert observed_item == Section13Item(
        "13.98",
        "Synthetic shared-verifier item",
        "Body text routed to a pre-registered verifier.",
    )
    assert "13.98" in fresh_default_audit_registry.item_ids
    assert "13.99" in fresh_default_audit_registry.item_ids
    assert (
        "| 13.98 | Synthetic shared-verifier item | PASS | real shared verifier executed | — |"
        in captured.out
    )
    assert "Verifier for §13.98 is not yet implemented" not in captured.out
    assert "| 13.99 | Synthetic placeholder fallback | FAIL |" in captured.out
    assert "Verifier for §13.99 is not yet implemented" in captured.out


def test_main_returns_nonzero_on_discovery_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "docs").mkdir()

    assert run_audit.main(["--repo", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "Expected exactly one match" in captured.err
