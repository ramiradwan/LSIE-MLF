"""Strict-mode tests for the executable §13 audit harness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import scripts.audit.registry as audit_registry_module
from scripts import run_audit
from scripts.audit import AuditContext, AuditRegistry, AuditResult
from scripts.audit.spec_items import Section13Item, enumerate_section13_items


@pytest.fixture(autouse=True)
def fresh_default_audit_registry(monkeypatch: pytest.MonkeyPatch) -> AuditRegistry:
    """Isolate strict-mode tests from the process-global audit registry."""
    registry = AuditRegistry()
    monkeypatch.setattr(audit_registry_module, "_DEFAULT_REGISTRY", registry)
    return registry


def _write_pdf(repo_root: Path) -> Path:
    pdf_path = repo_root / "docs" / "tech-spec-vsynthetic.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"fake-pdf")
    return pdf_path


def _spec_content(*item_numbers: int) -> dict[str, Any]:
    return {
        "audit_checklist": {
            "items": [
                {
                    "item_number": item_number,
                    "audit_item": f"Synthetic strict item {item_number}",
                    "verification_criterion": f"Criterion for synthetic item {item_number}.",
                }
                for item_number in item_numbers
            ]
        }
    }


def _use_spec_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    spec_content: dict[str, Any],
) -> Path:
    pdf_path = _write_pdf(tmp_path)
    observed_paths: list[Path] = []

    def fake_extract(path: Path) -> dict[str, Any]:
        observed_paths.append(path)
        return spec_content

    monkeypatch.setattr(run_audit, "extract_content_from_pdf", fake_extract)
    return pdf_path


def test_main_threads_strict_argument_to_run_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[tuple[Path, set[str] | None, bool]] = []

    def fake_run_audit(
        repo_root: Path,
        item_ids: set[str] | None = None,
        *,
        strict: bool = False,
    ) -> int:
        observed_calls.append((repo_root, item_ids, strict))
        return 7 if strict else 0

    monkeypatch.setattr(run_audit, "run_audit", fake_run_audit)

    assert run_audit.main(["--repo", str(tmp_path), "--strict"]) == 7
    assert run_audit.main(["--repo", str(tmp_path)]) == 0
    assert observed_calls == [
        (tmp_path, None, True),
        (tmp_path, None, False),
    ]


def test_strict_fails_placeholder_backed_rows_even_when_placeholder_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_content = _spec_content(1, 2)
    expected_item_ids = [item.item_id for item in enumerate_section13_items(spec_content)]
    _use_spec_content(monkeypatch, tmp_path, spec_content)
    placeholder_item_ids: list[str] = []

    monkeypatch.setattr(
        run_audit,
        "_register_concrete_verifiers",
        lambda registry, items: None,
    )

    def register_passing_placeholders(
        registry: AuditRegistry,
        items: Sequence[Section13Item],
    ) -> None:
        for item in items:
            if registry.has_verifier(item.item_id):
                continue
            placeholder_item_ids.append(item.item_id)

            def placeholder(
                context: AuditContext,
                current_item: Section13Item,
            ) -> AuditResult:
                _ = context
                return AuditResult(
                    current_item.item_id,
                    current_item.title,
                    True,
                    "synthetic placeholder passed",
                )

            placeholder.__name__ = f"placeholder_{item.item_id.replace('.', '_')}"
            placeholder.__audit_placeholder__ = True  # type: ignore[attr-defined]
            registry.register(item.item_id)(placeholder)

    monkeypatch.setattr(
        run_audit,
        "register_placeholder_verifiers",
        register_passing_placeholders,
    )

    assert run_audit.run_audit(tmp_path) == 0
    non_strict_report = capsys.readouterr().out
    placeholder_row = "| 13.1 | Synthetic strict item 1 | PASS | synthetic placeholder passed | — |"
    assert placeholder_row in non_strict_report

    assert run_audit.run_audit(tmp_path, strict=True) == 1
    strict_report = capsys.readouterr().out
    assert placeholder_row in strict_report
    assert placeholder_item_ids == expected_item_ids


def test_strict_keeps_failed_audit_results_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_content = _spec_content(3)
    _use_spec_content(monkeypatch, tmp_path, spec_content)

    def register_failing_concrete_verifier(
        registry: AuditRegistry,
        items: Sequence[Section13Item],
    ) -> None:
        for item in items:
            if registry.has_verifier(item.item_id):
                continue

            def verifier(
                context: AuditContext,
                current_item: Section13Item,
            ) -> AuditResult:
                _ = context
                return AuditResult(
                    current_item.item_id,
                    current_item.title,
                    False,
                    "synthetic verifier failed",
                )

            registry.register(item.item_id)(verifier)

    monkeypatch.setattr(
        run_audit,
        "_register_concrete_verifiers",
        register_failing_concrete_verifier,
    )
    monkeypatch.setattr(
        run_audit,
        "register_placeholder_verifiers",
        lambda registry, items: None,
    )

    failed_row = "| 13.3 | Synthetic strict item 3 | FAIL | synthetic verifier failed | — |"

    assert run_audit.run_audit(tmp_path) == 1
    assert failed_row in capsys.readouterr().out

    assert run_audit.run_audit(tmp_path, strict=True) == 1
    assert failed_row in capsys.readouterr().out


def test_strict_fails_missing_concrete_binding_even_when_dispatch_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_content = _spec_content(9)
    _use_spec_content(monkeypatch, tmp_path, spec_content)

    monkeypatch.setattr(
        run_audit,
        "_register_concrete_verifiers",
        lambda registry, items: None,
    )
    monkeypatch.setattr(
        run_audit,
        "register_placeholder_verifiers",
        lambda registry, items: None,
    )

    def dispatch_passing_rows(
        registry: AuditRegistry,
        items: list[Section13Item],
        context: AuditContext,
    ) -> list[AuditResult]:
        _ = registry, context
        return [
            AuditResult(item.item_id, item.title, True, "synthetic dispatch passed")
            for item in items
        ]

    monkeypatch.setattr(run_audit, "dispatch_items", dispatch_passing_rows)

    dispatch_row = "| 13.9 | Synthetic strict item 9 | PASS | synthetic dispatch passed | — |"

    assert run_audit.run_audit(tmp_path) == 0
    assert dispatch_row in capsys.readouterr().out

    assert run_audit.run_audit(tmp_path, strict=True) == 1
    assert dispatch_row in capsys.readouterr().out
