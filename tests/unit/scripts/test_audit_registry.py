"""Tests for the §13 audit registry and result rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.audit.registry as audit_registry_module
from scripts.audit import (
    AuditContext,
    AuditRegistry,
    AuditResult,
    Section13Item,
    get_default_registry,
    register_audit_verifier,
    register_placeholder_verifiers,
    render_table,
)


def test_registry_dispatches_known_item_with_context(tmp_path: Path) -> None:
    registry = AuditRegistry()
    item = Section13Item("13.1", "Directory structure", "Repository directories match §3.")
    spec_content: dict[str, Any] = {"audit_checklist": {"items": []}}
    context = AuditContext(repo_root=tmp_path, spec_content=spec_content)

    @registry.register("13.1")
    def verify_directory(ctx: AuditContext, current_item: Section13Item) -> AuditResult:
        assert ctx.repo_root == tmp_path
        assert ctx.spec_content is spec_content
        assert current_item == item
        return AuditResult(
            item_id=current_item.item_id,
            title=current_item.title,
            passed=True,
            evidence="known verifier executed",
        )

    result = registry.dispatch(item, context)

    assert result == AuditResult("13.1", "Directory structure", True, "known verifier executed")


def test_shared_registry_accessor_and_decorator_register_on_same_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_registry = AuditRegistry()
    monkeypatch.setattr(audit_registry_module, "_DEFAULT_REGISTRY", shared_registry)
    item = Section13Item("13.5", "Shared", "Registered through shared decorator.")
    context = AuditContext(repo_root=tmp_path, spec_content={})

    @register_audit_verifier("13.5")
    def verify_shared(ctx: AuditContext, current_item: Section13Item) -> AuditResult:
        assert ctx == context
        return AuditResult(current_item.item_id, current_item.title, True, "shared verifier")

    assert get_default_registry() is shared_registry
    assert shared_registry.item_ids == ("13.5",)
    assert shared_registry.dispatch(item, context) == AuditResult(
        "13.5", "Shared", True, "shared verifier"
    )
    assert verify_shared.__name__ == "verify_shared"


def test_missing_verifier_dispatch_returns_failure(tmp_path: Path) -> None:
    registry = AuditRegistry()
    item = Section13Item("13.9", "Unregistered", "Needs a verifier.")
    context = AuditContext(repo_root=tmp_path, spec_content={})

    result = registry.dispatch(item, context)

    assert result.passed is False
    assert result.item_id == "13.9"
    assert "No verifier registered" in result.evidence
    assert result.follow_up is not None


def test_placeholder_verifiers_fail_without_overriding_existing_verifier(tmp_path: Path) -> None:
    registry = AuditRegistry()
    known = Section13Item("13.1", "Known", "Already implemented.")
    placeholder = Section13Item("13.2", "Placeholder", "Pending implementation.")
    context = AuditContext(repo_root=tmp_path, spec_content={})

    @registry.register("13.1")
    def verify_known(ctx: AuditContext, current_item: Section13Item) -> AuditResult:
        assert ctx == context
        return AuditResult(current_item.item_id, current_item.title, True, "real verifier")

    register_placeholder_verifiers(registry, [known, placeholder])

    known_result = registry.dispatch(known, context)
    placeholder_result = registry.dispatch(placeholder, context)

    assert known_result.passed is True
    assert known_result.evidence == "real verifier"
    assert placeholder_result.passed is False
    assert "not yet implemented" in placeholder_result.evidence
    assert "Pending implementation." in placeholder_result.evidence
    assert placeholder_result.follow_up is not None


def test_render_table_is_deterministic_and_escapes_cells() -> None:
    results = [
        AuditResult("13.2", "Two", False, "line 1\nline 2", "fix | later"),
        AuditResult("13.1", "One", True, "ok", None),
    ]

    assert render_table(results) == "\n".join(
        [
            "| Item | Title | Status | Evidence | Follow-up |",
            "| --- | --- | --- | --- | --- |",
            "| 13.2 | Two | FAIL | line 1<br>line 2 | fix \\| later |",
            "| 13.1 | One | PASS | ok | — |",
        ]
    )
