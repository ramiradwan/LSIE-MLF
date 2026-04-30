"""Verifier registry for executable §13 audit checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.audit.results import AuditResult
from scripts.audit.spec_items import Section13Item


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Context passed to audit verifiers."""

    repo_root: Path
    spec_content: Mapping[str, Any]


AuditVerifier = Callable[[AuditContext, Section13Item], AuditResult]


class AuditRegistry:
    """Map bare §13 item ids to deterministic verifier callables."""

    def __init__(self) -> None:
        self._verifiers: dict[str, AuditVerifier] = {}

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Registered item ids in insertion order."""
        return tuple(self._verifiers)

    def has_verifier(self, item_id: str) -> bool:
        """Return whether ``item_id`` has a registered verifier."""
        return item_id in self._verifiers

    def register(self, item_id: str) -> Callable[[AuditVerifier], AuditVerifier]:
        """Register a verifier for a bare §13 item id.

        The decorator returns the verifier unchanged so call sites can keep a
        direct reference when desired. Duplicate registration is rejected to
        avoid accidentally shadowing a real verifier with a placeholder.
        """

        def decorator(verifier: AuditVerifier) -> AuditVerifier:
            if item_id in self._verifiers:
                raise ValueError(f"Verifier already registered for {item_id}")
            self._verifiers[item_id] = verifier
            return verifier

        return decorator

    def dispatch(self, item: Section13Item, context: AuditContext) -> AuditResult:
        """Run the verifier for ``item`` or return a failing missing-verifier result."""
        verifier = self._verifiers.get(item.item_id)
        if verifier is None:
            return AuditResult(
                item_id=item.item_id,
                title=item.title,
                passed=False,
                evidence=f"No verifier registered for §{item.item_id}.",
                follow_up="Register a deterministic verifier for this §13 audit item.",
            )
        return verifier(context, item)


_DEFAULT_REGISTRY = AuditRegistry()


def get_default_registry() -> AuditRegistry:
    """Return the shared verifier registry used by the audit harness."""
    return _DEFAULT_REGISTRY


def register_audit_verifier(item_id: str) -> Callable[[AuditVerifier], AuditVerifier]:
    """Register a verifier on the shared audit registry.

    This decorator is the pluggable declaration surface for production verifier
    modules. It delegates to the shared registry while preserving the
    instance-level ``AuditRegistry.register`` behavior.
    """
    return get_default_registry().register(item_id)


def register_placeholder_verifiers(
    registry: AuditRegistry,
    items: list[Section13Item],
) -> None:
    """Ensure every enumerated item has a failing placeholder verifier.

    Existing entries on ``registry`` are left intact so real verifiers registered
    through the shared declaration surface are not replaced by placeholders.
    """
    for item in items:
        if registry.has_verifier(item.item_id):
            continue

        def placeholder(context: AuditContext, current_item: Section13Item) -> AuditResult:
            _ = context
            return AuditResult(
                item_id=current_item.item_id,
                title=current_item.title,
                passed=False,
                evidence=(
                    f"Verifier for §{current_item.item_id} is not yet implemented. "
                    f"Checklist body: {current_item.body}"
                ),
                follow_up="Replace this placeholder with an implementation-specific verifier.",
            )

        placeholder.__name__ = f"placeholder_{item.item_id.replace('.', '_')}"
        registry.register(item.item_id)(placeholder)
