"""Executable audit harness components."""

from scripts.audit.registry import (
    AuditContext,
    AuditRegistry,
    AuditVerifier,
    get_default_registry,
    register_audit_verifier,
    register_placeholder_verifiers,
)
from scripts.audit.results import AuditResult, render_table
from scripts.audit.spec_items import (
    Section13Item,
    enumerate_section13_items,
    enumerate_section13_items_from_pdf,
)

__all__ = [
    "AuditContext",
    "AuditRegistry",
    "AuditResult",
    "AuditVerifier",
    "Section13Item",
    "enumerate_section13_items",
    "enumerate_section13_items_from_pdf",
    "get_default_registry",
    "register_audit_verifier",
    "register_placeholder_verifiers",
    "render_table",
]
