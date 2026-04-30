"""Production verifier registration for executable audit controls."""

from scripts.audit.verifiers.behavioral import (
    BEHAVIORAL_AUDIT_ITEM_IDS,
    BEHAVIORAL_VERIFIERS,
    discover_audit_item_markers,
    register_behavioral_verifiers,
    verify_behavioral_item,
)
from scripts.audit.verifiers.data_classification import (
    DATA_CLASSIFICATION_VERIFIERS,
    register_data_classification_verifiers,
    scan_data_classification,
    verify_data_classification,
)
from scripts.audit.verifiers.mechanical import (
    MECHANICAL_VERIFIERS,
    register_mechanical_verifiers,
)

register_mechanical_verifiers()
register_data_classification_verifiers()

__all__ = [
    "BEHAVIORAL_AUDIT_ITEM_IDS",
    "BEHAVIORAL_VERIFIERS",
    "DATA_CLASSIFICATION_VERIFIERS",
    "MECHANICAL_VERIFIERS",
    "discover_audit_item_markers",
    "register_behavioral_verifiers",
    "register_data_classification_verifiers",
    "register_mechanical_verifiers",
    "scan_data_classification",
    "verify_behavioral_item",
    "verify_data_classification",
]
