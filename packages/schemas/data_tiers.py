"""Machine-readable §5.2 data classification tier markers.

The enum values intentionally mirror the spec-named §5.2 tier labels exactly.
Helpers in this module are runtime no-ops: they make data-tier intent explicit
for AST verifiers and type checkers without changing production payload values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ParamSpec, TypeVar, cast


class DataTier(StrEnum):
    """Spec-bounded §5.2 data classification tiers."""

    TRANSIENT = "Transient Data"
    DEBUG = "Debug Storage"
    PERMANENT = "Permanent Analytical Storage"


@dataclass(frozen=True, slots=True)
class DataTierAnnotation:
    """Structured marker payload used by decorators and verifier-readable calls."""

    tier: DataTier
    spec_ref: str
    purpose: str = ""


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def mark_data_tier(
    value: T,
    tier: DataTier,
    *,
    spec_ref: str,
    purpose: str = "",
) -> T:
    """Annotate a value/call site with a §5.2 tier while preserving its type.

    Strings, bytes, dicts, and other built-in payloads cannot safely carry
    arbitrary attributes, so the marker is intentionally a transparent helper.
    The AST verifier treats this call as the machine-readable evidence.
    """
    _ = DataTierAnnotation(tier=tier, spec_ref=spec_ref, purpose=purpose)
    return value


def data_tier(
    tier: DataTier,
    *,
    spec_ref: str,
    purpose: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function with a structured §5.2 data-tier annotation."""

    annotation = DataTierAnnotation(tier=tier, spec_ref=spec_ref, purpose=purpose)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cast(Any, func).__lsie_data_tier__ = annotation
        return func

    return decorator


@contextmanager
def data_tier_context(
    tier: DataTier,
    *,
    spec_ref: str,
    purpose: str = "",
) -> Iterator[DataTierAnnotation]:
    """Context marker for structured calls that need block-level annotation."""

    yield DataTierAnnotation(tier=tier, spec_ref=spec_ref, purpose=purpose)


def get_data_tier_annotation(value: object) -> DataTierAnnotation | None:
    """Return a decorator-attached annotation when one exists."""

    annotation = getattr(value, "__lsie_data_tier__", None)
    if isinstance(annotation, DataTierAnnotation):
        return annotation
    return None


__all__ = [
    "DataTier",
    "DataTierAnnotation",
    "data_tier",
    "data_tier_context",
    "get_data_tier_annotation",
    "mark_data_tier",
]
