"""Structured audit result rendering utilities.

The audit harness produces one :class:`AuditResult` per §13 checklist item.
Keeping the result shape small and deterministic makes it suitable for both
human-readable Markdown reports and later machine-enforced CI gates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Outcome returned by a verifier for one §13 audit item."""

    item_id: str
    title: str
    passed: bool
    evidence: str
    follow_up: str | None = None


def _markdown_cell(value: object) -> str:
    """Render a stable, single-line Markdown table cell."""
    text = "—" if value is None else str(value)
    return text.replace("|", r"\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def render_table(results: Iterable[AuditResult]) -> str:
    """Render audit results as a deterministic Markdown table.

    The caller controls row order; the harness passes results in spec order so
    the table mirrors the runtime-enumerated §13 checklist.
    """
    lines = [
        "| Item | Title | Status | Evidence | Follow-up |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(result.item_id),
                    _markdown_cell(result.title),
                    status,
                    _markdown_cell(result.evidence),
                    _markdown_cell(result.follow_up),
                ]
            )
            + " |"
        )
    return "\n".join(lines)
