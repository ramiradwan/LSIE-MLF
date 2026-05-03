#!/usr/bin/env python3
"""Executable §13 audit harness.

This is the single entry point for the audit-as-code workflow. It discovers the
committed spec PDF at runtime, extracts the embedded content, enumerates every
§13 checklist item, dispatches each item through the registry, and renders a
Markdown report.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit import (  # noqa: E402
    AuditContext,
    AuditRegistry,
    AuditResult,
    Section13Item,
    enumerate_section13_items,
    get_default_registry,
    register_placeholder_verifiers,
    render_table,
)
from scripts.audit.verifiers.data_classification import (  # noqa: E402
    register_data_classification_verifiers,
)
from scripts.audit.verifiers.mechanical import register_mechanical_verifiers  # noqa: E402
from scripts.spec_ref_check import extract_content_from_pdf  # noqa: E402


def discover_spec_pdf(repo_root: Path) -> Path:
    """Return the single runtime-discovered ``docs/tech-spec-v*.pdf`` path."""
    docs_dir = repo_root / "docs"
    matches = sorted(docs_dir.glob("tech-spec-v*.pdf"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No spec PDF found. Expected exactly one match for {docs_dir / 'tech-spec-v*.pdf'}."
        )
    relative_matches = ", ".join(str(path.relative_to(repo_root)) for path in matches)
    raise FileExistsError(
        f"Expected exactly one docs/tech-spec-v*.pdf match; found {len(matches)}: "
        f"{relative_matches}"
    )


def _register_concrete_verifiers(
    registry: AuditRegistry,
    items: Sequence[Section13Item],
) -> None:
    """Register every known concrete verifier for the requested §13 items."""
    from scripts.audit.verifiers.behavioral import register_behavioral_verifiers

    legacy_item_ids = tuple(item.item_id for item in items)
    register_mechanical_verifiers(
        legacy_item_ids,
        registry=registry,
        items=items,
    )
    register_data_classification_verifiers(
        registry=registry,
        items=items,
    )
    register_behavioral_verifiers(
        registry=registry,
        items=items,
    )


def build_placeholder_registry(items: list[Section13Item]) -> AuditRegistry:
    """Return the shared registry after backfilling placeholders for missing items."""

    registry = get_default_registry()
    _register_concrete_verifiers(registry, items)
    register_placeholder_verifiers(registry, items)
    return registry


def dispatch_items(
    registry: AuditRegistry,
    items: list[Section13Item],
    context: AuditContext,
) -> list[AuditResult]:
    """Dispatch checklist items in spec order."""
    return [registry.dispatch(item, context) for item in items]


def _registered_verifier(registry: AuditRegistry, item_id: str) -> object | None:
    """Return the registered verifier object for strict coverage checks."""

    verifiers = getattr(registry, "_verifiers", {})
    if not isinstance(verifiers, dict):
        return None
    return verifiers.get(item_id)


def _is_placeholder_verifier(verifier: object | None) -> bool:
    """Return whether ``verifier`` is an audit placeholder."""

    if verifier is None:
        return False
    return bool(getattr(verifier, "__audit_placeholder__", False)) or str(
        getattr(verifier, "__name__", "")
    ).startswith("placeholder_")


def _item_ids_without_concrete_verifier(
    registry: AuditRegistry,
    items: Sequence[Section13Item],
) -> tuple[str, ...]:
    """Return item ids that lack a concrete verifier binding in spec order."""

    missing: list[str] = []
    for item in items:
        verifier = _registered_verifier(registry, item.item_id)
        if verifier is None or _is_placeholder_verifier(verifier):
            missing.append(item.item_id)
    return tuple(missing)


def _placeholder_backed_item_ids(
    registry: AuditRegistry,
    items: Sequence[Section13Item],
) -> tuple[str, ...]:
    """Return item ids whose dispatched row is backed by a placeholder verifier."""

    return tuple(
        item.item_id
        for item in items
        if _is_placeholder_verifier(_registered_verifier(registry, item.item_id))
    )


def _strict_coverage_failed(
    results: Sequence[AuditResult],
    missing_concrete_item_ids: Sequence[str],
    placeholder_item_ids: Sequence[str],
) -> bool:
    """Return whether strict audit semantics should fail the process."""

    return (
        any(not result.passed for result in results)
        or bool(missing_concrete_item_ids)
        or bool(placeholder_item_ids)
    )


def run_audit(
    repo_root: Path,
    item_ids: set[str] | None = None,
    *,
    strict: bool = False,
) -> int:
    """Run the §13 audit and return a process exit code."""
    resolved_root = repo_root.resolve()
    pdf_path = discover_spec_pdf(resolved_root)
    spec_content = extract_content_from_pdf(pdf_path)
    items = enumerate_section13_items(spec_content)
    if item_ids is not None:
        available_ids = {item.item_id for item in items}
        missing_ids = sorted(item_ids - available_ids)
        if missing_ids:
            raise ValueError(f"Requested §13 item(s) not found in spec: {', '.join(missing_ids)}")
        items = [item for item in items if item.item_id in item_ids]
    registry = get_default_registry()
    _register_concrete_verifiers(registry, items)
    missing_concrete_item_ids = _item_ids_without_concrete_verifier(registry, items)
    register_placeholder_verifiers(registry, items)
    placeholder_item_ids = _placeholder_backed_item_ids(registry, items)
    context = AuditContext(repo_root=resolved_root, spec_content=spec_content)
    results = dispatch_items(registry, items, context)
    print(render_table(results))
    if strict:
        return 1 if _strict_coverage_failed(
            results,
            missing_concrete_item_ids,
            placeholder_item_ids,
        ) else 0
    return 1 if any(not result.passed for result in results) else 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper for the audit harness."""
    parser = argparse.ArgumentParser(description="Run the executable §13 audit harness.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=_default_repo_root(),
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--item",
        action="append",
        dest="items",
        metavar="13.N",
        help="Run only the supplied §13 item id. May be passed multiple times.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail when any dispatched §13 item fails (the harness default; "
            "provided so CI/local gates can declare strict intent explicitly)."
        ),
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")

    try:
        item_ids = set(args.items) if args.items else None
        return run_audit(args.repo, item_ids=item_ids, strict=args.strict)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
