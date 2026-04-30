"""Runtime enumeration of §13 audit checklist items from spec content."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scripts.spec_ref_check import extract_content_from_pdf


@dataclass(frozen=True, slots=True)
class Section13Item:
    """One child item from §13 — Autonomous Implementation Audit Checklist."""

    item_id: str
    title: str
    body: str


_SECTION_13_ITEM_RE = re.compile(r"^13\.\d+$")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_item_id(raw_item: Mapping[str, Any], one_based_index: int) -> str:
    """Return a bare §13 child id such as ``13.7`` from a raw checklist item."""
    for field_name in ("item_id", "id", "section", "section_number"):
        value = raw_item.get(field_name)
        if value is None:
            continue
        candidate = _text(value).removeprefix("§").strip()
        if _SECTION_13_ITEM_RE.fullmatch(candidate):
            return candidate

    number = raw_item.get("item_number", one_based_index)
    number_text = _text(number)
    if number_text.endswith(".0"):
        number_text = number_text[:-2]
    candidate = f"13.{number_text}"
    if not _SECTION_13_ITEM_RE.fullmatch(candidate):
        raise ValueError(f"Invalid §13 audit item number: {number!r}")
    return candidate


def _coerce_item_title(raw_item: Mapping[str, Any], item_id: str) -> str:
    title = _text(raw_item.get("audit_item") or raw_item.get("title") or raw_item.get("name"))
    return title or item_id


def _coerce_item_body(raw_item: Mapping[str, Any]) -> str:
    for field_name in ("verification_criterion", "body", "criterion", "description", "content"):
        value = raw_item.get(field_name)
        if value is not None:
            return _text(value)

    metadata_fields = {
        "audit_item",
        "id",
        "item_id",
        "item_number",
        "name",
        "section",
        "section_number",
        "title",
    }
    remainder = {key: value for key, value in raw_item.items() if key not in metadata_fields}
    if not remainder:
        return ""
    return json.dumps(remainder, sort_keys=True, ensure_ascii=False)


def enumerate_section13_items(spec_content: Mapping[str, Any]) -> list[Section13Item]:
    """Enumerate the full ordered §13 child list from extracted spec content."""
    audit_checklist = spec_content.get("audit_checklist")
    if not isinstance(audit_checklist, Mapping):
        raise ValueError("Spec content does not contain an audit_checklist object")

    raw_items = audit_checklist.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        raise ValueError("Spec audit_checklist.items is not a list")

    items: list[Section13Item] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"Spec audit_checklist.items[{index - 1}] is not an object")
        item_mapping = cast(Mapping[str, Any], raw_item)
        item_id = _coerce_item_id(item_mapping, index)
        if item_id in seen_ids:
            raise ValueError(f"Duplicate §13 audit item id: {item_id}")
        seen_ids.add(item_id)
        items.append(
            Section13Item(
                item_id=item_id,
                title=_coerce_item_title(item_mapping, item_id),
                body=_coerce_item_body(item_mapping),
            )
        )

    if not items:
        raise ValueError("Spec audit_checklist.items is empty")
    return items


def enumerate_section13_items_from_pdf(pdf_path: Path) -> list[Section13Item]:
    """Extract content from any supplied spec PDF and enumerate §13 children."""
    return enumerate_section13_items(extract_content_from_pdf(pdf_path))
