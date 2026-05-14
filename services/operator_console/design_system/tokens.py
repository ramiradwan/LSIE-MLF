# AUTO-GENERATED FROM DESIGNER EXPORT. DO NOT EDIT.
"""Canonical design tokens for the Operator Console."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    background: str = "#0f1115"
    surface: str = "#171a21"
    surface_raised: str = "#1d2029"
    border: str = "#262a33"
    text_primary: str = "#e6e8ed"
    text_muted: str = "#8a909b"
    text_inverse: str = "#ffffff"
    accent: str = "#5b8def"
    status_ok: str = "#4ecb71"
    status_warn: str = "#e7b34a"
    status_bad: str = "#e26a6a"
    status_recovering: str = "#6cc3d5"
    status_degraded: str = "#d59b4a"


PALETTE = Palette()


def token_manifest(palette: Palette = PALETTE) -> dict[str, object]:
    return {
        "$schema": "https://www.designtokens.org/TR/drafts/format/",
        "color": {
            "background": {"$value": palette.background, "$type": "color"},
            "surface": {"$value": palette.surface, "$type": "color"},
            "surface_raised": {"$value": palette.surface_raised, "$type": "color"},
            "border": {"$value": palette.border, "$type": "color"},
            "text_primary": {"$value": palette.text_primary, "$type": "color"},
            "text_muted": {"$value": palette.text_muted, "$type": "color"},
            "text_inverse": {"$value": palette.text_inverse, "$type": "color"},
            "accent": {"$value": palette.accent, "$type": "color"},
        },
        "status": {
            "ok": {"$value": palette.status_ok, "$type": "color"},
            "warn": {"$value": palette.status_warn, "$type": "color"},
            "bad": {"$value": palette.status_bad, "$type": "color"},
            "recovering": {"$value": palette.status_recovering, "$type": "color"},
            "degraded": {"$value": palette.status_degraded, "$type": "color"},
        },
    }
