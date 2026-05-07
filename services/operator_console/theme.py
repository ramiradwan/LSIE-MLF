"""Compatibility theme entry points for the Operator Console."""

from __future__ import annotations

from services.operator_console.design_system import (
    PALETTE,
    STYLESHEET,
    Palette,
    build_stylesheet,
    install_application_stylesheet,
    repolish,
)

__all__ = [
    "PALETTE",
    "Palette",
    "STYLESHEET",
    "build_stylesheet",
    "install_application_stylesheet",
    "repolish",
]
