"""Design-system entry points for the Operator Console."""

from __future__ import annotations

from services.operator_console.design_system.components import COMPONENT_REGISTRY, COMPONENT_SPECS
from services.operator_console.design_system.qss_builder import (
    STYLESHEET,
    build_stylesheet,
    install_application_stylesheet,
    repolish,
)
from services.operator_console.design_system.shells import SHELL_REGISTRY, SHELL_SPECS
from services.operator_console.design_system.tokens import PALETTE, Palette, token_manifest

__all__ = [
    "COMPONENT_REGISTRY",
    "COMPONENT_SPECS",
    "PALETTE",
    "Palette",
    "SHELL_REGISTRY",
    "SHELL_SPECS",
    "STYLESHEET",
    "build_stylesheet",
    "install_application_stylesheet",
    "repolish",
    "token_manifest",
]
