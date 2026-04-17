"""Minimal dark-neutral QSS stylesheet for the Operator Console.

A single exported stylesheet (`STYLESHEET`) keeps the visual style in one
place. Component-specific styling uses object names (`#ContentSurface`,
`#SidebarNav`, etc.) rather than inline ``setStyleSheet`` calls so the
whole look can be swapped later without touching widget code.

The palette is intentionally restrained: near-black surfaces, a single
accent, and subtle dividers. No glass gradients, no glow effects —
different in spirit from the developer-facing Debug Studio on purpose.
"""

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
    accent: str = "#5b8def"
    status_ok: str = "#4ecb71"
    status_warn: str = "#e7b34a"
    status_bad: str = "#e26a6a"


PALETTE = Palette()


STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 13px;
    color: {PALETTE.text_primary};
}}

QMainWindow, QDialog {{
    background: {PALETTE.background};
}}

QWidget#ContentSurface {{
    background: {PALETTE.background};
}}

QFrame#SidebarNav {{
    background: {PALETTE.surface};
    border-right: 1px solid {PALETTE.border};
}}

QPushButton#NavButton {{
    background: transparent;
    border: none;
    text-align: left;
    padding: 10px 16px;
    color: {PALETTE.text_muted};
    border-left: 2px solid transparent;
}}
QPushButton#NavButton:hover {{
    color: {PALETTE.text_primary};
    background: {PALETTE.surface_raised};
}}
QPushButton#NavButton:checked {{
    color: {PALETTE.text_primary};
    background: {PALETTE.surface_raised};
    border-left: 2px solid {PALETTE.accent};
}}

QFrame#Panel {{
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
    border-radius: 6px;
}}

QLabel#PanelTitle {{
    font-size: 14px;
    font-weight: 600;
    color: {PALETTE.text_primary};
    padding: 12px 16px 6px 16px;
}}

QLabel#PanelSubtitle {{
    color: {PALETTE.text_muted};
    padding: 0 16px 10px 16px;
}}

QLabel#StatusBarLabel {{
    color: {PALETTE.text_muted};
}}

QTableView {{
    background: {PALETTE.surface};
    alternate-background-color: {PALETTE.surface_raised};
    gridline-color: {PALETTE.border};
    border: none;
    selection-background-color: {PALETTE.accent};
    selection-color: {PALETTE.text_primary};
}}
QHeaderView::section {{
    background: {PALETTE.surface_raised};
    color: {PALETTE.text_muted};
    padding: 6px 10px;
    border: none;
    border-bottom: 1px solid {PALETTE.border};
    font-weight: 500;
}}
QTableView::item {{
    padding: 6px 10px;
}}
"""
