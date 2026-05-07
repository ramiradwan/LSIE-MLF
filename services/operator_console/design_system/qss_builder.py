"""Build and install the Operator Console stylesheet from design tokens."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

from services.operator_console.design_system.tokens import PALETTE, Palette


def repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def build_stylesheet(palette: Palette = PALETTE) -> str:
    p = palette
    return f"""
* {{
    font-family: \"Segoe UI\", \"Inter\", system-ui, sans-serif;
    font-size: 13px;
    color: {p.text_primary};
}}

QMainWindow, QDialog {{
    background: {p.background};
}}

QWidget#ContentSurface {{
    background: {p.background};
}}

QFrame#SidebarNav {{
    background: {p.surface};
    border-right: 1px solid {p.border};
}}

QLabel#SidebarTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {p.text_primary};
    padding: 0 6px 4px 6px;
}}

QLabel#SidebarSubtitle {{
    color: {p.text_muted};
    padding: 0 6px 18px 6px;
}}

QPushButton#NavButton {{
    background: transparent;
    border: none;
    text-align: left;
    padding: 10px 16px;
    color: {p.text_muted};
    border-left: 2px solid transparent;
}}
QPushButton#NavButton:hover {{
    color: {p.text_primary};
    background: {p.surface_raised};
}}
QPushButton#NavButton:checked {{
    color: {p.text_primary};
    background: {p.surface_raised};
    border-left: 2px solid {p.accent};
}}

QFrame#Panel {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 6px;
}}

QLabel#PanelTitle {{
    font-size: 14px;
    font-weight: 600;
    color: {p.text_primary};
    padding: 12px 16px 6px 16px;
}}

QLabel#PanelSubtitle {{
    color: {p.text_muted};
    padding: 0 16px 10px 16px;
}}

QLabel#StatusBarLabel {{
    color: {p.text_muted};
}}

QWidget#StatusPill {{
    background: transparent;
}}

QLabel#StatusPillLabel {{
    color: {p.text_primary};
    font-size: 12px;
    font-weight: 600;
}}

QFrame#MetricCard {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 6px;
}}
QLabel#MetricCardTitle {{
    color: {p.text_muted};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#MetricCardPrimary {{
    color: {p.text_primary};
    font-size: 16px;
    font-weight: 600;
}}
QLabel#MetricCardSecondary {{
    color: {p.text_muted};
    font-size: 12px;
}}

QWidget#SectionHeader {{
    background: transparent;
}}
QLabel#SectionHeaderTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {p.text_primary};
}}
QLabel#SectionHeaderSubtitle {{
    color: {p.text_muted};
    font-size: 12px;
}}

QWidget#ActionBar {{
    background: {p.surface};
    border-top: 1px solid {p.border};
}}
QLabel#ActionBarSession {{
    color: {p.text_primary};
    font-weight: 600;
}}
QLabel#ActionBarGreeting {{
    color: {p.text_muted};
    font-style: italic;
}}
QLineEdit#ActionBarNote {{
    background: {p.surface_raised};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 6px 10px;
    color: {p.text_primary};
}}
QLineEdit#ActionBarNote:focus {{
    border: 1px solid {p.accent};
}}
QPushButton#ActionBarSubmit {{
    background: {p.accent};
    color: {p.text_inverse};
    border: none;
    border-radius: 4px;
    padding: 8px 18px;
    font-weight: 600;
}}
QPushButton#ActionBarSubmit:disabled {{
    background: {p.surface_raised};
    color: {p.text_muted};
}}
QLabel#ActionBarCountdown {{
    color: {p.status_warn};
    font-family: \"Cascadia Mono\", \"Consolas\", \"Menlo\", \"DejaVu Sans Mono\", monospace;
}}
QLabel#ActionBarMessage {{
    color: {p.text_muted};
    font-size: 12px;
}}

QFrame#AlertBanner,
QFrame#AlertBannerInfo,
QFrame#AlertBannerWarning,
QFrame#AlertBannerCritical {{
    border-radius: 4px;
    border: 1px solid {p.border};
}}
QFrame#AlertBannerInfo {{
    background: rgba(91, 141, 239, 0.12);
    border-color: {p.accent};
}}
QFrame#AlertBannerWarning {{
    background: rgba(231, 179, 74, 0.14);
    border-color: {p.status_warn};
}}
QFrame#AlertBannerCritical {{
    background: rgba(226, 106, 106, 0.16);
    border-color: {p.status_bad};
}}
QLabel#AlertBannerGlyph {{
    font-size: 16px;
    font-weight: 700;
    color: {p.text_primary};
}}
QLabel#AlertBannerMessage {{
    color: {p.text_primary};
}}

QWidget#EmptyState {{
    background: transparent;
}}
QLabel#EmptyStateTitle {{
    color: {p.text_primary};
    font-size: 15px;
    font-weight: 600;
}}
QLabel#EmptyStateMessage {{
    color: {p.text_muted};
}}

QWidget#EventTimeline {{
    background: transparent;
}}
QTableView#EventTimelineTable {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
}}

QTableView {{
    background: {p.surface};
    alternate-background-color: {p.surface_raised};
    gridline-color: {p.border};
    border: none;
    selection-background-color: {p.accent};
    selection-color: {p.text_primary};
}}
QHeaderView::section {{
    background: {p.surface_raised};
    color: {p.text_muted};
    padding: 6px 10px;
    border: none;
    border-bottom: 1px solid {p.border};
    font-weight: 500;
}}
QTableView::item {{
    padding: 6px 10px;
}}

QLabel#HealthRowOk {{ color: {p.status_ok}; font-weight: 600; }}
QLabel#HealthRowWarn {{ color: {p.status_warn}; font-weight: 600; }}
QLabel#HealthRowBad {{ color: {p.status_bad}; font-weight: 600; }}
QLabel#HealthRowDegraded {{ color: {p.status_degraded}; font-weight: 600; }}
QLabel#HealthRowRecovering {{ color: {p.status_recovering}; font-weight: 600; }}
"""


STYLESHEET = build_stylesheet()


def install_application_stylesheet(
    app: QApplication,
    palette: Palette = PALETTE,
) -> None:
    app.setStyleSheet(build_stylesheet(palette))
