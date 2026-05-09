"""Build and install the Operator Console stylesheet from design tokens.

Qt's QSS does not support `font-feature-settings`; tabular numerals are
applied at the `QFont` level via `QFont.setFeature` and propagated to
every widget through `QApplication.setFont`. Numbers stop dancing when
they tick without flooding the runtime with `Unknown property`
warnings.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

from services.operator_console.design_system.tokens import PALETTE, Palette


def repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _apply_tabular_features(font: QFont) -> QFont:
    """Enable `tnum` + `lnum` OpenType features on *font* when available.

    `QFont.setFeature` shipped in Qt 6.8 (PySide6 6.8); on older
    runtimes the call is silently skipped. Either way the resulting
    font is safe to push through `QApplication.setFont`.
    """

    setter = getattr(font, "setFeature", None)
    if setter is None:
        return font
    tag_cls = getattr(QFont, "Tag", None)
    for tag_name in ("tnum", "lnum"):
        try:
            if tag_cls is not None:
                setter(tag_cls(tag_name), 1)
            else:
                setter(tag_name, 1)
        except (TypeError, ValueError):
            # Different PySide6 minor versions accept different shapes;
            # fall through and try the next tag rather than crash.
            continue
    return font


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

/* Container surfaces inherit the page background by default. Without
 * these rules every plain QWidget/QScrollArea/viewport falls back to
 * the system style (light grey on Windows), which leaks through as
 * white bands between cards and behind SectionHeader widgets. More
 * specific rules below (QFrame#Panel, QFrame#MetricCard, etc.) keep
 * their own surface colours by selector specificity. */
QScrollArea {{
    background: {p.background};
    border: none;
}}
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {{
    background: {p.background};
}}
QStackedWidget,
QStackedWidget > QWidget {{
    background: {p.background};
}}
QSplitter::handle {{
    background: {p.border};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}

QScrollBar:vertical {{
    background: {p.background};
    width: 10px;
    margin: 2px 2px 2px 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {p.border};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {p.text_muted};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    background: transparent;
    height: 0px;
    border: none;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: {p.background};
    height: 10px;
    margin: 0 2px 2px 2px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {p.border};
    border-radius: 3px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {p.text_muted};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    background: transparent;
    width: 0px;
    border: none;
}}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

QFrame#SidebarNav {{
    background: {p.surface};
    border-right: 1px solid {p.border};
}}

QPushButton {{
    background: {p.surface_raised};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {p.accent};
    color: {p.text_primary};
}}
QPushButton:disabled {{
    color: {p.text_muted};
    background: {p.surface};
    border-color: {p.border};
}}
QPushButton:focus {{
    outline: none;
    border: 1px solid {p.accent};
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
QPushButton#NavButton:focus {{
    outline: none;
    border-left: 2px solid {p.accent};
    background: {p.surface_raised};
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
QFrame#MetricCard[clickable=\"true\"]:hover {{
    border-color: {p.accent};
    background: {p.surface_raised};
}}
QFrame#MetricCard[clickable=\"true\"]:focus {{
    outline: none;
    border: 2px solid {p.accent};
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
QLabel#MetricCardChevron {{
    color: {p.text_muted};
    font-size: 14px;
    font-weight: 600;
    padding: 0 0 0 6px;
}}
QFrame#MetricCard[clickable=\"true\"]:hover QLabel#MetricCardChevron {{
    color: {p.accent};
}}

QWidget#SectionHeader {{
    background: transparent;
}}
QLabel#SectionHeaderTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {p.text_primary};
}}
QLabel#SectionHeaderTitle[level=\"page\"] {{
    font-size: 18px;
    font-weight: 600;
}}
QLabel#SectionHeaderTitle[level=\"panel\"] {{
    font-size: 14px;
    font-weight: 600;
}}
QLabel#SectionHeaderTitle[level=\"sub\"] {{
    font-size: 12px;
    font-weight: 500;
    color: {p.text_muted};
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
    outline: none;
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
QPushButton#ActionBarSubmit:focus {{
    outline: none;
    border: 2px solid {p.accent};
    padding: 6px 16px;
}}
QLabel#ActionBarCountdown {{
    color: {p.status_warn};
    font-family: \"Cascadia Mono\", \"Consolas\", \"Menlo\", \"DejaVu Sans Mono\", monospace;
}}
QLabel#ActionBarMessage {{
    color: {p.text_muted};
    font-size: 12px;
}}
QLabel#ActionBarMessage[severity=\"blocked\"] {{
    color: {p.status_warn};
}}
QPushButton#ActionBarNoteToggle {{
    background: transparent;
    border: 1px dashed {p.border};
    border-radius: 4px;
    padding: 4px 10px;
    color: {p.text_muted};
    font-size: 12px;
}}
QPushButton#ActionBarNoteToggle:hover {{
    color: {p.text_primary};
    border-color: {p.accent};
}}
QFrame#ActionBarProgress {{
    background: transparent;
    border: none;
}}

QFrame#LiveSessionReadinessStrip {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
}}
QFrame#LiveSessionReadinessStrip[severity=\"warn\"] {{
    background: rgba(231, 179, 74, 0.12);
    border-color: {p.status_warn};
}}
QFrame#LiveSessionReadinessStrip[severity=\"error\"] {{
    background: rgba(226, 106, 106, 0.16);
    border-color: {p.status_bad};
}}
QFrame#LiveSessionReadinessStrip[severity=\"recovering\"] {{
    background: rgba(108, 195, 213, 0.10);
    border-color: {p.status_recovering};
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
    /* Selection uses surface_raised + accent left-border so it reads
     * as a deliberate row highlight instead of the saturated accent
     * fill that previously matched the cell-hover paint and made the
     * two states indistinguishable. */
    selection-background-color: {p.surface_raised};
    selection-color: {p.text_primary};
    outline: none;
}}
/* Outline-on-focus paints around the entire QAbstractScrollArea —
 * including scrollbars — so we drop the focus border entirely. The
 * selection highlight is enough to indicate the active row. */
QTableView:focus {{
    outline: none;
    border: none;
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
/* Subtle per-row hover that does not collide with selection. Qt's
 * default cell-hover would otherwise paint the same accent colour as
 * a click, so a hover read as "selected" until the cursor moved. */
QTableView::item:hover {{
    background: {p.border};
}}
QTableView::item:selected {{
    background: {p.surface_raised};
    color: {p.text_primary};
    border-left: 2px solid {p.accent};
}}
QTableView::item:selected:active {{
    background: {p.surface_raised};
    color: {p.text_primary};
}}

QLineEdit {{
    background: {p.surface_raised};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 6px 10px;
    color: {p.text_primary};
    selection-background-color: {p.accent};
}}
QLineEdit:focus {{
    border: 1px solid {p.accent};
    outline: none;
}}

QLineEdit#SessionsFilterInput {{
    background: {p.surface_raised};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 6px 10px;
}}
QLineEdit#SessionsFilterInput:focus {{
    border: 1px solid {p.accent};
}}

QFrame#ExperimentsSlideOver {{
    background: {p.surface};
    border-left: 1px solid {p.border};
}}
QFrame#ExperimentsSlideOverScrim {{
    background: rgba(0, 0, 0, 0.45);
}}

QFrame#ProbeSparkline {{
    background: transparent;
    border: none;
}}

QLabel#HealthRowOk {{ color: {p.status_ok}; font-weight: 600; }}
QLabel#HealthRowWarn {{ color: {p.status_warn}; font-weight: 600; }}
QLabel#HealthRowBad {{ color: {p.status_bad}; font-weight: 600; }}
QLabel#HealthRowDegraded {{ color: {p.status_degraded}; font-weight: 600; }}
QLabel#HealthRowRecovering {{ color: {p.status_recovering}; font-weight: 600; }}
QLabel#HealthRowMuted {{ color: {p.text_muted}; font-weight: 600; }}

QLabel#EncounterVerdictHeadline {{
    color: {p.text_primary};
    font-size: 16px;
    font-weight: 600;
}}
QLabel#EncounterVerdictReason {{
    color: {p.text_muted};
    font-size: 13px;
}}
QLabel#EncounterInputsLabel {{
    color: {p.text_muted};
    font-family: \"Cascadia Mono\", \"Consolas\", \"Menlo\", \"DejaVu Sans Mono\", monospace;
    font-size: 12px;
    padding-left: 16px;
}}

QLabel#NullValidPill {{
    background: rgba(91, 141, 239, 0.18);
    color: {p.accent};
    border: 1px solid {p.accent};
    border-radius: 999px;
    padding: 1px 10px;
    font-size: 11px;
    font-weight: 600;
}}
"""


STYLESHEET = build_stylesheet()


def install_application_stylesheet(
    app: QApplication,
    palette: Palette = PALETTE,
) -> None:
    app.setStyleSheet(build_stylesheet(palette))
    app.setFont(_apply_tabular_features(app.font()))
