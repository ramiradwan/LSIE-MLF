"""Dark-neutral QSS stylesheet for the Operator Console.

The theme covers every widget primitive used by the Operator Console
(`MetricCard`, `ActionBar`, `AlertBanner`, `EmptyState`, `SectionHeader`,
`EventTimeline`) plus the health-row
state classes (`ok` / `warn` / `bad` / `degraded` / `recovering`) that
the Health view needs to distinguish degraded-but-recovering subsystems
from hard errors (§12 error-handling matrix).

`build_stylesheet()` is the sanctioned entry point — `app.py` calls it
after the `QApplication` is instantiated. The module also re-exports
`STYLESHEET` as the eagerly-composed string for backwards compatibility
with any caller that imports the constant directly.

Styling strategy:
  * All component-specific rules key off stable object names set by
    widget classes (`#MetricCard`, `#ActionBar`, `#AlertBannerInfo`,
    ...). Widget code never calls `setStyleSheet` inline.
  * `#HealthRowRecovering` and `#HealthRowDegraded` are deliberately
    coloured between `ok` and `bad` so an operator sees "this is
    recovering on its own" as a distinct signal from "this is broken".

Spec references:
  §4.E.1         — operator-facing execution details
  §12            — error-handling matrix; recovery vs error colouring
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
    # §12: a subsystem in `recovery_mode=True` is distinct from a hard
    # error; it should read as "working on it" rather than "broken".
    status_recovering: str = "#6cc3d5"
    status_degraded: str = "#d59b4a"


PALETTE = Palette()


def build_stylesheet() -> str:
    """Compose the full operator-console QSS string.

    Factored out so `app.py` / tests can call it explicitly, and so the
    palette can be swapped for a light theme in the future without
    mutating a module-level constant.
    """
    p = PALETTE
    return f"""
* {{
    font-family: "Segoe UI", "Inter", system-ui, sans-serif;
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

/* ---- MetricCard --------------------------------------- */

QFrame#MetricCard {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 6px;
}}
QLabel#MetricCardTitle {{
    color: {p.text_muted};
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QLabel#MetricCardPrimary {{
    color: {p.text_primary};
    font-size: 18px;
    font-weight: 600;
}}
QLabel#MetricCardSecondary {{
    color: {p.text_muted};
    font-size: 12px;
}}

/* ---- SectionHeader ------------------------------------ */

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

/* ---- ActionBar ---------------------------------------- */

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
    color: #ffffff;
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
    /* QSS does not implement `font-variant-numeric`; pin a monospace
       family so the digits stay column-aligned as they tick down. */
    font-family: "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", monospace;
}}
QLabel#ActionBarMessage {{
    color: {p.text_muted};
    font-size: 12px;
}}

/* ---- AlertBanner -------------------------------------- */

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

/* ---- EmptyState --------------------------------------- */

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

/* ---- EventTimeline ------------------------------------ */

QWidget#EventTimeline {{
    background: transparent;
}}
QTableView#EventTimelineTable {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
}}

/* ---- Tables (shared) -------------------------------------------- */

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

/* ---- Health-row state classes (§12) ----------------------------- */

/* Used by HealthTableModel via
 * `model.setData(index, "HealthRowOk", ForegroundRole)` proxy — the
 * visual intent is documented here so the Health view can pick the
 * right styling hook without re-deriving §12 colour semantics. */

QLabel#HealthRowOk        {{ color: {p.status_ok}; font-weight: 600; }}
QLabel#HealthRowWarn      {{ color: {p.status_warn}; font-weight: 600; }}
QLabel#HealthRowBad       {{ color: {p.status_bad}; font-weight: 600; }}
QLabel#HealthRowDegraded  {{ color: {p.status_degraded}; font-weight: 600; }}
QLabel#HealthRowRecovering {{ color: {p.status_recovering}; font-weight: 600; }}
"""


# Back-compat: existing imports of the raw constant continue to work.
STYLESHEET = build_stylesheet()
