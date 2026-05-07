"""Posterior α/β bar delegate for the Experiments arms table.

Renders the Thompson Sampling posterior as a 120px horizontal bar — the
green segment proportional to α/(α+β), the red segment for β — so the
operator compares arms preattentively rather than doing the arithmetic
in their head. The raw α and β values stay accessible via tooltip and
through the table model's own tooltip role.

The delegate paints a single column. Its data contract is two consecutive
floats (α, β) addressed via the `Qt.ItemDataRole.UserRole` payload set by
the experiments table model, so the delegate stays decoupled from arm
DTO internals.

Spec references:
  §7B            — Thompson Sampling posterior (α, β)
  §4.E.1         — Experiments operator surface
"""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from services.operator_console.design_system.tokens import PALETTE


class PosteriorBarDelegate(QStyledItemDelegate):
    """Paint α and β as a side-by-side bar instead of two integer cells."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bar_width = 120
        self._bar_height = 8

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        payload = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(payload, tuple) or len(payload) != 2:
            super().paint(painter, option, index)
            return
        alpha = _safe_float(payload[0])
        beta = _safe_float(payload[1])
        total = alpha + beta
        if total <= 0:
            super().paint(painter, option, index)
            return

        widget_option = QStyleOptionViewItem(option)
        self.initStyleOption(widget_option, cast(QModelIndex, index))
        # Paint the row background ourselves first so selection state
        # still highlights the cell.
        painter.save()
        painter.fillRect(widget_option.rect, widget_option.backgroundBrush)
        if widget_option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(widget_option.rect, widget_option.palette.highlight())

        rect = widget_option.rect
        bar_width = min(self._bar_width, max(40, rect.width() - 16))
        bar_x = rect.x() + 8
        bar_y = rect.y() + max(0, (rect.height() - self._bar_height) // 2)

        track_rect = QRect(bar_x, bar_y, bar_width, self._bar_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(PALETTE.surface_raised))
        painter.drawRect(track_rect)

        alpha_ratio = alpha / total if total else 0.0
        alpha_pixels = int(round(bar_width * alpha_ratio))
        if alpha_pixels > 0:
            painter.setBrush(QColor(PALETTE.status_ok))
            painter.drawRect(QRect(bar_x, bar_y, alpha_pixels, self._bar_height))
        beta_pixels = bar_width - alpha_pixels
        if beta_pixels > 0:
            painter.setBrush(QColor(PALETTE.status_bad))
            painter.drawRect(QRect(bar_x + alpha_pixels, bar_y, beta_pixels, self._bar_height))

        # Compact ratio label after the bar so the operator still sees a
        # number when the bar is hard to size with the eye.
        label = f"{int(round(alpha_ratio * 100))}% α"
        text_rect = QRect(
            bar_x + bar_width + 8,
            rect.y(),
            rect.right() - (bar_x + bar_width + 8),
            rect.height(),
        )
        painter.setPen(QColor(PALETTE.text_muted))
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            label,
        )
        painter.restore()


def _safe_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
