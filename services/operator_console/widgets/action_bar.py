"""Stimulus action rail — session-context header + submit button.

The single write path in the Operator Console: the operator issues a
stimulus for the currently-selected session, and the bar's visual state
walks through idle → submitting → accepted → measuring → completed.
The *authoritative* `stimulus_time` is set by the orchestrator (§4.C);
this widget only reflects the action lifecycle and never talks to the
network directly.

Three persistent visual signals make the bar safe to operate from:

* the submit button never silently disables — the message line always
  states the gating reason from `formatters.py` so the operator never
  clicks into nothing
* a 2px progress strip pinned to the bottom edge of the bar drains over
  the §7B response window so time pressure is preattentive without
  re-reading digits
* in compact mode (<1024px) the countdown and message stay visible on a
  second row instead of disappearing; the note field collapses behind a
  `+ note` chip that expands inline when the operator needs it

Spec references:
  §4.C           — orchestrator-owned `_active_arm`, `_expected_greeting`,
                   authoritative `_stimulus_time`
  §4.E.1         — operator-facing action rail
  §7B            — measurement-window countdown applies once accepted
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import StimulusActionState, UiStatusKind
from services.operator_console.design_system.qss_builder import repolish
from services.operator_console.design_system.tokens import PALETTE
from services.operator_console.formatters import (
    format_action_bar_session_context,
    format_session_id_compact,
)
from services.operator_console.state import StimulusUiContext
from services.operator_console.widgets.status_pill import StatusPill

# Map lifecycle states to button label + enabled flag + pill kind.
# This stays inside the widget (not in `formatters.py`) because the
# mapping is UI-semantic, not operator-language; formatters are about
# translating DTO values, not widget internal states.
_BUTTON_LABEL: dict[StimulusActionState, str] = {
    StimulusActionState.IDLE: "Send Stimulus",
    StimulusActionState.SUBMITTING: "Sending…",
    StimulusActionState.ACCEPTED: "Accepted",
    StimulusActionState.MEASURING: "Measuring…",
    StimulusActionState.COMPLETED: "Send Stimulus",
    StimulusActionState.FAILED: "Retry",
}

# §4.C: while the action is in flight we must not allow a second submit,
# both for UX and because `client_action_id` dedup is per-submission.
_BUTTON_ENABLED: dict[StimulusActionState, bool] = {
    StimulusActionState.IDLE: True,
    StimulusActionState.SUBMITTING: False,
    StimulusActionState.ACCEPTED: False,
    StimulusActionState.MEASURING: False,
    StimulusActionState.COMPLETED: True,
    StimulusActionState.FAILED: True,
}

_STATE_PILL: dict[StimulusActionState, tuple[UiStatusKind, str]] = {
    StimulusActionState.IDLE: (UiStatusKind.NEUTRAL, "ready"),
    StimulusActionState.SUBMITTING: (UiStatusKind.PROGRESS, "submitting"),
    StimulusActionState.ACCEPTED: (UiStatusKind.INFO, "accepted"),
    StimulusActionState.MEASURING: (UiStatusKind.PROGRESS, "measuring"),
    StimulusActionState.COMPLETED: (UiStatusKind.OK, "completed"),
    StimulusActionState.FAILED: (UiStatusKind.ERROR, "failed"),
}


class _ProgressStrip(QFrame):
    """2px progress bar pinned to the bottom edge of the ActionBar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActionBarProgress")
        self.setFixedHeight(2)
        self._ratio = 0.0
        self._color = QColor(PALETTE.status_warn)
        self.setVisible(False)

    def set_ratio(self, ratio: float) -> None:
        clamped = max(0.0, min(1.0, ratio))
        if abs(clamped - self._ratio) < 1e-4:
            return
        self._ratio = clamped
        self.update()

    def set_color(self, color: QColor) -> None:
        if color == self._color:
            return
        self._color = color
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        del event
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        width = int(self.width() * self._ratio)
        painter.drawRect(0, 0, width, self.height())


class ActionBar(QWidget):
    """Persistent stimulus action rail below the content area."""

    # Payload is the operator note the user typed (possibly empty).
    stimulus_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActionBar")

        self._session_id: UUID | None = None
        self._active_arm: str | None = None
        self._expected_greeting: str | None = None
        self._operator_ready_for_submit: bool = True
        self._compact_mode = False
        self._gating_reason: str | None = None
        self._note_expanded = False
        self._countdown_total_s: float | None = None
        self._countdown_remaining_s: float | None = None

        self._session_label = QLabel("No session selected", self)
        self._session_label.setObjectName("ActionBarSession")
        self._session_label.setWordWrap(True)
        self._greeting_label = QLabel("", self)
        self._greeting_label.setObjectName("ActionBarGreeting")
        self._greeting_label.setWordWrap(True)

        self._note_input = QLineEdit(self)
        self._note_input.setObjectName("ActionBarNote")
        self._note_input.setAccessibleName("Operator note")
        self._note_input.setAccessibleDescription("Optional note sent with the stimulus.")
        self._note_input.setPlaceholderText("Optional operator note…")
        # §4.C stimulus may be dispatched from keyboard: return triggers submit.
        self._note_input.returnPressed.connect(self._on_submit_clicked)

        self._note_toggle = QPushButton("+ note", self)
        self._note_toggle.setObjectName("ActionBarNoteToggle")
        self._note_toggle.setAccessibleName("Toggle operator note")
        self._note_toggle.setAccessibleDescription("Show or hide the operator note field.")
        self._note_toggle.setFlat(True)
        self._note_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._note_toggle.clicked.connect(self._on_note_toggle_clicked)
        self._note_toggle.setVisible(False)

        self._submit_button = QPushButton(
            _BUTTON_LABEL[StimulusActionState.IDLE],
            self,
        )
        self._submit_button.setObjectName("ActionBarSubmit")
        self._submit_button.setAccessibleName("Send stimulus")
        self._submit_button.setAccessibleDescription(
            "Sends the current stimulus when the session is ready."
        )
        self._submit_button.setEnabled(False)  # no session yet
        self._submit_button.clicked.connect(self._on_submit_clicked)

        self._state_pill = StatusPill(self)
        self._state_pill.set_kind(UiStatusKind.NEUTRAL)
        self._state_pill.set_text("idle")

        self._countdown_label = QLabel("", self)
        self._countdown_label.setObjectName("ActionBarCountdown")
        self._countdown_label.setVisible(False)

        self._message_label = QLabel("", self)
        self._message_label.setObjectName("ActionBarMessage")
        self._message_label.setWordWrap(True)
        self._message_label.setVisible(False)
        self._message_label.setProperty("severity", "")

        self._progress_strip = _ProgressStrip(self)

        self._header_layout = QHBoxLayout()
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.setSpacing(12)

        self._status_layout = QHBoxLayout()
        self._status_layout.setContentsMargins(0, 0, 0, 0)
        self._status_layout.setSpacing(12)

        self._input_layout = QHBoxLayout()
        self._input_layout.setContentsMargins(0, 0, 0, 0)
        self._input_layout.setSpacing(12)

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(16, 12, 16, 12)
        self._content_layout.setSpacing(6)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)
        self._root_layout.addLayout(self._content_layout, 1)
        self._root_layout.addWidget(self._progress_strip)

        self._rebuild_layout()

    # ---- external API --------------------------------------------------

    def set_session_context(
        self,
        session_id: UUID | None,
        active_arm: str | None,
        expected_greeting: str | None,
        operator_ready_for_submit: bool | None = None,
    ) -> None:
        """Update the session header + enable/disable submit.

        The button stays disabled when no session is selected so the
        operator cannot fire a stimulus at nothing. When a session is
        selected, submit gating uses the console's derived readiness
        (safe-submit frame threshold), not the worker lifecycle flag.
        ``None`` preserves the legacy ready behavior.
        """
        self._session_id = session_id
        self._active_arm = active_arm
        self._expected_greeting = expected_greeting
        self._operator_ready_for_submit = operator_ready_for_submit is not False

        if session_id is None:
            self._session_label.setText("No session selected")
            self._greeting_label.setText("")
            self._greeting_label.setVisible(False)
            self._submit_button.setEnabled(False)
            return

        session_display = format_action_bar_session_context(
            session_id,
            active_arm,
            expected_greeting,
        )
        self._session_label.setText(self._session_header_text(session_display.session_text))
        if session_display.expected_response_text is not None:
            self._greeting_label.setText(session_display.expected_response_text)
            self._greeting_label.setVisible(True)
        else:
            self._greeting_label.setText("")
            self._greeting_label.setVisible(False)
        # Restore enabled state based on current action state — do not
        # auto-reset to IDLE here, a re-select mid-flight should not
        # unblock a second submit.
        self._submit_button.setEnabled(self._submit_enabled())

    def set_action_state(self, ctx: StimulusUiContext) -> None:
        """Walk the button + pill through the lifecycle state in `ctx`."""
        self._last_ctx = ctx
        state = ctx.state
        self._submit_button.setText(_BUTTON_LABEL[state])
        self._submit_button.setEnabled(self._submit_enabled())
        kind, label = _STATE_PILL[state]
        self._state_pill.set_kind(kind)
        self._state_pill.set_text(label)
        if state is StimulusActionState.MEASURING:
            self._progress_strip.set_color(QColor(PALETTE.status_warn))
        elif state is StimulusActionState.COMPLETED:
            self._progress_strip.set_color(QColor(PALETTE.status_ok))
        elif state is StimulusActionState.FAILED:
            self._progress_strip.set_color(QColor(PALETTE.status_bad))
        else:
            self._progress_strip.set_color(QColor(PALETTE.status_warn))
        # Clear any in-progress countdown when state leaves MEASURING.
        if state != StimulusActionState.MEASURING:
            self.set_countdown_remaining(None)
        if ctx.message is not None:
            self.set_last_message(ctx.message)

    def set_countdown_remaining(
        self,
        seconds: float | None,
        *,
        total_seconds: float | None = None,
    ) -> None:
        """Show a `00:12` countdown while the §7B measurement window runs.

        The orchestrator owns the authoritative `stimulus_time` that
        anchors the window; the viewmodel feeds that seconds-remaining
        value in. We render it as both digits and a 2px progress strip
        pinned to the bottom edge so the operator can pick up time
        pressure peripherally.
        """

        if seconds is None or seconds <= 0:
            self._countdown_remaining_s = None
            self._countdown_total_s = None
            self._countdown_label.setText("")
            self._countdown_label.setVisible(False)
            self._progress_strip.setVisible(False)
            self._progress_strip.set_ratio(0.0)
            return
        if total_seconds is not None and total_seconds > 0:
            self._countdown_total_s = total_seconds
        elif self._countdown_total_s is None or self._countdown_total_s < seconds:
            # Anchor the total to the first seen value so the strip drains
            # monotonically across the response window.
            self._countdown_total_s = seconds
        self._countdown_remaining_s = seconds
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        self._countdown_label.setText(f"{minutes:02d}:{secs:02d} left")
        self._countdown_label.setVisible(True)
        anchor = self._countdown_total_s or seconds
        self._progress_strip.setVisible(True)
        self._progress_strip.set_ratio(seconds / anchor if anchor > 0 else 0.0)

    def set_last_message(self, text: str | None) -> None:
        if text:
            self._message_label.setText(text)
            self._message_label.setVisible(True)
        else:
            self._message_label.setText("")
            self._message_label.setVisible(False)

    def set_gating_reason(self, reason: str | None) -> None:
        """Surface the reason the submit button is disabled.

        Disabled state has to read as deliberate, not mysterious — when
        the bar is gated (no session, calibration not complete, lifecycle
        in flight) the message line states why so the operator does not
        keep clicking into nothing.
        """

        self._gating_reason = reason
        if reason and not self._submit_enabled():
            self._message_label.setText(reason)
            self._message_label.setProperty("severity", "blocked")
            self._message_label.setVisible(True)
            repolish(self._message_label)
            return
        # When unblocked, fall back to whatever progress message was set
        # most recently. No message → hide.
        if not self._message_label.text():
            self._message_label.setVisible(False)
        self._message_label.setProperty("severity", "")
        repolish(self._message_label)

    def set_compact_mode(self, compact: bool) -> None:
        if self._compact_mode == compact:
            return
        self._compact_mode = compact
        if not compact:
            # Wide layout always shows the full note input.
            self._note_expanded = False
        if self._session_id is None:
            self._session_label.setText("No session selected")
        else:
            display = format_action_bar_session_context(
                self._session_id,
                self._active_arm,
                self._expected_greeting,
            )
            self._session_label.setText(self._session_header_text(display.session_text))
        self._rebuild_layout()

    # ---- internals -----------------------------------------------------

    def _session_header_text(self, wide_text: str) -> str:
        if not self._compact_mode or self._session_id is None:
            return wide_text
        strategy_part = self._active_arm if self._active_arm else "no active strategy"
        compact_session = format_session_id_compact(self._session_id)
        return f"Session {compact_session} · stimulus strategy {strategy_part}"

    def _rebuild_layout(self) -> None:
        self._clear_layout(self._content_layout)
        self._clear_layout(self._header_layout)
        self._clear_layout(self._status_layout)
        self._clear_layout(self._input_layout)

        if self._compact_mode:
            self._note_input.setVisible(self._note_expanded)
            self._note_toggle.setVisible(True)
            self._note_toggle.setText("− note" if self._note_expanded else "+ note")

            # Row 1: session label · pill · submit · note toggle
            self._header_layout.addWidget(self._session_label, 1)
            self._header_layout.addWidget(self._state_pill)

            self._input_layout.addWidget(self._submit_button)
            self._input_layout.addWidget(self._note_toggle)
            self._input_layout.addStretch(1)

            row1 = QHBoxLayout()
            row1.setContentsMargins(0, 0, 0, 0)
            row1.setSpacing(12)
            row1.addLayout(self._header_layout, 1)
            row1.addLayout(self._input_layout)

            # Row 2: countdown left, message right — only visible when
            # something is to show. Both labels manage their own visibility.
            self._status_layout.addWidget(self._countdown_label)
            self._status_layout.addStretch(1)
            self._status_layout.addWidget(self._message_label)

            self._content_layout.setContentsMargins(12, 10, 12, 10)
            self._content_layout.setSpacing(4)
            self._content_layout.addLayout(row1)
            self._content_layout.addLayout(self._status_layout)
            self._content_layout.addWidget(self._greeting_label)
            self._content_layout.addWidget(self._note_input)
            return

        # Wide layout — note always visible, classic two-row stack.
        self._note_input.setVisible(True)
        self._note_toggle.setVisible(False)

        self._status_layout.addWidget(self._state_pill)
        self._status_layout.addWidget(self._countdown_label)
        self._status_layout.addStretch(1)

        self._header_layout.addWidget(self._session_label)
        self._header_layout.addStretch(1)
        self._header_layout.addLayout(self._status_layout)

        self._input_layout.addWidget(self._note_input, 1)
        self._input_layout.addWidget(self._submit_button)

        self._content_layout.setContentsMargins(16, 12, 16, 12)
        self._content_layout.setSpacing(6)
        self._content_layout.addLayout(self._header_layout)
        self._content_layout.addWidget(self._greeting_label)
        self._content_layout.addLayout(self._input_layout)
        self._content_layout.addWidget(self._message_label)

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item is None:
                continue
            child_layout = item.layout()
            if isinstance(child_layout, QLayout):
                self._clear_layout(child_layout)

    def _on_submit_clicked(self) -> None:
        if not self._submit_button.isEnabled() or self._session_id is None:
            return
        note = self._note_input.text().strip()
        self.stimulus_requested.emit(note)

    def _on_note_toggle_clicked(self) -> None:
        if not self._compact_mode:
            return
        self._note_expanded = not self._note_expanded
        self._note_input.setVisible(self._note_expanded)
        self._note_toggle.setText("− note" if self._note_expanded else "+ note")
        if self._note_expanded:
            self._note_input.setFocus(Qt.FocusReason.OtherFocusReason)

    def _current_state(self) -> StimulusActionState:
        ctx = getattr(self, "_last_ctx", None)
        if ctx is None:
            return StimulusActionState.IDLE
        # `ctx` is typed as StimulusUiContext above; narrow for mypy.
        assert isinstance(ctx, StimulusUiContext)
        return ctx.state

    def _submit_enabled(self) -> bool:
        if self._session_id is None:
            return False
        if not self._operator_ready_for_submit:
            return False
        return _BUTTON_ENABLED[self._current_state()]

    # ---- keyboard focus hook -------------------------------------------

    def focus_note_input(self) -> None:
        """Give keyboard focus to the operator-note input."""
        if self._compact_mode and not self._note_expanded:
            self._note_expanded = True
            self._note_input.setVisible(True)
            self._note_toggle.setText("− note")
        self._note_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
