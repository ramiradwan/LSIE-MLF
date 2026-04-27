"""Stimulus action rail — session-context header + submit button.

The single write path in the Operator Console: the operator issues a
stimulus for the currently-selected session, and the bar's visual state
walks through idle → submitting → accepted → measuring → completed.
The *authoritative* `stimulus_time` is set by the orchestrator (§4.C);
this widget only reflects the action lifecycle and never talks to the
network directly.

Spec references:
  §4.C           — orchestrator-owned `_active_arm`, `_expected_greeting`,
                   authoritative `_stimulus_time`
  §4.E.1         — operator-facing action rail
  §7B            — measurement-window countdown applies once accepted
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from packages.schemas.operator_console import StimulusActionState, UiStatusKind
from services.operator_console.state import StimulusUiContext
from services.operator_console.widgets.status_pill import StatusPill

# Map lifecycle states to button label + enabled flag + pill kind.
# This stays inside the widget (not in `formatters.py`) because the
# mapping is UI-semantic, not operator-language; formatters are about
# translating DTO values, not widget internal states.
_BUTTON_LABEL: dict[StimulusActionState, str] = {
    StimulusActionState.IDLE: "Send stimulus",
    StimulusActionState.SUBMITTING: "Sending…",
    StimulusActionState.ACCEPTED: "Accepted",
    StimulusActionState.MEASURING: "Measuring…",
    StimulusActionState.COMPLETED: "Send stimulus",
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

        self._session_label = QLabel("No session selected", self)
        self._session_label.setObjectName("ActionBarSession")
        self._greeting_label = QLabel("", self)
        self._greeting_label.setObjectName("ActionBarGreeting")
        self._greeting_label.setWordWrap(True)

        self._note_input = QLineEdit(self)
        self._note_input.setObjectName("ActionBarNote")
        self._note_input.setPlaceholderText("Optional operator note…")
        # §4.C stimulus may be dispatched from keyboard: return triggers submit.
        self._note_input.returnPressed.connect(self._on_submit_clicked)

        self._submit_button = QPushButton(
            _BUTTON_LABEL[StimulusActionState.IDLE],
            self,
        )
        self._submit_button.setObjectName("ActionBarSubmit")
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

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self._session_label)
        top.addStretch(1)
        top.addWidget(self._state_pill)
        top.addWidget(self._countdown_label)

        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(12)
        mid.addWidget(self._note_input, 1)
        mid.addWidget(self._submit_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(6)
        root.addLayout(top)
        root.addWidget(self._greeting_label)
        root.addLayout(mid)
        root.addWidget(self._message_label)

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

        arm_part = active_arm if active_arm else "no active arm"
        self._session_label.setText(f"Session {session_id} — arm: {arm_part}")
        if expected_greeting:
            self._greeting_label.setText(f"Expected greeting: “{expected_greeting}”")
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
        # Clear any in-progress countdown when state leaves MEASURING.
        if state != StimulusActionState.MEASURING:
            self.set_countdown_remaining(None)
        if ctx.message is not None:
            self.set_last_message(ctx.message)

    def set_countdown_remaining(self, seconds: float | None) -> None:
        """Show a `00:12` countdown while the §7B measurement window runs.

        The orchestrator owns the authoritative `stimulus_time` that
        anchors the window; Phase 8's viewmodel feeds that seconds-
        remaining value in. We only format + show.
        """
        if seconds is None or seconds <= 0:
            self._countdown_label.setText("")
            self._countdown_label.setVisible(False)
            return
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        self._countdown_label.setText(f"{minutes:02d}:{secs:02d} left")
        self._countdown_label.setVisible(True)

    def set_last_message(self, text: str | None) -> None:
        if text:
            self._message_label.setText(text)
            self._message_label.setVisible(True)
        else:
            self._message_label.setText("")
            self._message_label.setVisible(False)

    # ---- internals -----------------------------------------------------

    def _on_submit_clicked(self) -> None:
        if not self._submit_button.isEnabled() or self._session_id is None:
            return
        note = self._note_input.text().strip()
        self.stimulus_requested.emit(note)

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

    # ---- keyboard shortcut hook ---------------------------------------

    def focus_note_input(self) -> None:
        """Give keyboard focus to the operator-note input.

        Phase-6 main window wires a global shortcut (e.g. Ctrl+Enter)
        to this method so the operator can stimulate without leaving
        the keyboard.
        """
        self._note_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
