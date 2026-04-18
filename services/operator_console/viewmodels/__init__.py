"""Operator Console viewmodels — Phase 7/8.

Viewmodels subscribe to `OperatorStore` signals, expose read-only
accessors to their paired view, and emit lightweight change notifications.
They hold no network code and no widget references.

Spec references:
  §4.E.1         — operator-facing page surfaces
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from services.operator_console.viewmodels.base import ViewModelBase

__all__ = ["ViewModelBase"]
