"""Qt item models for the Operator Console tabular surfaces — Phase 7.

Each model is a thin `QAbstractTableModel` subclass bound to a specific
Phase-1 DTO list. Models hold no business logic and no network code —
they are purely the Qt-side adaptor between store state and a view's
`QTableView`. Formatting for display cells flows through
`services.operator_console.formatters` so the operator vocabulary stays
consistent across tables and cards.

Spec references:
  §4.E.1         — operator-facing multi-page layout
  §7B            — Thompson Sampling reward = p90_intensity × semantic_gate
  §7C            — Co-Modulation Index null-valid
  §12            — subsystem recovery-mode vs error distinction
  SPEC-AMEND-008 — PySide6 Operator Console
"""

from __future__ import annotations

from services.operator_console.table_models.alerts_table_model import AlertsTableModel
from services.operator_console.table_models.encounters_table_model import (
    EncountersTableModel,
)
from services.operator_console.table_models.experiments_table_model import (
    ExperimentsTableModel,
)
from services.operator_console.table_models.health_table_model import HealthTableModel
from services.operator_console.table_models.sessions_table_model import (
    SessionsTableModel,
)

__all__ = [
    "AlertsTableModel",
    "EncountersTableModel",
    "ExperimentsTableModel",
    "HealthTableModel",
    "SessionsTableModel",
]
