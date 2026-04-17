"""Sessions panel — live list of recent sessions sourced from the API.

This is the plumbing-test panel for the console: confirms that
``PollingWorker`` + ``ApiClient`` + the QTableView model chain render
correctly without freezing the UI when the API is slow or unreachable.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    QThread,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from services.operator_console.api_client import ApiClient, ApiError
from services.operator_console.config import ConsoleConfig
from services.operator_console.theme import PALETTE
from services.operator_console.widgets.status_pill import StatusPill
from services.operator_console.workers import PollingWorker

_Idx = QModelIndex | QPersistentModelIndex
# Qt's QAbstractItemModel API uses invalid-index sentinels for the root
# parent; we cache one at module load so ``rowCount`` / ``columnCount``
# don't construct fresh QModelIndex instances on every call (and so ruff
# B008 is satisfied).
_ROOT_INDEX: QModelIndex = QModelIndex()


class SessionsTableModel(QAbstractTableModel):
    _COLUMNS = ("session_id", "stream_url", "started_at", "ended_at", "metric_count")
    _HEADERS = ("Session", "Stream URL", "Started", "Ended", "Metrics")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: _Idx = _ROOT_INDEX) -> int:  # noqa: N802 — Qt override
        if isinstance(parent, QModelIndex) and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: _Idx = _ROOT_INDEX) -> int:  # noqa: N802 — Qt override
        return len(self._COLUMNS)

    def data(self, index: _Idx, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        key = self._COLUMNS[index.column()]
        value = row.get(key)
        if value is None:
            return "—"
        return str(value)

    def headerData(  # noqa: N802 — Qt override
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._HEADERS[section]
        return None


class SessionsView(QWidget):
    def __init__(self, config: ConsoleConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ContentSurface")
        self._config = config
        self._api = ApiClient(config.api_base_url, config.api_timeout_seconds)

        self._model = SessionsTableModel(self)
        self._status = StatusPill("Loading…", "idle", self)

        header = QLabel("Sessions", self)
        header.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {PALETTE.text_primary};")

        refresh_btn = QPushButton("Refresh", self)
        refresh_btn.clicked.connect(self._refresh_now)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(12)
        top_bar.addWidget(header)
        top_bar.addStretch(1)
        top_bar.addWidget(self._status)
        top_bar.addWidget(refresh_btn)

        table = QTableView(self)
        table.setModel(self._model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        horiz = table.horizontalHeader()
        horiz.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        horiz.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        horiz.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addLayout(top_bar)
        layout.addWidget(table)

        self._thread = QThread(self)
        self._worker = PollingWorker(
            job_name="scaffold_sessions",
            interval_ms=config.session_poll_interval_ms,
            fetch=self._api.list_sessions,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.data_ready.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _refresh_now(self) -> None:
        self._status.set_state("Refreshing…", "idle")
        self._worker.refresh_now()

    def _on_data(self, _job_name: str, payload: object) -> None:
        # Phase 3 shifted `list_sessions()` to return `list[SessionSummary]`.
        # Phase 10 rewrites this scaffold around the new table model;
        # until then the scaffold just renders a row count.
        rows: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    # Pydantic DTO — render a minimal compat dict.
                    rows.append(
                        {
                            "session_id": getattr(item, "session_id", "?"),
                            "stream_url": "—",
                            "started_at": getattr(item, "started_at_utc", "—"),
                            "ended_at": getattr(item, "ended_at_utc", "—"),
                            "metric_count": "—",
                        }
                    )
        self._model.set_rows(rows)
        self._status.set_state(f"Live · {len(rows)} session(s)", "ok")

    def _on_error(self, _job_name: str, error: object) -> None:
        message = (
            error.message
            if isinstance(error, ApiError)
            else str(error)
        )
        self._status.set_state(message, "bad")

    def shutdown(self) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
