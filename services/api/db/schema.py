"""
Database Schema Bootstrap — §5.2 Data Classification / §11 Variable Extraction Matrix

Canonical PostgreSQL bootstrap SQL for the Persistent Store. The API bootstrap
surface mirrors the SQL files mounted into the postgres container so fresh
and application-driven initialization paths share one deterministic order.
"""

from __future__ import annotations

from pathlib import Path

# §5.2 — Permanent Analytical Storage tier
# Only anonymized analytical metrics and derived/versioned attribution
# artifacts are persisted.

_SQL_DIR = Path(__file__).resolve().parents[3] / "data" / "sql"

# Keep this tuple explicit: PostgreSQL's docker-entrypoint processes the same
# file names alphabetically, while API-side bootstrap consumers should not rely
# on filesystem glob ordering. 05-attribution.sql is intentionally loaded
# immediately after the v3.3/v3.4 metrics/acoustics rollout migration.
SQL_BOOTSTRAP_FILES: tuple[Path, ...] = (
    _SQL_DIR / "01-schema.sql",
    _SQL_DIR / "02-seed-experiments.sql",
    _SQL_DIR / "03-encounter-log.sql",
    _SQL_DIR / "03-physiology.sql",
    _SQL_DIR / "04-metrics-observational-acoustics.sql",
    _SQL_DIR / "05-attribution.sql",
)


def _load_bootstrap_sql(files: tuple[Path, ...] = SQL_BOOTSTRAP_FILES) -> str:
    """Load schema bootstrap SQL once in deterministic canonical order."""

    return "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in files) + "\n"


SCHEMA_SQL: str = _load_bootstrap_sql()
