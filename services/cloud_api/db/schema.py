from __future__ import annotations

from pathlib import Path

_SQL_DIR = Path(__file__).resolve().parent / "sql"

SQL_BOOTSTRAP_FILES: tuple[Path, ...] = (
    _SQL_DIR / "01-schema.sql",
    _SQL_DIR / "02-seed-experiments.sql",
    _SQL_DIR / "03-encounter-log.sql",
    _SQL_DIR / "03-physiology.sql",
    _SQL_DIR / "04-metrics-observational-acoustics.sql",
    _SQL_DIR / "05-attribution.sql",
    _SQL_DIR / "06-cloud-sync.sql",
)


def _load_bootstrap_sql(files: tuple[Path, ...] = SQL_BOOTSTRAP_FILES) -> str:
    return "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in files) + "\n"


SCHEMA_SQL: str = _load_bootstrap_sql()
