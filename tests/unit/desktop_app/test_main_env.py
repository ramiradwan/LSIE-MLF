from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import services.desktop_app.__main__ as desktop_main


def test_load_local_env_reads_repo_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    load_dotenv = Mock()
    monkeypatch.setattr(desktop_main, "load_dotenv", load_dotenv)

    desktop_main._load_local_env()

    load_dotenv.assert_called_once_with(
        dotenv_path=Path(desktop_main.__file__).resolve().parents[2] / ".env",
        override=False,
    )
