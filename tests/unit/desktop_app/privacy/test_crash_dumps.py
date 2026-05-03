"""Crash-dialog suppression and WER LocalDumps exclusion tests."""

from __future__ import annotations

import contextlib
import sys

import pytest

from services.desktop_app.os_adapter import (
    register_localdumps_exclusion,
    suppress_crash_dialogs,
)
from services.desktop_app.privacy.crash_dumps import install_crash_privacy_guards


def test_suppress_crash_dialogs_is_callable_on_every_platform() -> None:
    """Must not raise; Windows calls SetErrorMode, POSIX is a no-op."""
    suppress_crash_dialogs()
    suppress_crash_dialogs()  # Idempotent.


def test_register_localdumps_exclusion_returns_false_on_posix() -> None:
    if sys.platform == "win32":
        pytest.skip("Windows-specific; tested separately on Windows hosts")
    assert register_localdumps_exclusion() is False


@pytest.mark.skipif(sys.platform != "win32", reason="HKCU registry write is Windows-only")
def test_register_localdumps_exclusion_writes_subkey() -> None:
    import winreg

    app_name = "lsie-mlf-test-fixture.exe"
    assert register_localdumps_exclusion(app_name) is True

    subkey = r"Software\Microsoft\Windows\Windows Error Reporting\LocalDumps\\" + app_name
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey) as key:
            dump_type, _ = winreg.QueryValueEx(key, "DumpType")
            dump_count, _ = winreg.QueryValueEx(key, "DumpCount")
        assert dump_type == 0
        assert dump_count == 0
    finally:
        # Tidy up after ourselves so we don't pollute the user's registry.
        with contextlib.suppress(OSError):
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)


def test_install_crash_privacy_guards_does_not_raise() -> None:
    install_crash_privacy_guards("lsie-mlf-test-installer.exe")
    # POSIX silently returns; Windows succeeds. Either way, no exception.
    if sys.platform == "win32":
        import winreg

        subkey = (
            r"Software\Microsoft\Windows\Windows Error Reporting"
            r"\LocalDumps\\lsie-mlf-test-installer.exe"
        )
        with contextlib.suppress(OSError):
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
