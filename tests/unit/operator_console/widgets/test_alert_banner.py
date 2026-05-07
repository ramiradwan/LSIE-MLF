"""Tests for `AlertBanner` — Phase 5.

Covers:
  - banner hidden by default
  - `set_alert(None, ...)` or empty-message hides the banner
  - `set_alert(severity, message)` shows it and updates the text + glyph
  - object name changes with severity so the stylesheet targets the
    right severity bucket
"""

from __future__ import annotations

import pytest

from packages.schemas.operator_console import AlertSeverity
from services.operator_console.widgets.alert_banner import AlertBanner

pytestmark = pytest.mark.usefixtures("qt_app")


def test_banner_hidden_by_default() -> None:
    banner = AlertBanner()
    assert banner.isHidden() is True


def test_set_alert_none_hides_banner() -> None:
    banner = AlertBanner()
    banner.set_alert(AlertSeverity.WARNING, "stale physiology")
    banner.set_alert(None, None)
    assert banner.isHidden() is True


def test_set_alert_shows_banner_with_glyph_and_message() -> None:
    banner = AlertBanner()
    banner.set_alert(AlertSeverity.WARNING, "stale physiology")
    assert banner.isHidden() is False
    assert banner._message.text() == "stale physiology"  # type: ignore[attr-defined]
    # Glyph is now a QSvgWidget; severity shows up via accessible name.
    assert banner._glyph.accessibleName() == "warning"  # type: ignore[attr-defined]


def test_severity_drives_object_name() -> None:
    banner = AlertBanner()
    banner.set_alert(AlertSeverity.INFO, "session starting")
    assert banner.objectName() == "AlertBannerInfo"
    banner.set_alert(AlertSeverity.CRITICAL, "GPU lost")
    assert banner.objectName() == "AlertBannerCritical"
    banner.set_alert(None, None)
    assert banner.objectName() == "AlertBanner"
