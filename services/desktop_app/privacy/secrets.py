"""Persist long-lived secrets in the OS secret store (WS4 P4 / §5).

The desktop app needs to keep a small handful of long-lived credentials
across restarts:

* The cloud OAuth refresh token issued by the WS5 PKCE flow. Without
  it the operator must re-authenticate every cold start.
* The ``OURA_WEBHOOK_SECRET`` if and when the desktop ever ingests the
  Oura webhook locally (today the v4.0 deployment routes the webhook
  through the cloud API server, so this slot is reserved but unused).

The §5 privacy clause forbids embedding any secret literal in the
shipped binary or in the configuration file the operator can read.
``keyring`` resolves to ``WinVaultKeyring`` on Windows — DPAPI-backed
and TPM-backed where available — so the secret is encrypted with a key
the user cannot extract by reading the application's install tree.

All Win32 / POSIX branching lives in
:mod:`services.desktop_app.os_adapter` per the Platform Abstraction
Rule. This module is a thin service-bound veneer over those primitives
so call sites read as ``set_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN, t)``
instead of repeating the ``"lsie-mlf"`` service-name literal everywhere.
"""

from __future__ import annotations

from services.desktop_app import os_adapter
from services.desktop_app.os_adapter import SecretStoreUnavailableError

LSIE_SECRET_SERVICE: str = "lsie-mlf"
"""Keyring ``service`` namespace for every LSIE-MLF secret slot.

Callers always use this constant via the helpers below; it appears in
exactly one place in the source tree to keep the ``strings`` audit
narrow.
"""

SECRET_KEY_CLOUD_REFRESH_TOKEN: str = "cloud_oauth_refresh_token"
"""Refresh token issued by the WS5 PKCE flow."""

SECRET_KEY_OURA_WEBHOOK_SECRET: str = "oura_webhook_secret"
"""Reserved slot for an on-desktop Oura webhook ingest (deferred)."""


__all__ = [
    "LSIE_SECRET_SERVICE",
    "SECRET_KEY_CLOUD_REFRESH_TOKEN",
    "SECRET_KEY_OURA_WEBHOOK_SECRET",
    "SecretStoreUnavailableError",
    "delete_secret",
    "get_secret",
    "set_secret",
]


def set_secret(key: str, value: str) -> None:
    """Persist ``value`` under :data:`LSIE_SECRET_SERVICE` / ``key``.

    Raises :class:`SecretStoreUnavailableError` when no keyring backend
    is available (typical CI configuration). Callers in the cloud-sync
    path turn that into a hard error on the operator health page.
    """
    os_adapter.set_secret(LSIE_SECRET_SERVICE, key, value)


def get_secret(key: str) -> str | None:
    """Return the secret at ``key`` or ``None`` if it is not set."""
    return os_adapter.get_secret(LSIE_SECRET_SERVICE, key)


def delete_secret(key: str) -> bool:
    """Delete the secret at ``key``. Idempotent — ``False`` if absent."""
    return os_adapter.delete_secret(LSIE_SECRET_SERVICE, key)
