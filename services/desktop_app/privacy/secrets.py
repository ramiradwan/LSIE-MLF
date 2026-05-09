"""Persist long-lived secrets in the OS secret store (§5.1.6).

The desktop app keeps a small handful of long-lived credentials across
restarts:

* The cloud OAuth refresh token issued by the desktop PKCE flow.
* The ``OURA_WEBHOOK_SECRET`` if and when the desktop ever ingests the
  Oura webhook locally.

§5.1.6 requires OAuth refresh tokens and cloud credentials to live in
Windows Credential Manager via keyring rather than in SQLite. This
module is the thin service-bound veneer over
:mod:`services.desktop_app.os_adapter` so call sites do not repeat the
``"lsie-mlf"`` service namespace literal everywhere.
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
"""Refresh token issued by the desktop PKCE flow."""

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
