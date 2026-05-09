"""Secret-store wrapper around keyring.

Three surfaces are exercised:

* The os_adapter primitives ``set_secret`` / ``get_secret`` /
  ``delete_secret`` against a synthetic in-memory keyring backend.
* The ``SecretStoreUnavailableError`` translation when ``keyring``
  resolves to ``keyring.backends.fail.Keyring`` (CI default; explicit
  via env injection on dev machines that lack a Credential Manager).
* The :mod:`services.desktop_app.privacy.secrets` thin veneer that
  binds every call to the ``"lsie-mlf"`` service namespace.

A real-Credential-Manager round-trip is gated by the
``LSIE_INTEGRATION_KEYRING=1`` env so the unit suite stays hermetic on
CI runners.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import keyring
import keyring.backends.fail
import keyring.errors
import pytest
from keyring.backend import KeyringBackend

from services.desktop_app import os_adapter
from services.desktop_app.os_adapter import SecretStoreUnavailableError
from services.desktop_app.privacy import secrets


class _InMemoryKeyring(KeyringBackend):
    """Test double that stores secrets in a process-local dict.

    Priority is set above ``fail.Keyring`` so an explicit
    ``keyring.set_keyring`` swap is unambiguous, but the fixture below
    swaps via ``keyring.set_keyring`` regardless to avoid relying on
    backend resolution order.
    """

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError as exc:
            raise keyring.errors.PasswordDeleteError(
                f"no entry for {service!r}/{username!r}"
            ) from exc


@pytest.fixture
def memory_keyring() -> Iterator[_InMemoryKeyring]:
    """Swap the active keyring for an in-memory dict for the test's duration."""
    backend = _InMemoryKeyring()
    previous = keyring.get_keyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(previous)


@pytest.fixture
def fail_keyring() -> Iterator[None]:
    """Pin keyring to the no-backend sentinel for the test's duration."""
    previous = keyring.get_keyring()
    keyring.set_keyring(keyring.backends.fail.Keyring())  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        keyring.set_keyring(previous)


# --------------------------------------------------------------------- #
# os_adapter primitives                                                  #
# --------------------------------------------------------------------- #


def test_set_then_get_round_trips(memory_keyring: _InMemoryKeyring) -> None:
    os_adapter.set_secret("svc", "k", "value-1")
    assert os_adapter.get_secret("svc", "k") == "value-1"
    assert memory_keyring._store == {("svc", "k"): "value-1"}


def test_get_missing_returns_none(memory_keyring: _InMemoryKeyring) -> None:
    assert os_adapter.get_secret("svc", "absent") is None


def test_set_overwrites_existing(memory_keyring: _InMemoryKeyring) -> None:
    os_adapter.set_secret("svc", "k", "first")
    os_adapter.set_secret("svc", "k", "second")
    assert os_adapter.get_secret("svc", "k") == "second"


def test_delete_removes_existing(memory_keyring: _InMemoryKeyring) -> None:
    os_adapter.set_secret("svc", "k", "v")
    assert os_adapter.delete_secret("svc", "k") is True
    assert os_adapter.get_secret("svc", "k") is None


def test_delete_missing_returns_false(memory_keyring: _InMemoryKeyring) -> None:
    assert os_adapter.delete_secret("svc", "absent") is False
    # Idempotent — second call still returns False.
    assert os_adapter.delete_secret("svc", "absent") is False


def test_set_raises_unavailable_when_no_backend(fail_keyring: None) -> None:
    with pytest.raises(SecretStoreUnavailableError):
        os_adapter.set_secret("svc", "k", "v")


def test_get_raises_unavailable_when_no_backend(fail_keyring: None) -> None:
    with pytest.raises(SecretStoreUnavailableError):
        os_adapter.get_secret("svc", "k")


def test_delete_raises_unavailable_when_no_backend(fail_keyring: None) -> None:
    with pytest.raises(SecretStoreUnavailableError):
        os_adapter.delete_secret("svc", "k")


# --------------------------------------------------------------------- #
# privacy.secrets veneer                                                 #
# --------------------------------------------------------------------- #


def test_privacy_secrets_binds_lsie_mlf_service(memory_keyring: _InMemoryKeyring) -> None:
    secrets.set_secret(secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN, "rt-fixture")
    assert secrets.get_secret(secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN) == "rt-fixture"
    # The veneer must address the keyring under the canonical service.
    canonical_key = (secrets.LSIE_SECRET_SERVICE, secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN)
    assert canonical_key in memory_keyring._store


def test_privacy_secrets_delete_round_trip(memory_keyring: _InMemoryKeyring) -> None:
    secrets.set_secret(secrets.SECRET_KEY_OURA_WEBHOOK_SECRET, "ws-fixture")
    assert secrets.delete_secret(secrets.SECRET_KEY_OURA_WEBHOOK_SECRET) is True
    assert secrets.get_secret(secrets.SECRET_KEY_OURA_WEBHOOK_SECRET) is None
    # Second delete is a no-op rather than an error.
    assert secrets.delete_secret(secrets.SECRET_KEY_OURA_WEBHOOK_SECRET) is False


def test_privacy_secrets_propagates_unavailable(fail_keyring: None) -> None:
    with pytest.raises(SecretStoreUnavailableError):
        secrets.set_secret(secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN, "rt-fixture")
    with pytest.raises(SecretStoreUnavailableError):
        secrets.get_secret(secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN)


# --------------------------------------------------------------------- #
# Strings audit                                                          #
# --------------------------------------------------------------------- #


_SECRET_KEY_NAMES = (
    "cloud_oauth_refresh_token",
    "oura_webhook_secret",
    "OURA_WEBHOOK_SECRET",
)


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _scan_dirs() -> list[Path]:
    return [_REPO_ROOT / "services", _REPO_ROOT / "packages"]


def test_no_hardcoded_secret_literals_in_runtime_tree() -> None:
    """Source-tree analogue of the §5.1.6 secret-storage rule.

    Asserts that the runtime tree (``services/`` and ``packages/``)
    never assigns a literal string longer than four characters to any
    of the canonical secret-key identifiers. The full ``strings`` grep
    over the signed binary lands once the signing pipeline
    produces an artefact to scan.

    Rationale: secret values must arrive at the keyring via OAuth flow,
    operator action, or environment injection rather than as baked-in
    literals. A review-time grep is the cheapest way to keep that
    contract honest.
    """
    # Match assignments like:  cloud_oauth_refresh_token = "real-token-value"
    # but not bare references:  return SECRET_KEY_CLOUD_REFRESH_TOKEN.
    names_alt = "|".join(re.escape(n) for n in _SECRET_KEY_NAMES)
    pattern = re.compile(rf"\b({names_alt})\s*=\s*['\"]([^'\"]{{5,}})['\"]")

    offenders: list[tuple[Path, str, str]] = []
    for root in _scan_dirs():
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for m in pattern.finditer(text):
                offenders.append((py, m.group(1), m.group(2)))

    # The legitimate constant declaration in privacy/secrets.py assigns
    # the *identifier name itself* to the constant, e.g.
    # ``SECRET_KEY_CLOUD_REFRESH_TOKEN = "cloud_oauth_refresh_token"``.
    # That is the keyring username slot, not a credential value, and is
    # what we assert below as the only acceptable shape.
    allowed = {
        secrets.SECRET_KEY_CLOUD_REFRESH_TOKEN,
        secrets.SECRET_KEY_OURA_WEBHOOK_SECRET,
    }
    illegitimate = [(p, name, val) for (p, name, val) in offenders if val not in allowed]
    assert illegitimate == [], (
        "hardcoded secret-like literal(s) found in runtime tree: "
        + ", ".join(f"{p}::{name}={val!r}" for (p, name, val) in illegitimate)
    )


# --------------------------------------------------------------------- #
# Real backend (gated)                                                   #
# --------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("LSIE_INTEGRATION_KEYRING") != "1" or sys.platform != "win32",
    reason="set LSIE_INTEGRATION_KEYRING=1 on a Windows host to exercise Credential Manager",
)
def test_credential_manager_round_trip() -> None:
    """End-to-end set/get/delete against the real WinVaultKeyring.

    Uses a fixture-namespaced key so a failed teardown leaves only one
    obvious Credential Manager entry behind for the operator to delete.
    """
    test_key = "ws4-p4-roundtrip-fixture"
    test_value = "fake-token-for-roundtrip-only"
    try:
        secrets.set_secret(test_key, test_value)
        assert secrets.get_secret(test_key) == test_value
    finally:
        secrets.delete_secret(test_key)
    assert secrets.get_secret(test_key) is None
