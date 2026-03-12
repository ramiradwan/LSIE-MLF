"""
Tests for packages/ml_core/encryption.py — Phase 0 validation.

Verifies Ephemeral Vault AES-256-GCM encryption per §5.1:
round-trip encrypt/decrypt, nonce uniqueness, key properties.
"""

from __future__ import annotations

import pytest

from packages.ml_core.encryption import EphemeralVault, VaultSession


class TestVaultSession:
    """§5.1 — Per-session encryption context."""

    def test_key_is_256_bits(self) -> None:
        session = VaultSession()
        assert len(session.key) == 32  # 256 bits

    def test_nonce_is_96_bits(self) -> None:
        session = VaultSession()
        nonce = session.generate_nonce()
        assert len(nonce) == 12  # 96 bits

    def test_nonces_are_unique(self) -> None:
        session = VaultSession()
        nonces = {session.generate_nonce() for _ in range(100)}
        assert len(nonces) == 100

    def test_keys_differ_across_sessions(self) -> None:
        s1 = VaultSession()
        s2 = VaultSession()
        assert s1.key != s2.key


class TestEphemeralVault:
    """§5.1 — AES-256-GCM encrypt/decrypt round-trip."""

    def test_encrypt_decrypt_roundtrip(self) -> None:
        vault = EphemeralVault()
        plaintext = b"sensitive biometric data for testing"
        nonce, ciphertext, tag = vault.encrypt(plaintext)
        decrypted = vault.decrypt(nonce, ciphertext, tag)
        assert decrypted == plaintext

    def test_ciphertext_differs_from_plaintext(self) -> None:
        vault = EphemeralVault()
        plaintext = b"test data"
        _, ciphertext, _ = vault.encrypt(plaintext)
        assert ciphertext != plaintext

    def test_tampered_ciphertext_fails(self) -> None:
        vault = EphemeralVault()
        nonce, ciphertext, tag = vault.encrypt(b"original")
        tampered = bytes([b ^ 0xFF for b in ciphertext])
        with pytest.raises(ValueError):
            vault.decrypt(nonce, tampered, tag)

    def test_wrong_key_fails(self) -> None:
        vault1 = EphemeralVault()
        vault2 = EphemeralVault()
        nonce, ciphertext, tag = vault1.encrypt(b"secret")
        with pytest.raises(ValueError):
            vault2.decrypt(nonce, ciphertext, tag)
