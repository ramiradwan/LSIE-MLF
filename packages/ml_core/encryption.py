"""
Ephemeral Vault — §5.1 Encryption Specification

AES-256-GCM authenticated encryption for transient media buffers.
Keys exist in process memory only and are never written to disk.
Mandatory 24-hour secure deletion via shred.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class VaultSession:
    """
    Per-session encryption context.

    §5.1 Key lifecycle: generated at session start, destroyed when
    the container terminates. No persistence mechanism exists.
    """

    key: bytes = field(default_factory=lambda: os.urandom(32))  # 256 bits
    nonce_length: int = 12  # 96 bits

    def generate_nonce(self) -> bytes:
        """Generate a unique IV/Nonce using os.urandom(12)."""
        return os.urandom(self.nonce_length)


class EphemeralVault:
    """
    §5.1 — AES-256-GCM encrypted transient storage.

    Encryption: PyCryptodome (pycryptodome >= 3.20.0)
    Secure deletion: shred -vfz -n 3 on /data/raw/ and /data/interim/
    Retention: Maximum 24 hours for any raw media buffer touching disk.
    """

    def __init__(self) -> None:
        self.session: VaultSession = VaultSession()

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
        """
        Encrypt data with AES-256-GCM.

        §5.1 — PyCryptodome AES-256-GCM authenticated encryption.

        Returns:
            Tuple of (nonce, ciphertext, tag).
        """
        from Crypto.Cipher import AES  # §5.1 — pycryptodome >= 3.20.0

        nonce: bytes = self.session.generate_nonce()
        cipher = AES.new(self.session.key, AES.MODE_GCM, nonce=nonce)
        ciphertext: bytes
        tag: bytes
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        return nonce, ciphertext, tag

    def decrypt(self, nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes:
        """
        Decrypt AES-256-GCM ciphertext.

        §5.1 — Authenticated decryption; raises ValueError on tamper.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If authentication tag verification fails.
        """
        from Crypto.Cipher import AES  # §5.1 — pycryptodome >= 3.20.0

        cipher = AES.new(self.session.key, AES.MODE_GCM, nonce=nonce)
        try:
            plaintext: bytes = cipher.decrypt_and_verify(ciphertext, tag)
        except (ValueError, KeyError) as exc:
            raise ValueError("Decryption failed: authentication tag mismatch") from exc
        return plaintext

    @staticmethod
    def secure_delete(target_dir: str) -> None:
        """
        Execute shred -vfz -n 3 on all files in target_dir.

        §5.1 — Mandatory 24-hour secure deletion via shred on
        /data/raw/ and /data/interim/.
        """
        import pathlib
        import subprocess

        target: pathlib.Path = pathlib.Path(target_dir)
        if not target.is_dir():
            return

        files: list[pathlib.Path] = [f for f in target.iterdir() if f.is_file()]
        if not files:
            return

        # §5.1 — shred -vfz -n 3 for secure deletion
        subprocess.run(
            ["shred", "-vfz", "-n", "3", *(str(f) for f in files)],
            check=True,
        )
