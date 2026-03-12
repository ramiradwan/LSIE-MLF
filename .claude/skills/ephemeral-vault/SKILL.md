---
name: ephemeral-vault
description: AES-256-GCM encryption and 24-hour secure deletion policy for transient media buffers. Use when working on packages/ml_core/encryption.py, vault_cron.py, data governance code, or any file touching /data/raw/ or /data/interim/ directories.
---

# Ephemeral Vault Specification (§5.1)

## Encryption

Algorithm: AES-256-GCM (authenticated encryption with associated data).
Library: PyCryptodome (`pycryptodome >= 3.20.0`), specifically `Crypto.Cipher.AES`.
Key: 256 bits via `os.urandom(32)`. Unique per debug session.
Nonce: 96 bits (12 bytes) via `os.urandom(12)`. Unique per encryption call.
Key storage: process memory only. NEVER disk, env vars, or config files.
Key lifecycle: generated at session start, destroyed on container termination.

## Secure deletion

Command: `shred -vfz -n 3` executed on `/data/raw/` and `/data/interim/`.
Schedule: internal cron every 24 hours.
Maximum retention: 24 hours for any raw media buffer touching physical disk.

## Data classification tiers (§5.2)

Transient: raw PCM + video in `/tmp/ipc/` (tmpfs) and kernel pipe buffers. No disk persistence.
Debug: encrypted media in `/data/raw/` and `/data/interim/`. 24h max then shredded.
Permanent: anonymized metrics only in PostgreSQL. NEVER raw facial images, voiceprints, or PII.

## Critical constraint

MediaPipe landmarks MUST be reduced to normalized geometric ratios (AU12 intensity values) before any persistence. Raw 478-vertex arrays are transient only.
