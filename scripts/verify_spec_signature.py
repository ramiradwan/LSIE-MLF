"""
Spec Signature Verification — Trust Gate

Verifies that docs/tech-spec-v2.0.pdf carries a valid digital signature
from a trusted signer before an AI agent references it as authoritative.
Enforces strict Azure Artifact Signing PAdES-LTA pipeline structure and
pins the trusted Root CA certificate by SHA-256 fingerprint.

Trusted signers are loaded from the TRUSTED_SPEC_SIGNERS environment
variable (comma-separated). This variable is NOT checked into the repo.
You must inspect the signing certificate in the spec PDF, identify the
signer organization, and explicitly add it to your .env file.

Usage:
    python scripts/verify_spec_signature.py docs/tech-spec-v2.0.pdf

Exit codes:
    0 — Signature valid, signer trusted
    1 — Signature invalid, missing, signer not trusted, or env not configured
"""

from __future__ import annotations

import binascii
import hashlib
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project root .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

# Expected signature fields in the PDF
EXPECTED_SIG_FIELDS: list[str] = ["ArchiveSignature"]

# SHA-256 fingerprint of the trusted root CA.
# Microsoft Identity Verification Root Certificate Authority 2020.
# This is a publicly known value — pinning it here prevents root cert
# substitution attacks without leaking any project-specific secrets.
TRUSTED_ROOT_SHA256: str = (
    "5367f20c7ade0e2bca790915056d086b720c33c1fa2a2661acf787e3292e1270"
)


def load_trusted_signers() -> set[str] | None:
    """
    Load trusted signer organizations from TRUSTED_SPEC_SIGNERS env var.

    Returns None if the variable is not set or empty, which means the
    developer has not yet inspected the spec's certificate and explicitly
    granted trust. This is intentional — the trust decision must be
    conscious, not inherited from a cloned repo.
    """
    raw = os.environ.get("TRUSTED_SPEC_SIGNERS", "").strip()
    if not raw:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def preview_signature_natively(pdf_path: Path) -> str:
    """
    Zero-dependency extraction of Organization and Common Name from the PDF.

    Parses raw ASN.1 DER bytes in the signature /Contents hex stream to
    find OID 2.5.4.10 (Organization) and 2.5.4.3 (Common Name) values.
    Filters out Microsoft CA boilerplate to surface only the end-entity
    signer identity.
    """
    try:
        data = pdf_path.read_bytes()
        matches = re.findall(rb"/Contents\s*<([0-9a-fA-F]+)>", data)
        if not matches:
            return "  No hex-encoded signature contents found."

        previews: list[str] = []
        for i, hex_data in enumerate(matches):
            try:
                raw_bytes = binascii.unhexlify(hex_data)
                entities: list[tuple[str, str]] = []

                # OIDs: 2.5.4.10 = Organization, 2.5.4.3 = Common Name
                target_oids = [
                    ("Organization (O)", b"\x06\x03\x55\x04\x0a"),
                    ("Common Name (CN)", b"\x06\x03\x55\x04\x03"),
                ]

                for label, oid in target_oids:
                    # String tags: 0x0c=UTF8, 0x13=Printable, 0x14=T61, 0x1e=BMP
                    for match in re.finditer(
                        oid + b"([\x0c\x13\x14\x1e])(.)", raw_bytes, re.DOTALL
                    ):
                        length = match.group(2)[0]
                        start_idx = match.end()
                        value = raw_bytes[start_idx : start_idx + length].decode(
                            "utf-8", errors="ignore"
                        )

                        # Filter out Microsoft CA boilerplate to surface end-entity only
                        ca_keywords = ("Microsoft", "Authority", "Root", "Timestamp")
                        if any(kw in value for kw in ca_keywords):
                            continue
                        if value not in [e[1] for e in entities]:
                            entities.append((label, value))

                if entities:
                    formatted = "\n".join(f"    {k}: {v}" for k, v in entities)
                    previews.append(
                        f"  --- Signature {i + 1} Signer Identity ---\n{formatted}"
                    )
            except Exception:
                continue

        return "\n\n".join(previews) if previews else (
            "  Could not locate signer identity OIDs in signature."
        )
    except Exception as e:
        return f"  Error reading signature natively: {e}"


def verify_with_pyhanko(
    pdf_path: Path, trusted_signers: set[str]
) -> tuple[bool, str]:
    """
    Full cryptographic verification using pyhanko.

    Enforces:
      - Anti-tamper: no uncovered bytes after the final signature
      - Pipeline structure: exactly 2 signatures (CAdES + RFC3161 timestamp)
      - DSS presence for PAdES-LTA long-term validation
      - Root CA pinning via SHA-256 fingerprint
      - Signer organization in TRUSTED_SPEC_SIGNERS
    """
    try:
        import logging

        from pyhanko.pdf_utils.reader import PdfFileReader
        from pyhanko.sign.validation import validate_pdf_signature
        from pyhanko.sign.validation.settings import KeyUsageConstraints

        logging.getLogger("pyhanko").setLevel(logging.CRITICAL)

        with open(pdf_path, "rb") as f:
            reader = PdfFileReader(f)
            sig_fields = reader.embedded_signatures

            if not sig_fields:
                return False, "FAIL: No digital signatures found in PDF."

            # --- Anti-tamper: uncovered bytes after final signature ---
            f.seek(0, os.SEEK_END)
            actual_file_length = f.tell()
            last_sig = sig_fields[-1]
            byte_range = last_sig.sig_object.get("/ByteRange")

            if byte_range:
                covered_length = int(byte_range[2]) + int(byte_range[3])
                if actual_file_length > covered_length + 2:
                    return False, (
                        f"FAIL: Tampering detected: {actual_file_length - covered_length} "
                        "uncovered bytes after the final signature."
                    )

            # --- Pipeline structure: exactly CAdES sig + RFC3161 timestamp ---
            if len(sig_fields) != 2:
                return False, (
                    f"FAIL: Expected 2 signature objects (Sig + TimeStamp), "
                    f"found {len(sig_fields)}."
                )

            main_sig = sig_fields[0]
            ts_sig = sig_fields[1]

            if main_sig.sig_object.get("/Type") != "/Sig":
                return False, "FAIL: First signature is not a standard /Sig."

            if ts_sig.sig_object.get("/Type") != "/DocTimeStamp":
                return False, "FAIL: Final signature is not a /DocTimeStamp."

            # --- DSS presence for PAdES-LTA ---
            if "/DSS" not in reader.root:
                return False, "FAIL: Missing Document Security Store (/DSS) for LTA."

            # --- Root CA pinning via SHA-256 fingerprint ---
            root_found = False
            signed_data = main_sig.signed_data

            if (
                signed_data
                and "certificates" in signed_data
                and signed_data["certificates"]
            ):
                for cert_choice in signed_data["certificates"]:
                    if cert_choice.name == "certificate":
                        cert_der = cert_choice.chosen.dump()
                        cert_hash = hashlib.sha256(cert_der).hexdigest()
                        if cert_hash.lower() == TRUSTED_ROOT_SHA256.lower():
                            root_found = True
                            break

            if not root_found:
                return False, (
                    "FAIL: Trusted Root CA fingerprint not found in the signature "
                    "chain. Possible spoofed or self-signed certificate."
                )

            # --- Signer trust validation ---
            ku_settings = KeyUsageConstraints(
                key_usage={"digital_signature", "non_repudiation"}
            )
            status = validate_pdf_signature(main_sig, key_usage_settings=ku_settings)

            if not status.intact:
                return False, (
                    f"FAIL: Signature integrity check failed: {main_sig.field_name}"
                )
            if not status.valid:
                return False, (
                    f"FAIL: Signature validation failed: {main_sig.field_name}"
                )

            cert = status.signing_cert
            org = cert.subject.native.get("organization_name", "")

            if org not in trusted_signers:
                return False, (
                    f"FAIL: Signer '{org}' is not in TRUSTED_SPEC_SIGNERS. "
                    "Inspect the certificate and add it to your .env file."
                )

            return True, (
                "Valid PAdES-LTA signature from trusted signer. "
                f"Main: {main_sig.field_name}, TimeStamp: {ts_sig.field_name}, "
                "DSS: present, Root CA: pinned."
            )

    except ImportError:
        return verify_structural(pdf_path, trusted_signers)
    except Exception as e:
        return False, f"FAIL: Verification error: {e}"


def verify_structural(
    pdf_path: Path, trusted_signers: set[str]
) -> tuple[bool, str]:
    """
    Lightweight structural verification when pyhanko is unavailable.

    Confirms the PDF contains the expected signature objects, CAdES/PAdES
    SubFilter, RFC 3161 timestamp, DSS, and a trusted signer name in the
    certificate chain. This does NOT perform cryptographic validation.
    """
    try:
        raw = pdf_path.read_bytes()

        checks: dict[str, bool] = {
            "has_sig_field": b"/FT /Sig" in raw,
            "has_archive_sig": b"ArchiveSignature" in raw,
            "has_cades": b"/ETSI.CAdES.detached" in raw,
            "has_timestamp": b"/ETSI.RFC3161" in raw,
            "has_byterange": b"/ByteRange" in raw,
            "has_dss": b"/DSS" in raw,
            "has_trusted_signer": any(s.encode() in raw for s in trusted_signers),
        }

        failed = [k for k, v in checks.items() if not v]

        if failed:
            if "has_trusted_signer" in failed:
                return False, (
                    "FAIL: Signer organization not found in PDF certificate chain. "
                    f"Expected one of: {trusted_signers}. "
                    "Inspect the certificate and update TRUSTED_SPEC_SIGNERS in .env."
                )
            return False, f"FAIL: Structural checks failed: {failed}"

        return True, (
            "Structural verification passed (no cryptographic validation). "
            "Install pyhanko for full signature chain verification."
        )

    except Exception as e:
        return False, f"FAIL: Structural verification error: {e}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-spec-pdf>")
        return 1

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"FAIL: Spec file not found: {pdf_path}")
        return 1

    trusted_signers = load_trusted_signers()

    if trusted_signers is None:
        print(
            "FAIL: TRUSTED_SPEC_SIGNERS is not set.\n"
            "\n"
            "  This is the Trust Gate. Before the agent can reference the\n"
            "  spec as authoritative, you must inspect the PDF's digital\n"
            "  signature, identify the signing organization, and explicitly\n"
            "  grant trust by adding it to your .env file:\n"
            "\n"
            "    TRUSTED_SPEC_SIGNERS=<org-name-from-certificate>\n"
            "\n"
            "  --- Native Signature Inspection ---\n"
        )
        print(preview_signature_natively(pdf_path))
        print("\n  If you trust this signer, update your .env file and re-run.")
        return 1

    valid, message = verify_with_pyhanko(pdf_path, trusted_signers)

    if valid:
        print(f"PASS: {message}")
        return 0
    else:
        print(message)
        return 1


if __name__ == "__main__":
    sys.exit(main())
