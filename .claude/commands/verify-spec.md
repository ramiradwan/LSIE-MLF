Before referencing docs/tech-spec-v3.1.pdf as authoritative, verify its digital signature.

Steps:
1. Run `python scripts/verify_spec_signature.py docs/tech-spec-v3.1.pdf`
2. If PASS: the spec is signed by a signer listed in TRUSTED_SPEC_SIGNERS. Proceed with the task.
3. If FAIL due to missing TRUSTED_SPEC_SIGNERS: the developer must inspect the PDF's digital
   signature certificate, identify the signing organization, and add it to .env. Do NOT guess
   the signer name or hardcode it. The trust decision belongs to the human operator.
4. If FAIL for any other reason: STOP. Do not reference the spec. Report the failure to the user.

This verification confirms the PDF was signed with an Azure Artifact Signing certificate
issued to a known organization and has not been tampered with since signing.
