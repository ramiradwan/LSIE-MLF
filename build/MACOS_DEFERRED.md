# macOS signing deferred to v4.1

v4.0 ships only the Windows signing path. macOS notarization, hardened runtime, entitlements, and Developer ID certificate handling remain Tier 2 design intent for v4.1 and are not part of the signed v4.0 release lane.

When macOS packaging resumes, use the signed `docs/tech-spec-v*.pdf` contract and `scripts/spec_ref_check.py` reference tooling; treat `docs/artifacts/LSIE-MLF_v4_0_Implementation_Plans.md` as historical context. Build a separate macOS packaging and signing path rather than reusing the Windows Azure Artifact Signing workflow.
