# macOS signing deferred to v4.1

WS1 P3 intentionally ships only the Windows signing path. macOS notarization, hardened runtime, entitlements, and Developer ID certificate handling are deferred to v4.1 so the v4.0 sprint can finish the Windows-native desktop runtime without adding a second platform-signing surface.

The deferred macOS work should resume from `docs/artifacts/LSIE-MLF_v4_0_Implementation_Plans.md` and produce a separate packaging/signing path rather than sharing the Windows Azure Artifact Signing workflow.
