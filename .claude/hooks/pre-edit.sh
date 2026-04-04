#!/usr/bin/env bash
# Pre-edit hook: enforce trust gate and warn on schema/requirements changes
# This hook fires before any Write/Edit/MultiEdit tool call.

FILE="$CLAUDE_FILE_PATH"
TRUST_LOCK="/tmp/.lsie_spec_verified.lock"

# Note: 
# Spec trust gate is now enforced at the system level once per session because Claude bypassed the 'gate' in /implement-file

if [[ ! -f "$TRUST_LOCK" ]]; then
    python scripts/verify_spec_signature.py docs/tech-spec-v3.0.pdf > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        echo "❌ Error: Spec signature verification failed." >&2
        echo "Context: Attempted to edit '$FILE' without a verified spec." >&2
        echo "Action: Trust Gate constraint violated. Stop execution, report this failure, and wait for human to update TRUSTED_SPEC_SIGNERS in .env." >&2
        exit 1
    fi
    touch "$TRUST_LOCK"
fi

# Block edits to requirements/ that don't match §10.2 pinning
if [[ "$FILE" == *"requirements/"* ]]; then
    echo "⚠️  Editing pinned dependencies. Versions MUST match §10.2 dependency matrix." >&2
fi

# Warn on schema edits — contracts affect multiple modules
if [[ "$FILE" == *"packages/schemas/"* ]]; then
    echo "⚠️  Editing interface contract. Changes propagate to Modules C, D, and E. Run pytest after." >&2
fi

exit 0