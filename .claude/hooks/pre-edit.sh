#!/usr/bin/env bash
# Pre-edit hook: warn on schema or requirements changes
# This hook fires before any Write/Edit/MultiEdit tool call.

FILE="$CLAUDE_FILE_PATH"

# Block edits to requirements/ that don't match §10.2 pinning
if [[ "$FILE" == *"requirements/"* ]]; then
    echo "⚠️  Editing pinned dependencies. Versions MUST match §10.2 dependency matrix." >&2
fi

# Warn on schema edits — contracts affect multiple modules
if [[ "$FILE" == *"packages/schemas/"* ]]; then
    echo "⚠️  Editing interface contract. Changes propagate to Modules C, D, and E. Run pytest after." >&2
fi

exit 0
