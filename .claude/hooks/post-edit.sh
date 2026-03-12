#!/usr/bin/env bash
# Post-edit hook: syntax-check any edited Python file.
# This hook fires after any Write/Edit/MultiEdit tool call.

FILE="$CLAUDE_FILE_PATH"

if [[ "$FILE" == *.py ]]; then
    python3 -m py_compile "$FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Syntax error in $FILE — fix before continuing." >&2
        exit 1
    fi
fi

exit 0
