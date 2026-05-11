#!/usr/bin/env bash
# Post-edit hook: cheap file-local validation only.

set -euo pipefail

export PATH="$HOME/.local/bin:${USERPROFILE:-}/.local/bin:$PATH"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
INPUT="$(cat || true)"

if command -v python3 > /dev/null 2>&1; then
    HELPER_PYTHON=(python3)
elif command -v python > /dev/null 2>&1; then
    HELPER_PYTHON=(python)
else
    HELPER_PYTHON=(py.exe -3)
fi

is_wsl() {
    [[ -r /proc/version ]] && grep -qi microsoft /proc/version
}

ps_quote() {
    printf "%s" "$1" | sed "s/'/''/g"
}

run_project_python() {
    if is_wsl && command -v powershell.exe > /dev/null 2>&1 && command -v wslpath > /dev/null 2>&1; then
        local win_root
        local command_text
        win_root="$(wslpath -w "$REPO_ROOT")"
        command_text="Set-Location -LiteralPath '$(ps_quote "$win_root")'; uv run python"
        for arg in "$@"; do
            command_text="$command_text '$(ps_quote "$arg")'"
        done
        powershell.exe -NoProfile -Command "$command_text"
    elif command -v uv > /dev/null 2>&1; then
        (cd "$REPO_ROOT" && uv run python "$@")
    else
        (cd "$REPO_ROOT" && "${HELPER_PYTHON[@]}" "$@")
    fi
}

mapfile -t FILES < <(
    HOOK_INPUT="$INPUT" CLAUDE_FILE_PATH="${CLAUDE_FILE_PATH:-}" "${HELPER_PYTHON[@]}" - <<'PY'
from __future__ import annotations

import json
import os

raw = os.environ.get("HOOK_INPUT", "")
files: list[str] = []
if raw.strip():
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    tool_input = payload.get("tool_input", {}) if isinstance(payload, dict) else {}
    if isinstance(tool_input, dict):
        file_path = tool_input.get("file_path")
        if isinstance(file_path, str):
            files.append(file_path)
        edits = tool_input.get("edits")
        if isinstance(edits, list):
            for edit in edits:
                if isinstance(edit, dict) and isinstance(edit.get("file_path"), str):
                    files.append(edit["file_path"])

fallback = os.environ.get("CLAUDE_FILE_PATH")
if fallback:
    files.append(fallback)

for file_path in dict.fromkeys(files):
    print(file_path)
PY
)

if [[ ${#FILES[@]} -eq 0 ]]; then
    exit 0
fi

normalize_path() {
    FILE_TO_NORMALIZE="$1" REPO_ROOT="$REPO_ROOT" "${HELPER_PYTHON[@]}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

repo_root = Path(os.environ["REPO_ROOT"])
value = os.environ["FILE_TO_NORMALIZE"]
path = Path(value)
if path.is_absolute():
    try:
        print(path.resolve().relative_to(repo_root).as_posix())
    except ValueError:
        print(path.as_posix())
else:
    print(str(PurePosixPath(value.replace("\\", "/"))))
PY
}

for file in "${FILES[@]}"; do
    rel="$(normalize_path "$file")"
    abs="$REPO_ROOT/$rel"
    if [[ ! -e "$abs" ]]; then
        continue
    fi

    if [[ "$rel" == *.py ]]; then
        if ! run_project_python -m py_compile "$rel" > /dev/null; then
            echo "Syntax error in $rel — fix before continuing." >&2
            exit 1
        fi
    fi

    if [[ "$rel" == automation/work-items/active/*.json || "$rel" == automation/work-items/templates/spec_work_item.json ]]; then
        if ! run_project_python automation/schemas/spec_work_item.py "$rel" > /dev/null; then
            echo "Spec work-item validation failed for $rel." >&2
            exit 1
        fi
    fi

    if [[ "$rel" == .claude/hooks/*.sh ]]; then
        if ! bash -n "$abs"; then
            echo "Shell syntax validation failed for $rel." >&2
            exit 1
        fi
    fi

    if [[ "$rel" == .claude/settings.json ]]; then
        if ! run_project_python -m json.tool "$rel" > /dev/null; then
            echo "Claude settings JSON validation failed for $rel." >&2
            exit 1
        fi
    fi
done

exit 0
