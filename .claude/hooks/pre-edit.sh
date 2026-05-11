#!/usr/bin/env bash
# Pre-edit hook: trust gate plus fast SDD context warnings.

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

TRUST_LOCK="${TMPDIR:-/tmp}/lsie-spec-verified-$(printf '%s' "$REPO_ROOT" | "${HELPER_PYTHON[@]}" -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:16])').lock"

if [[ ! -f "$TRUST_LOCK" ]]; then
    pushd "$REPO_ROOT" > /dev/null
    mapfile -t SPEC_PDFS < <(compgen -G 'docs/tech-spec-v*.pdf' | sort)
    if [[ ${#SPEC_PDFS[@]} -ne 1 ]]; then
        echo "Error: Expected exactly one docs/tech-spec-v*.pdf match; found ${#SPEC_PDFS[@]}." >&2
        popd > /dev/null
        exit 1
    fi
    if ! run_project_python scripts/verify_spec_signature.py "${SPEC_PDFS[0]}" > /dev/null 2>&1; then
        echo "Error: Spec signature verification failed." >&2
        echo "Context: Attempted to edit '${FILES[*]}' without a verified spec." >&2
        echo "Action: Stop and ask a human to update TRUSTED_SPEC_SIGNERS in .env." >&2
        popd > /dev/null
        exit 1
    fi
    touch "$TRUST_LOCK"
    popd > /dev/null
fi

HOOK_FILES="$(printf '%s\n' "${FILES[@]}")" REPO_ROOT="$REPO_ROOT" "${HELPER_PYTHON[@]}" - <<'PY' >&2
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

repo_root = Path(os.environ["REPO_ROOT"])
files = [line.strip() for line in os.environ.get("HOOK_FILES", "").splitlines() if line.strip()]

def normalize(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()
    return str(PurePosixPath(value.replace("\\", "/")))

active_packets = sorted((repo_root / "automation" / "work-items" / "active").glob("*.json"))
packet_scopes: list[tuple[str, set[str], set[str]]] = []
for packet in active_packets:
    try:
        payload = json.loads(packet.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    target_files = payload.get("target_files", [])
    dependencies = payload.get("dependencies", [])
    packet_scopes.append(
        (
            packet.relative_to(repo_root).as_posix(),
            {normalize(item) for item in target_files if isinstance(item, str)},
            {normalize(item) for item in dependencies if isinstance(item, str)},
        )
    )

implementation_prefixes = ("packages/", "services/", "automation/", "tests/", "scripts/")
schema_prefixes = ("packages/schemas/", "services/cloud_api/db/sql/", "automation/schemas/")

for file_path in files:
    rel = normalize(file_path)
    if rel in {"pyproject.toml", "uv.lock"}:
        print(
            "Warning: editing canonical dependency surfaces. Dependency changes require §10.2/spec governance and the Dependabot process; hooks do not own dependency review."
        )
    if rel.startswith(schema_prefixes):
        print(
            "Warning: editing schema/contract surfaces. Use schema-consistency guidance and verify with scripts/check_schema_consistency.py before completion."
        )
    if not rel.startswith(implementation_prefixes):
        continue
    if rel.startswith("automation/work-items/active/"):
        continue
    if not packet_scopes:
        print(
            "Warning: no local active spec-work-item packet found. Work-item-first SDD expects automation/work-items/active/*.json created from automation/work-items/templates/spec_work_item.json and validated with automation/schemas/spec_work_item.py."
        )
        continue
    target_matches = [packet for packet, targets, _deps in packet_scopes if rel in targets]
    if target_matches:
        continue
    dependency_matches = [packet for packet, _targets, deps in packet_scopes if rel in deps]
    if dependency_matches:
        print(
            f"Warning: {rel} is listed only as a dependency in active packet(s): {', '.join(dependency_matches)}. Update packet target_files if this becomes implementation scope."
        )
    else:
        packet_names = ", ".join(packet for packet, _targets, _deps in packet_scopes)
        print(
            f"Warning: {rel} is outside active packet target_files/dependencies. Active packet(s): {packet_names}. Keep scope aligned or update the local packet."
        )
PY

exit 0
