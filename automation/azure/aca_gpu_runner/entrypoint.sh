#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 1
  fi
}

require_env GITHUB_PAT
require_env GH_URL
require_env REGISTRATION_TOKEN_API_URL

RUNNER_ROOT="/home/runner/actions-runner"
RUNNER_LABELS="${RUNNER_LABELS:-aca-gpu,gpu,turing}"
RUNNER_NAME_PREFIX="${RUNNER_NAME_PREFIX:-aca-gpu}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"
REMOVE_TOKEN_API_URL="${REMOVE_TOKEN_API_URL:-${REGISTRATION_TOKEN_API_URL%/registration-token}/remove-token}"
RUNNER_NAME="${RUNNER_NAME_PREFIX}-$(hostname)-$(date +%s)"

api_headers=(
  -H "Accept: application/vnd.github+json"
  -H "Authorization: Bearer ${GITHUB_PAT}"
  -H "X-GitHub-Api-Version: 2022-11-28"
)

fetch_token() {
  local url="$1"
  curl -fsSL -X POST "${api_headers[@]}" "$url" | jq -r '.token'
}

cleanup() {
  if [[ ! -f "${RUNNER_ROOT}/.runner" ]]; then
    return 0
  fi

  local remove_token
  if ! remove_token="$(fetch_token "${REMOVE_TOKEN_API_URL}")"; then
    return 0
  fi
  if [[ -z "$remove_token" || "$remove_token" == "null" ]]; then
    return 0
  fi

  "${RUNNER_ROOT}/config.sh" remove --unattended --token "$remove_token" || true
}

trap cleanup EXIT INT TERM

registration_token="$(fetch_token "${REGISTRATION_TOKEN_API_URL}")"
if [[ -z "$registration_token" || "$registration_token" == "null" ]]; then
  echo "Failed to acquire a GitHub runner registration token" >&2
  exit 1
fi

cd "$RUNNER_ROOT"
./config.sh \
  --url "$GH_URL" \
  --token "$registration_token" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work "$RUNNER_WORKDIR" \
  --unattended \
  --ephemeral \
  --replace

exec ./run.sh
