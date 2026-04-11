#!/usr/bin/env bash
# =============================================================================
# LSIE-MLF End-to-End Experiment Verification
#
# Checks every stage of the data flow to confirm a successful experiment run:
#   Stage 1 — Infrastructure alive (containers, DB schema, seed data)
#   Stage 2 — Capture pipeline streaming (scrcpy, IPC pipes)
#   Stage 3 — Orchestrator running (session, arm selection, AU12 init)
#   Stage 4 — Stimulus injected (calibration ended, timestamp recorded)
#   Stage 5 — Segments produced and processed (Module D pipeline)
#   Stage 6 — Metrics persisted to PostgreSQL
#   Stage 7 — Thompson Sampling posterior updated (the money check)
#
# Usage:
#   bash scripts/verify_e2e.sh              # Check everything
#   bash scripts/verify_e2e.sh --wait 120   # Wait up to 120s for first TS update
#   bash scripts/verify_e2e.sh --quick      # Skip the wait, check DB state only
#
# Requires: docker, docker compose, psql (via docker exec), jq (optional)
# =============================================================================

set -euo pipefail

# --- Colors and formatting ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass()  { echo -e "  ${GREEN}✓${NC} $1"; ((PASS_COUNT++)); }
fail()  { echo -e "  ${RED}✗${NC} $1"; ((FAIL_COUNT++)); }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; ((WARN_COUNT++)); }
info()  { echo -e "  ${DIM}$1${NC}"; }
header(){ echo -e "\n${CYAN}━━ $1 ━━${NC}"; }

# --- Parse args ---
WAIT_TIMEOUT=90
QUICK=false
for arg in "$@"; do
    case $arg in
        --wait) shift; WAIT_TIMEOUT="${1:-90}"; shift ;;
        --quick) QUICK=true ;;
    esac
done

# --- Helpers ---
db_query() {
    docker compose exec -T postgres psql -U "${POSTGRES_USER:-lsie}" \
        -d "${POSTGRES_DB:-lsie_mlf}" -t -A -c "$1" 2>/dev/null
}

container_running() {
    docker compose ps --status running --format '{{.Name}}' 2>/dev/null | grep -q "^$1$"
}

container_healthy() {
    local health
    health=$(docker inspect --format='{{.State.Health.Status}}' "$1" 2>/dev/null || echo "none")
    [ "$health" = "healthy" ]
}

log_contains() {
    # $1 = service, $2 = pattern
    docker compose logs "$1" 2>/dev/null | grep -q "$2"
}

log_last_match() {
    # $1 = service, $2 = pattern — returns last matching line
    docker compose logs "$1" 2>/dev/null | grep "$2" | tail -1
}

# =============================================================================
echo -e "\n${BOLD}LSIE-MLF End-to-End Experiment Verification${NC}"
echo -e "${DIM}$(date -u '+%Y-%m-%d %H:%M:%S UTC')${NC}\n"

# --- Load .env if present ---
if [ -f .env ]; then
    set -a; source .env 2>/dev/null; set +a
fi

# =============================================================================
header "Stage 1 — Infrastructure"
# =============================================================================

# Container health
for svc in redis postgres; do
    if container_healthy "$svc"; then
        pass "$svc container healthy"
    else
        fail "$svc container NOT healthy"
    fi
done

for svc in worker orchestrator api stream_scrcpy; do
    if container_running "$svc"; then
        pass "$svc container running"
    else
        fail "$svc container NOT running"
    fi
done

# Schema exists
TABLE_COUNT=$(db_query "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" || echo "0")
if [ "${TABLE_COUNT:-0}" -ge 7 ]; then
    pass "Database schema initialized ($TABLE_COUNT tables)"
else
    fail "Database schema missing or incomplete ($TABLE_COUNT tables, expected ≥7)"
fi

# Seed data exists
ARM_COUNT=$(db_query "SELECT COUNT(*) FROM experiments WHERE experiment_id='greeting_line_v1';" || echo "0")
if [ "${ARM_COUNT:-0}" -ge 4 ]; then
    pass "Experiment seed data present ($ARM_COUNT arms)"
else
    fail "Experiment seed data missing ($ARM_COUNT arms, expected 4)"
fi

# =============================================================================
header "Stage 2 — Capture Pipeline"
# =============================================================================

# IPC pipes
if docker compose exec -T stream_scrcpy test -p /tmp/ipc/audio_stream.raw 2>/dev/null; then
    pass "Audio IPC pipe exists"
else
    warn "Audio IPC pipe not found (capture may still be initializing)"
fi

if docker compose exec -T stream_scrcpy test -p /tmp/ipc/video_stream.mkv 2>/dev/null; then
    pass "Video IPC pipe exists"
else
    warn "Video IPC pipe not found (video capture may be unavailable)"
fi

# scrcpy processes
if log_contains stream_scrcpy "Audio scrcpy launched"; then
    pass "Audio scrcpy instance launched"
else
    warn "Audio scrcpy launch not confirmed in logs"
fi

if log_contains stream_scrcpy "Video scrcpy launched"; then
    pass "Video scrcpy instance launched"
else
    warn "Video scrcpy launch not confirmed in logs"
fi

# =============================================================================
header "Stage 3 — Orchestrator Initialization"
# =============================================================================

if log_contains orchestrator "Session registered in Persistent Store"; then
    SESSION_LINE=$(log_last_match orchestrator "Session registered in Persistent Store")
    pass "Session registered"
    info "$SESSION_LINE"
else
    fail "Session NOT registered in Persistent Store"
fi

if log_contains orchestrator "Thompson Sampling selected arm"; then
    ARM_LINE=$(log_last_match orchestrator "Thompson Sampling selected arm")
    pass "Thompson Sampling arm selected"
    info "$ARM_LINE"
elif log_contains orchestrator "Thompson Sampling unavailable"; then
    warn "Thompson Sampling unavailable — using fallback arm"
    info "$(log_last_match orchestrator 'Thompson Sampling unavailable')"
else
    fail "No arm selection found in orchestrator logs"
fi

if log_contains orchestrator "AU12 normalizer initialized"; then
    pass "AU12 normalizer initialized (v3.0)"
    info "$(log_last_match orchestrator 'AU12 normalizer initialized')"
else
    warn "AU12 normalizer not yet initialized (waiting for first video frame)"
fi

if log_contains orchestrator "FFmpeg resampler started"; then
    pass "FFmpeg audio resampler running"
else
    fail "FFmpeg resampler NOT started"
fi

# =============================================================================
header "Stage 4 — Stimulus Injection"
# =============================================================================

if log_contains orchestrator "Stimulus injected at"; then
    STIM_LINE=$(log_last_match orchestrator "Stimulus injected at")
    pass "Stimulus injection recorded"
    info "$STIM_LINE"
else
    warn "Stimulus not yet injected (auto-trigger may still be waiting)"
    info "Auto-trigger delay: \${AUTO_STIMULUS_DELAY_S:-15}s"
    info "Manual trigger: curl -X POST http://localhost:8000/api/v1/stimulus"
fi

# =============================================================================
header "Stage 5 — Segment Processing"
# =============================================================================

SEGMENT_COUNT=$(docker compose logs worker 2>/dev/null | grep -c "Module D: processing" || echo "0")
if [ "$SEGMENT_COUNT" -gt 0 ]; then
    pass "Segments processed by Module D ($SEGMENT_COUNT segments)"
else
    warn "No segments processed yet (first segment arrives after 30s of audio)"
fi

if log_contains worker "Metrics persisted for"; then
    PERSIST_COUNT=$(docker compose logs worker 2>/dev/null | grep -c "Metrics persisted for" || echo "0")
    pass "Metrics persisted to PostgreSQL ($PERSIST_COUNT segments)"
else
    warn "No metrics persisted yet"
fi

# =============================================================================
header "Stage 6 — Database State"
# =============================================================================

SESSION_COUNT=$(db_query "SELECT COUNT(*) FROM sessions;" || echo "0")
info "Sessions: $SESSION_COUNT"

METRIC_COUNT=$(db_query "SELECT COUNT(*) FROM metrics;" || echo "0")
if [ "${METRIC_COUNT:-0}" -gt 0 ]; then
    pass "Metrics table has data ($METRIC_COUNT rows)"

    # Sample the latest metric
    LATEST=$(db_query "SELECT segment_id, au12_intensity, pitch_f0 FROM metrics ORDER BY created_at DESC LIMIT 1;" || echo "")
    if [ -n "$LATEST" ]; then
        info "Latest metric: $LATEST"
    fi
else
    warn "Metrics table is empty"
fi

TRANSCRIPT_COUNT=$(db_query "SELECT COUNT(*) FROM transcripts;" || echo "0")
info "Transcripts: $TRANSCRIPT_COUNT"

EVAL_COUNT=$(db_query "SELECT COUNT(*) FROM evaluations;" || echo "0")
info "Evaluations: $EVAL_COUNT"

# =============================================================================
header "Stage 7 — Thompson Sampling Posterior (THE MONEY CHECK)"
# =============================================================================

if log_contains worker "Thompson Sampling updated:"; then
    TS_LINE=$(log_last_match worker "Thompson Sampling updated:")
    pass "Thompson Sampling posterior updated!"
    info "$TS_LINE"

    # Extract reward value from log
    REWARD=$(echo "$TS_LINE" | grep -oP 'reward=\K[0-9.]+' || echo "?")
    P90=$(echo "$TS_LINE" | grep -oP 'p90=\K[0-9.]+' || echo "?")
    GATE=$(echo "$TS_LINE" | grep -oP 'gate=\K[0-9]+' || echo "?")
    FRAMES=$(echo "$TS_LINE" | grep -oP 'frames=\K[0-9]+' || echo "?")

    echo ""
    info "  Reward:  $REWARD"
    info "  P90:     $P90"
    info "  Gate:    $GATE (1=semantic match, 0=gated)"
    info "  Frames:  $FRAMES (expect ~135 at 30fps)"

elif log_contains worker "Thompson Sampling update SKIPPED"; then
    SKIP_LINE=$(log_last_match worker "Thompson Sampling update SKIPPED")
    warn "Thompson Sampling update was SKIPPED"
    info "$SKIP_LINE"

    if echo "$SKIP_LINE" | grep -q "no AU12 telemetry"; then
        info "→ Cause: stimulus was never injected or video capture failed"
    elif echo "$SKIP_LINE" | grep -q "invalid"; then
        info "→ Cause: insufficient AU12 frames in the measurement window"
    fi
else
    if [ "$QUICK" = false ]; then
        info "Waiting up to ${WAIT_TIMEOUT}s for first Thompson Sampling update..."
        ELAPSED=0
        while [ $ELAPSED -lt "$WAIT_TIMEOUT" ]; do
            if log_contains worker "Thompson Sampling updated:"; then
                TS_LINE=$(log_last_match worker "Thompson Sampling updated:")
                pass "Thompson Sampling posterior updated!"
                info "$TS_LINE"
                break
            fi
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            printf "\r  ${DIM}  ...waiting (%ds / %ds)${NC}" "$ELAPSED" "$WAIT_TIMEOUT"
        done
        echo ""

        if [ $ELAPSED -ge "$WAIT_TIMEOUT" ]; then
            fail "No Thompson Sampling update within ${WAIT_TIMEOUT}s"
            info "Run: bash scripts/debug_e2e.sh for diagnosis"
        fi
    else
        warn "No Thompson Sampling update found (--quick mode, skipped wait)"
    fi
fi

# Final posterior state
echo ""
info "Current experiment posteriors:"
db_query "SELECT arm, alpha_param, beta_param, ROUND(alpha_param/(alpha_param+beta_param)::numeric, 4) AS mean FROM experiments WHERE experiment_id='greeting_line_v1' ORDER BY arm;" || warn "Could not query experiments table"

# Check if any arm moved from prior
MOVED=$(db_query "SELECT COUNT(*) FROM experiments WHERE experiment_id='greeting_line_v1' AND (alpha_param != 1.0 OR beta_param != 1.0);" || echo "0")
echo ""
if [ "${MOVED:-0}" -gt 0 ]; then
    pass "═══ EXPERIMENT RUN CONFIRMED: $MOVED arm(s) updated from Beta(1,1) prior ═══"
else
    if [ "$FAIL_COUNT" -eq 0 ]; then
        warn "All arms still at Beta(1,1) prior — experiment may need more time"
    else
        fail "═══ EXPERIMENT RUN NOT CONFIRMED ═══"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}━━ Summary ━━${NC}"
echo -e "  ${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}  $FAIL_COUNT"
echo -e "  ${YELLOW}Warned:${NC} $WARN_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ] && [ "${MOVED:-0}" -gt 0 ]; then
    echo -e "${GREEN}${BOLD}  ✓ E2E experiment run verified successfully.${NC}"
    echo -e "${DIM}  The full chain fired: USB → IPC → FFmpeg → FaceMesh → AU12 →${NC}"
    echo -e "${DIM}  P90 → semantic gate → fractional Beta update → PostgreSQL.${NC}"
    exit 0
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  ⚠ Infrastructure healthy but posterior not yet updated.${NC}"
    echo -e "${DIM}  Wait for the first 30s segment to complete, or run with --wait 180.${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}  ✗ E2E verification failed. Run debug script for diagnosis:${NC}"
    echo -e "${DIM}    bash scripts/debug_e2e.sh${NC}"
    exit 1
fi