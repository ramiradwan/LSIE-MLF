#!/usr/bin/env bash
# =============================================================================
# LSIE-MLF End-to-End Debugging Helper
#
# Walks through the data flow path in dependency order and stops at the
# first broken link. Each check prints what it found, what it expected,
# and a concrete fix command.
#
# Usage:
#   bash scripts/debug_e2e.sh            # Full diagnosis
#   bash scripts/debug_e2e.sh --logs     # Also dump recent logs per service
#   bash scripts/debug_e2e.sh --fix      # Attempt automatic fixes where safe
#
# Requires: docker, docker compose
# =============================================================================

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

SHOW_LOGS=false
AUTO_FIX=false
for arg in "$@"; do
    case $arg in
        --logs) SHOW_LOGS=true ;;
        --fix)  AUTO_FIX=true ;;
    esac
done

ok()      { echo -e "  ${GREEN}OK${NC}   $1"; }
broken()  { echo -e "  ${RED}FAIL${NC} $1"; }
hint()    { echo -e "       ${YELLOW}→ Fix:${NC} $1"; }
detail()  { echo -e "       ${DIM}$1${NC}"; }
section() { echo -e "\n${CYAN}${BOLD}[$1] $2${NC}"; }
divider() { echo -e "${DIM}$(printf '%.0s─' {1..60})${NC}"; }

dump_logs() {
    if [ "$SHOW_LOGS" = true ]; then
        echo -e "       ${DIM}--- Last 10 lines of $1 ---${NC}"
        docker compose logs "$1" --tail=10 2>/dev/null | sed 's/^/       /'
        echo -e "       ${DIM}--- end ---${NC}"
    fi
}

if [ -f .env ]; then
    set -a; source .env 2>/dev/null; set +a
fi

DB_USER="${POSTGRES_USER:-lsie}"
DB_NAME="${POSTGRES_DB:-lsie_mlf}"

db_query() {
    docker compose exec -T postgres psql -U "$DB_USER" -d "$DB_NAME" -t -A -c "$1" 2>/dev/null
}

apply_schema_sql_directly() {
    local sql_file
    local ordered_schema_files=(
        "/docker-entrypoint-initdb.d/01-schema.sql"
        "/docker-entrypoint-initdb.d/03-encounter-log.sql"
        "/docker-entrypoint-initdb.d/03-physiology.sql"
        "/docker-entrypoint-initdb.d/04-metrics-observational-acoustics.sql"
    )

    for sql_file in "${ordered_schema_files[@]}"; do
        echo -e "       ${DIM}Applying $(basename "$sql_file")...${NC}"
        docker compose exec -T postgres psql -U "$DB_USER" -d "$DB_NAME" \
            -f "$sql_file" 2>&1 | sed 's/^/       /'
    done
}

echo -e "\n${BOLD}LSIE-MLF E2E Debugger${NC}"
echo -e "${DIM}Walking the data flow path to find the first broken link...${NC}"

# =============================================================================
section "1/10" "Docker Compose services"
# =============================================================================

ALL_UP=true
for svc in redis postgres stream_scrcpy worker orchestrator api; do
    STATUS=$(docker compose ps --format '{{.State}}' "$svc" 2>/dev/null || echo "missing")
    if echo "$STATUS" | grep -qi "running\|up"; then
        ok "$svc is running"
    else
        broken "$svc is $STATUS"
        ALL_UP=false

        # Service-specific hints
        case $svc in
            redis|postgres)
                hint "docker compose up -d $svc && docker compose logs $svc --tail=5"
                ;;
            stream_scrcpy)
                hint "Check USB device: adb devices"
                hint "docker compose up -d stream_scrcpy && docker compose logs stream_scrcpy --tail=10"
                ;;
            worker)
                hint "docker compose up -d worker && docker compose logs worker --tail=10"
                hint "Common cause: missing GPU driver or CUDA mismatch"
                ;;
            orchestrator)
                hint "docker compose up -d orchestrator && docker compose logs orchestrator --tail=10"
                hint "Common cause: import error in run_orchestrator.py or missing .env vars"
                ;;
            api)
                hint "docker compose up -d api && docker compose logs api --tail=10"
                ;;
        esac
        dump_logs "$svc"
    fi
done

if [ "$ALL_UP" = false ]; then
    echo ""
    broken "Some services are down. Fix them before continuing."
    hint "Quick restart: docker compose down && docker compose up -d"
    exit 1
fi

# =============================================================================
section "2/10" "PostgreSQL schema and seed data"
# =============================================================================

TABLE_COUNT=$(db_query "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" || echo "0")
if [ "${TABLE_COUNT:-0}" -ge 7 ]; then
    ok "Schema present ($TABLE_COUNT tables)"
else
    broken "Schema missing or incomplete ($TABLE_COUNT tables)"
    hint "Check if data/sql/ is mounted:"
    hint "  docker compose exec postgres ls /docker-entrypoint-initdb.d/"
    hint "If empty, the volume mount in docker-compose.yml is wrong."
    hint "If files exist but tables don't, the Persistent Store was already initialized."
    hint "Nuclear fix: docker compose down -v && docker compose up -d"
    detail "(Warning: -v deletes all data including pg-data volume)"

    if [ "$AUTO_FIX" = true ]; then
        echo -e "       ${YELLOW}AUTO-FIX:${NC} Executing ordered schema SQL directly..."
        apply_schema_sql_directly
    fi
fi

ACOUSTIC_COLUMN_COUNT=$(db_query "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='public' AND table_name='metrics' AND column_name IN ('f0_valid_measure','f0_valid_baseline','perturbation_valid_measure','perturbation_valid_baseline','voiced_coverage_measure_s','voiced_coverage_baseline_s','f0_mean_measure_hz','f0_mean_baseline_hz','f0_delta_semitones','jitter_mean_measure','jitter_mean_baseline','jitter_delta','shimmer_mean_measure','shimmer_mean_baseline','shimmer_delta');" || echo "0")
if [ "${ACOUSTIC_COLUMN_COUNT:-0}" -eq 15 ]; then
    ok "Observational acoustic metrics migration present (15/15 columns)"
else
    broken "Observational acoustic metrics migration missing ($ACOUSTIC_COLUMN_COUNT of 15 columns)"
    hint "Existing databases must apply ordered schema files before writer services resume:"
    hint "  01-schema.sql → 03-encounter-log.sql → 03-physiology.sql → 04-metrics-observational-acoustics.sql"
    hint "Then restart worker, orchestrator, and api so new writes see the upgraded schema"

    if [ "$AUTO_FIX" = true ]; then
        echo -e "       ${YELLOW}AUTO-FIX:${NC} Applying ordered schema SQL for existing database..."
        apply_schema_sql_directly
    fi
fi

ARM_COUNT=$(db_query "SELECT COUNT(*) FROM experiments WHERE experiment_id='greeting_line_v1';" || echo "0")
if [ "${ARM_COUNT:-0}" -ge 4 ]; then
    ok "Experiment arms seeded ($ARM_COUNT arms)"
else
    broken "Experiment arms missing ($ARM_COUNT of 4)"
    hint "Seed manually:"
    hint "  docker compose exec -T postgres psql -U $DB_USER -d $DB_NAME -f /docker-entrypoint-initdb.d/02-seed-experiments.sql"

    if [ "$AUTO_FIX" = true ]; then
        echo -e "       ${YELLOW}AUTO-FIX:${NC} Seeding experiments..."
        docker compose exec -T postgres psql -U "$DB_USER" -d "$DB_NAME" \
            -f /docker-entrypoint-initdb.d/02-seed-experiments.sql 2>&1 | sed 's/^/       /'
    fi
fi

# =============================================================================
section "3/10" "USB device and scrcpy capture"
# =============================================================================

if docker compose exec -T stream_scrcpy adb devices 2>/dev/null | grep -q "device$"; then
    ok "Android device detected via ADB"
else
    broken "No Android device connected"
    hint "Connect device via USB and enable USB debugging"
    hint "Verify on host: adb devices"
    hint "Check Docker USB passthrough: ls -la /dev/bus/usb/"
fi

if docker compose exec -T stream_scrcpy test -p /tmp/ipc/audio_stream.raw 2>/dev/null; then
    ok "Audio IPC pipe exists"
else
    broken "Audio IPC pipe missing"
    hint "scrcpy may have failed to start. Check: docker compose logs stream_scrcpy --tail=20"
    dump_logs stream_scrcpy
fi

# =============================================================================
section "4/10" "Orchestrator session and arm selection"
# =============================================================================

if docker compose logs orchestrator 2>/dev/null | grep -q "Session registered"; then
    ok "Session registered in PostgreSQL"
else
    broken "Session registration failed"
    hint "Check orchestrator can reach postgres:"
    hint "  docker compose exec orchestrator python3.11 -c \"import psycopg2; print('OK')\""
    hint "Check .env has POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB"
    dump_logs orchestrator
fi

if docker compose logs orchestrator 2>/dev/null | grep -q "Thompson Sampling selected arm"; then
    ARM_LINE=$(docker compose logs orchestrator 2>/dev/null | grep "Thompson Sampling selected arm" | tail -1)
    ok "Arm selected via Thompson Sampling"
    detail "$ARM_LINE"
elif docker compose logs orchestrator 2>/dev/null | grep -q "Thompson Sampling unavailable"; then
    broken "Thompson Sampling failed — using fallback arm"
    hint "Check experiments table has seed data (Step 2 above)"
    hint "Check psycopg2 and scipy are installed in the worker image"
    dump_logs orchestrator
else
    broken "No arm selection in logs"
    hint "Orchestrator may have crashed before reaching arm selection"
    dump_logs orchestrator
fi

# =============================================================================
section "5/10" "FFmpeg audio resampler"
# =============================================================================

if docker compose logs orchestrator 2>/dev/null | grep -q "FFmpeg resampler started"; then
    ok "FFmpeg 48→16 kHz resampler running"
else
    broken "FFmpeg resampler not started"
    hint "Check the audio IPC Pipe exists (Step 3) and FFmpeg is installed:"
    hint "  docker compose exec orchestrator which ffmpeg"
    dump_logs orchestrator
fi

# =============================================================================
section "6/10" "Video capture and AU12 initialization"
# =============================================================================

if docker compose logs orchestrator 2>/dev/null | grep -q "Video capture thread started"; then
    ok "Video capture thread running"
elif docker compose logs orchestrator 2>/dev/null | grep -q "Video capture unavailable"; then
    broken "Video capture unavailable"
    hint "Check IPC Pipe video endpoint: docker compose exec orchestrator test -p /tmp/ipc/video_stream.mkv"
    hint "Check PyAV installed: docker compose exec orchestrator python3.11 -c 'import av; print(av.__version__)'"
else
    broken "No video capture log entry"
    dump_logs orchestrator
fi

if docker compose logs orchestrator 2>/dev/null | grep -q "AU12 normalizer initialized"; then
    ALPHA_LINE=$(docker compose logs orchestrator 2>/dev/null | grep "AU12 normalizer initialized" | tail -1)
    ok "AU12 normalizer initialized"
    detail "$ALPHA_LINE"
    if echo "$ALPHA_LINE" | grep -q "6.0"; then
        ok "α_scale = 6.0"
    else
        broken "α_scale is NOT 6.0 — wrong AU12Normalizer version"
        hint "Check packages/ml_core/au12.py has DEFAULT_ALPHA_SCALE = 6.0"
    fi
else
    broken "AU12 normalizer not initialized (no video frames received yet)"
    hint "Video capture must be working (Step 6) for AU12 to initialize"
fi

# =============================================================================
section "7/10" "Stimulus injection"
# =============================================================================

if docker compose logs orchestrator 2>/dev/null | grep -q "Stimulus injected at"; then
    STIM_LINE=$(docker compose logs orchestrator 2>/dev/null | grep "Stimulus injected at" | tail -1)
    ok "Stimulus injection recorded"
    detail "$STIM_LINE"
else
    broken "Stimulus never injected — calibration still running"
    echo ""
    detail "Without stimulus injection:"
    detail "  - _is_calibrating stays True"
    detail "  - _au12_series stays empty (no frames accumulated)"
    detail "  - _stimulus_time stays None"
    detail "  - persist_metrics SKIPS Thompson Sampling update every time"
    echo ""
    hint "Option A (auto-trigger): set AUTO_STIMULUS_DELAY_S=15 in .env and restart orchestrator"
    hint "Option B (manual): curl -X POST http://localhost:8000/api/v1/stimulus"
    hint "Option C (check auto-trigger): docker compose logs orchestrator | grep 'Auto-trigger'"
fi

# =============================================================================
section "8/10" "Celery task dispatch and processing"
# =============================================================================

DISPATCH_COUNT=$(docker compose logs orchestrator 2>/dev/null | grep -c "dispatch segment" || echo "0")
PROCESS_COUNT=$(docker compose logs worker 2>/dev/null | grep -c "Module D: processing" || echo "0")

if [ "$PROCESS_COUNT" -gt 0 ]; then
    ok "ML Worker processed $PROCESS_COUNT segment(s)"
elif [ "$DISPATCH_COUNT" -gt 0 ]; then
    broken "Orchestrator Container dispatched $DISPATCH_COUNT segment(s) but ML Worker processed 0"
    hint "Check Celery connection to the Message Broker:"
    hint "  docker compose exec worker celery -A services.worker.celery_app inspect ping"
    hint "Common causes: Message Broker URL mismatch or JSON serialization error in binary payload fields"
    dump_logs worker
else
    broken "No segments dispatched yet"
    hint "First segment requires 30s of audio accumulation after FFmpeg starts"
    hint "If FFmpeg is running (Step 5), wait 30-45 seconds and re-run this script"
fi

# Check for serialization errors in binary payload transport.
if docker compose logs worker 2>/dev/null | grep -qi "not JSON serializable\|TypeError.*bytes"; then
    broken "JSON serialization error detected: raw bytes reached a JSON payload"
    hint "The Orchestrator Container must base64-encode _audio_data and _frame_data"
    hint "Check services/worker/pipeline/serialization.py exists"
    hint "Check Orchestrator Container assembly imports and calls encode_bytes_fields()"
fi

# =============================================================================
section "9/10" "Metrics persistence"
# =============================================================================

METRIC_ROWS=$(db_query "SELECT COUNT(*) FROM metrics;" || echo "0")
if [ "${METRIC_ROWS:-0}" -gt 0 ]; then
    ok "Metrics table has $METRIC_ROWS row(s)"

    # Check for null AU12 (indicates frame data not reaching Module D)
    NULL_AU12=$(db_query "SELECT COUNT(*) FROM metrics WHERE au12_intensity IS NULL;" || echo "0")
    TOTAL=$(db_query "SELECT COUNT(*) FROM metrics;" || echo "1")
    if [ "$NULL_AU12" = "$TOTAL" ]; then
        broken "All AU12 values are NULL — frame data not reaching Module D"
        hint "Check _frame_data is being attached in assemble_segment()"
        hint "Check base64 encoding/decoding is working (serialization.py)"
    else
        ok "AU12 intensity values present ($((TOTAL - NULL_AU12)) of $TOTAL non-null)"
    fi
else
    broken "Metrics table is empty"
    hint "persist_metrics may be failing. Check ML Worker logs:"
    hint "  docker compose logs worker | grep -i 'error\|fail\|exception' | tail -10"
    dump_logs worker
fi

# Check for missing reward telemetry forwarding.
if docker compose logs worker 2>/dev/null | grep -q "SKIPPED (no AU12 telemetry)"; then
    SKIP_COUNT=$(docker compose logs worker 2>/dev/null | grep -c "SKIPPED (no AU12 telemetry)" || echo "0")
    broken "Thompson Sampling skipped $SKIP_COUNT time(s) due to missing AU12 telemetry"
    echo ""
    detail "This usually means one of:"
    detail "  1. process_segment() is not forwarding _au12_series to persist_metrics"
    detail "  2. Stimulus was never injected so _au12_series is always empty"
    detail "  3. _stimulus_time is None because record_stimulus_injection() was never called"
    echo ""
    hint "Check inference.py has the _FORWARD_FIELDS loop before persist_metrics.delay()"
    hint "Check stimulus was injected (Step 7 above)"
fi

# =============================================================================
section "10/10" "Thompson Sampling posterior update"
# =============================================================================

if docker compose logs worker 2>/dev/null | grep -q "Thompson Sampling updated:"; then
    TS_LINE=$(docker compose logs worker 2>/dev/null | grep "Thompson Sampling updated:" | tail -1)
    ok "Posterior updated!"
    detail "$TS_LINE"
    echo ""

    # Show current posteriors
    ok "Current experiment state:"
    db_query "SELECT arm, alpha_param, beta_param, ROUND(alpha_param/(alpha_param+beta_param)::numeric, 4) AS mean FROM experiments WHERE experiment_id='greeting_line_v1' ORDER BY arm;" \
        | sed 's/^/       /' || true

    MOVED=$(db_query "SELECT COUNT(*) FROM experiments WHERE experiment_id='greeting_line_v1' AND (alpha_param != 1.0 OR beta_param != 1.0);" || echo "0")
    if [ "${MOVED:-0}" -gt 0 ]; then
        echo ""
        echo -e "  ${GREEN}${BOLD}✓ E2E EXPERIMENT RUN CONFIRMED${NC}"
        echo -e "  ${DIM}$MOVED arm(s) have posteriors different from Beta(1,1) prior.${NC}"
    fi
else
    broken "No Thompson Sampling update found"
    echo ""
    detail "The full chain must fire for a posterior update:"
    detail "  USB → scrcpy → IPC pipe → FFmpeg → orchestrator → stimulus injection →"
    detail "  AU12 accumulation → segment assembly → Celery dispatch → process_segment →"
    detail "  field forwarding → persist_metrics → compute_reward → fractional TS update"
    echo ""
    detail "Work backward from the last successful step above to find the break."
fi

# =============================================================================
# Final summary
# =============================================================================
divider
echo ""
if docker compose logs worker 2>/dev/null | grep -q "Thompson Sampling updated:"; then
    echo -e "${GREEN}${BOLD}  Diagnosis: system is working end-to-end.${NC}"
else
    echo -e "${YELLOW}${BOLD}  Diagnosis: review the FAIL items above in order.${NC}"
    echo -e "${DIM}  The first FAIL in the chain is usually the root cause.${NC}"
    echo -e "${DIM}  Fix it, restart the affected service, and re-run this script.${NC}"
fi
echo ""
