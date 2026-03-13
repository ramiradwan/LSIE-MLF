#!/usr/bin/env bash
# =============================================================================
# Capture Container Entrypoint — §4.A Module A Hardware & Transport
#
# IPC Pipe Lifecycle (§4.A.1):
#   1. Create IPC Pipe with mkfifo
#   2. Non-blocking open with exec 3<>
#   3. Launch scrcpy with raw PCM output → write to fd 3
#   4. On shutdown: close fd 3, delete pipe file
#   5. On USB loss: poll reconnection every 2s for 60s
#
# Audio format: raw PCM s16le 48kHz mono (§2 step 1)
# =============================================================================

set -euo pipefail

IPC_PIPE="/tmp/ipc/audio_stream.raw"
USB_RETRY_INTERVAL=2
USB_RETRY_MAX=60
SCRCPY_AUDIO_BUFFER=30

# Track scrcpy PID for cleanup
SCRCPY_PID=""

# --- Step 1: Create IPC Pipe (§4.A.1 step 1) ---
setup_pipe() {
    if [ -p "$IPC_PIPE" ]; then
        rm -f "$IPC_PIPE"
    fi
    mkfifo "$IPC_PIPE"
    echo "[stream_ingest] Created IPC pipe at $IPC_PIPE"
}

# --- Step 2: Non-blocking open (§4.A.1 step 2) ---
open_pipe() {
    exec 3<>"$IPC_PIPE"
    echo "[stream_ingest] Opened pipe fd 3 (non-blocking)"
}

# --- Wait for USB device (§12.2 Module A error handling) ---
# §12 Hardware loss A: poll 2s for 60s then restart
wait_for_device() {
    local elapsed=0
    while ! adb devices | grep -q "device$"; do
        if [ "$elapsed" -ge "$USB_RETRY_MAX" ]; then
            echo "[stream_ingest] USB device not found after ${USB_RETRY_MAX}s. Exiting."
            exit 1
        fi
        echo "[stream_ingest] Waiting for USB device... (${elapsed}s)"
        sleep "$USB_RETRY_INTERVAL"
        elapsed=$((elapsed + USB_RETRY_INTERVAL))
    done
    echo "[stream_ingest] USB device connected."
}

# --- Launch scrcpy audio capture (§4.A.1 steps 3–4) ---
start_audio_capture() {
    # §4.A.1 flags:
    #   --audio-codec=raw     → bypass AAC/Opus, force raw PCM s16le 48kHz
    #   --audio-buffer=30     → 30ms internal buffer
    #   --audio-dup           → mirror audio to container
    #   --no-audio-playback   → suppress local playback
    #   --no-video            → audio-only capture (reduce bandwidth)
    #
    # §4.A.1 step 3: scrcpy stdout → dd → fd 3 (IPC Pipe)
    # §12 Worker crash A: EAGAIN on pipe overflow (non-blocking fd)
    # §12 Queue overload A: silent discard via dd obs/ibs alignment

    echo "[stream_ingest] Starting scrcpy audio capture → IPC Pipe"

    while true; do
        scrcpy \
            --audio-codec=raw \
            --audio-buffer="$SCRCPY_AUDIO_BUFFER" \
            --audio-dup \
            --no-audio-playback \
            --no-video \
            2>/dev/null | dd of=/proc/self/fd/3 bs=3840 obs=3840 iflag=fullblock 2>/dev/null &
        SCRCPY_PID=$!

        echo "[stream_ingest] scrcpy launched (PID $SCRCPY_PID)"

        # Wait for scrcpy to exit
        wait "$SCRCPY_PID" || true

        echo "[stream_ingest] scrcpy exited, checking USB..."

        # §12 Hardware loss A: poll 2s for 60s then restart
        wait_for_device

        echo "[stream_ingest] Reconnected, restarting capture..."
    done
}

# --- Graceful shutdown (§4.A.1 step 4) ---
cleanup() {
    echo "[stream_ingest] Shutting down..."
    # Kill scrcpy if running
    if [ -n "$SCRCPY_PID" ] && kill -0 "$SCRCPY_PID" 2>/dev/null; then
        kill "$SCRCPY_PID" 2>/dev/null || true
        wait "$SCRCPY_PID" 2>/dev/null || true
    fi
    # §4.A.1 step 4: close fd 3, delete pipe file
    exec 3>&- 2>/dev/null || true
    rm -f "$IPC_PIPE"
    echo "[stream_ingest] Pipe closed and removed."
    exit 0
}

trap cleanup SIGTERM SIGINT

# --- Main ---
setup_pipe
open_pipe
wait_for_device
start_audio_capture
