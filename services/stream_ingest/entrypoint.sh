#!/usr/bin/env bash
# =============================================================================
# Capture Container Entrypoint — §4.A Module A Hardware & Transport
# REVISED: Stage 2 Remediation (Gap G-03)
#
# Dual-sink scrcpy v3.3.4 pipeline:
#   - Instance 1 (audio): raw PCM s16le 48kHz → IPC Pipe (fd 3)
#   - Instance 2 (video): H.264 in MKV container → video IPC Pipe
#
# IPC Pipe Lifecycle (§4.A.1):
#   1. Create audio + video IPC Pipes with mkfifo
#   2. Non-blocking open audio pipe with exec 3<>
#   3. Launch scrcpy audio instance → dd → fd 3
#   4. Launch scrcpy video instance → --record to video pipe (MKV)
#   5. On shutdown: close fds, kill processes, delete pipes
#   6. On USB loss: poll reconnection every 2s for 60s
#
# Key v3.3.4 changes from v3.1:
#   - --no-playback replaces deprecated --no-display (headless mode)
#   - --no-video / --no-audio split enables independent sinks
#   - --video-codec=h264 + --max-fps=30 for bandwidth control
#   - --record-format=mkv is a streaming container safe for named pipes
#
# Audio format: raw PCM s16le 48kHz mono (§2 step 1)
# Video format: H.264 wrapped in MKV (streaming container, no seek)
# =============================================================================

set -euo pipefail

IPC_PIPE="/tmp/ipc/audio_stream.raw"
VIDEO_PIPE="/tmp/ipc/video_stream.mkv"
USB_RETRY_INTERVAL=2
USB_RETRY_MAX=60
SCRCPY_AUDIO_BUFFER=30
SCRCPY_MAX_FPS=30

# Track PIDs for cleanup
AUDIO_PID=""
VIDEO_PID=""

# --- Step 1: Create IPC Pipes (§4.A.1 step 1) ---
setup_pipes() {
    # Audio pipe (existing)
    if [ -p "$IPC_PIPE" ]; then
        rm -f "$IPC_PIPE"
    fi
    mkfifo "$IPC_PIPE"
    echo "[stream_ingest] Created audio IPC pipe at $IPC_PIPE"

    # Video pipe (Gap G-03 remediation)
    if [ -p "$VIDEO_PIPE" ]; then
        rm -f "$VIDEO_PIPE"
    fi
    mkfifo "$VIDEO_PIPE"
    echo "[stream_ingest] Created video IPC pipe at $VIDEO_PIPE"
}

# --- Step 2: Non-blocking open audio pipe (§4.A.1 step 2) ---
open_audio_pipe() {
    exec 3<>"$IPC_PIPE"
    echo "[stream_ingest] Opened audio pipe fd 3 (non-blocking)"
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

# --- Launch dual scrcpy capture (§4.A.1 steps 3–4, Gap G-03) ---
start_capture() {
    echo "[stream_ingest] Starting dual scrcpy capture (audio + video)"

    while true; do
        # --- Audio Instance ---
        # §4.A.1: --no-video isolates audio-only stream
        # --no-playback: headless (v3.3.4 replaces deprecated --no-display)
        # --audio-codec=raw: force PCM s16le 48kHz (§2 step 1)
        # stdout → dd → fd 3 (IPC Pipe)
        # §12 Queue overload A: silent discard via dd block alignment
        scrcpy \
            --no-video \
            --no-playback \
            --audio-codec=raw \
            --audio-buffer="$SCRCPY_AUDIO_BUFFER" \
            2>/dev/null | dd of=/proc/self/fd/3 bs=3840 obs=3840 iflag=fullblock 2>/dev/null &
        AUDIO_PID=$!
        echo "[stream_ingest] Audio scrcpy launched (PID $AUDIO_PID)"

        # --- Video Instance (Gap G-03 remediation) ---
        # --no-audio: isolates video-only stream (avoids audio mux conflict)
        # --no-playback: headless Docker (no X11/Wayland)
        # --video-codec=h264: standard AVC, compatible with PyAV (§4.D.2)
        # --max-fps=30: cap frame rate to preserve GPU resources
        # --record: write to video FIFO pipe in MKV streaming container
        # MKV is a streaming format — does not require seek, safe for pipes
        scrcpy \
            --no-audio \
            --no-playback \
            --video-codec=h264 \
            --max-fps="$SCRCPY_MAX_FPS" \
            --record="$VIDEO_PIPE" \
            --record-format=mkv \
            2>/dev/null &
        VIDEO_PID=$!
        echo "[stream_ingest] Video scrcpy launched (PID $VIDEO_PID)"

        # Wait for either process to exit
        wait -n "$AUDIO_PID" "$VIDEO_PID" 2>/dev/null || true

        echo "[stream_ingest] scrcpy process exited, killing peers..."
        kill "$AUDIO_PID" 2>/dev/null || true
        kill "$VIDEO_PID" 2>/dev/null || true
        wait "$AUDIO_PID" 2>/dev/null || true
        wait "$VIDEO_PID" 2>/dev/null || true

        echo "[stream_ingest] Both processes stopped, checking USB..."

        # §12 Hardware loss A: poll 2s for 60s then restart
        wait_for_device

        echo "[stream_ingest] Reconnected, restarting capture..."
    done
}

# --- Graceful shutdown (§4.A.1 step 4) ---
cleanup() {
    echo "[stream_ingest] Shutting down..."
    # Kill scrcpy processes
    for pid_var in AUDIO_PID VIDEO_PID; do
        local pid="${!pid_var}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    # §4.A.1 step 4: close fd 3, delete pipe files
    exec 3>&- 2>/dev/null || true
    rm -f "$IPC_PIPE"
    rm -f "$VIDEO_PIPE"
    echo "[stream_ingest] Pipes closed and removed."
    exit 0
}

trap cleanup SIGTERM SIGINT

# --- Main ---
setup_pipes
open_audio_pipe
wait_for_device
start_capture