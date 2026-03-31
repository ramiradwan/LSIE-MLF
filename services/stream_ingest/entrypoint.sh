#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
IPC_DIR="/tmp/ipc"
VIDEO_PIPE="$IPC_DIR/video_stream.mkv"
AUDIO_PIPE="$IPC_DIR/audio_stream.raw"

# Resolve Host IP for Docker environment
HOST_IP=$(getent ahostsv4 host.docker.internal | head -n1 | awk '{print $1}')

# --- Functions ---

# 1. Initialize Named Pipes
setup_pipes() {
    echo "[setup] Initializing IPC directory and pipes..."
    mkdir -p "$IPC_DIR"
    
    # Clean and recreate pipes to avoid stale data
    rm -f "$VIDEO_PIPE" "$AUDIO_PIPE" || true
    mkfifo "$VIDEO_PIPE"
    mkfifo "$AUDIO_PIPE"
    
    # Shield AUDIO so it survives scrcpy restarts
    exec 3<> "$AUDIO_PIPE"
    
    # VIDEO remains unshielded so PyAV crashes trigger a scrcpy restart (fresh MKV header)
}

# 2. ADB Connectivity Check
wait_for_device() {
    echo "[stream_ingest] waiting for adb device..."
    local elapsed=0
    local timeout=60

    while ! adb devices | grep -q "device$"; do
        if (( elapsed >= timeout )); then
            echo "[stream_ingest] adb device timeout"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "[stream_ingest] device connected"
}

# 3. Dual-Stream Capture Loop
start_capture() {
    echo "[stream_ingest] Audio scrcpy launched"
    echo "[stream_ingest] Video scrcpy launched"
    
    while true; do
        # 1. Cleanup: Kill zombie ADB network tunnels from previous runs
        adb forward --remove-all 2>/dev/null || true
        adb reverse --remove-all 2>/dev/null || true

        # 2. Launch AUDIO Capture
        # Bypassed 'dd'. Writing directly to pipe. Added --audio-dup to keep stream alive.
        scrcpy \
            --no-video \
            --no-playback \
            --audio-codec=raw \
            --audio-buffer=30 \
            --audio-dup \
            --record="$AUDIO_PIPE" \
            --record-format=wav \
            --port=27100:27199 \
            --tunnel-host="$HOST_IP" \
            2>/tmp/scrcpy_audio.log &
        AUDIO_PID=$!

        # 3. Stagger Startup: Prevent concurrent ADB server pushes to the device
        sleep 4

        # 4. Launch VIDEO Capture
        scrcpy \
            --no-audio \
            --no-playback \
            --video-codec=h264 \
            --max-fps=30 \
            --record="$VIDEO_PIPE" \
            --record-format=mkv \
            --port=27200:27299 \
            --tunnel-host="$HOST_IP" \
            2>/tmp/scrcpy_video.log &
        VIDEO_PID=$!

        # 5. Monitor: Wait for either process to exit
        wait -n "$AUDIO_PID" "$VIDEO_PID" 2>/dev/null || true
        
        # 6. Cleanup & Restart
        echo "[stream_ingest] Pipeline break detected, cleaning up..."
        kill "$AUDIO_PID" "$VIDEO_PID" 2>/dev/null || true
        wait "$AUDIO_PID" "$VIDEO_PID" 2>/dev/null || true
        
        sleep 2
        wait_for_device
    done
}

# --- Execution ---

setup_pipes
wait_for_device
start_capture