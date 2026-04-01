---
name: ipc-pipeline
description: IPC named pipe lifecycle and audio transport between Capture Container and ML Worker. Use when working on services/stream_ingest/, the entrypoint.sh script, audio resampling in orchestrator.py, or any POSIX pipe / FFmpeg subprocess code.
---

# IPC Pipe Lifecycle (§4.A.1, SPEC-AMEND-004)

## Dual-instance scrcpy architecture

Two separate scrcpy processes run concurrently in the Capture Container, each handling one media type. This avoids muxing issues and allows independent restart on failure.

- **Audio instance:** port range 27100:27199, writes to `/tmp/ipc/audio_stream.raw`
- **Video instance:** port range 27200:27299, writes to `/tmp/ipc/video_stream.mkv` (MKV streaming container)
- **Staggered startup:** 4-second delay between audio and video launch to prevent concurrent ADB server push collisions on the device.

## Pipe creation and open sequence

1. `mkdir -p /tmp/ipc` — ensure IPC directory exists.
2. `rm -f` both pipes, then `mkfifo` to recreate — avoids stale data from previous runs.
3. `exec 3<> /tmp/ipc/audio_stream.raw` — fd 3 shield keeps audio pipe open across scrcpy restarts. Non-blocking open decouples reader/writer.
4. Video pipe (`/tmp/ipc/video_stream.mkv`) is **unshielded** — when PyAV crashes, the broken pipe triggers a scrcpy restart which writes a fresh MKV header.
5. Shutdown: close fd 3, kill both scrcpy processes, delete pipe files.
6. Crash: pipe buffer fills, writes return EAGAIN, overflow silently discarded (§12).

## scrcpy flags

Audio instance: `--no-video --no-playback --audio-codec=raw --audio-buffer=30 --audio-dup --record=AUDIO_PIPE --record-format=wav --port=27100:27199 --tunnel-host=HOST_IP`.
Video instance: `--no-audio --no-playback --video-codec=h264 --max-fps=30 --record=VIDEO_PIPE --record-format=mkv --port=27200:27299 --tunnel-host=HOST_IP`.

Note: `--record` writes directly to the named pipe — no dd or fd redirection needed for video. Audio uses the fd 3 shield for resilience.

## Audio format

Input: PCM signed 16-bit little-endian (s16le), 48 kHz, mono (via `--audio-codec=raw`).
Output after resampling: PCM s16le, 16 kHz, mono.

## Video format

Input: H.264 in MKV container, up to 30 fps (via `--max-fps=30`).
Consumer: PyAV opens `/tmp/ipc/video_stream.mkv` in the orchestrator's VideoCapture thread. Frames decoded to numpy arrays for MediaPipe Face Mesh.

## FFmpeg resampling command (§4.C.2)

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i /tmp/ipc/audio_stream.raw -ar 16000 -f s16le -ac 1 pipe:1
```

FFmpeg runs as a persistent subprocess in the orchestrator container. On unexpected exit, restart within 1 second.

## USB reconnection (§12.2 Module A)

Poll every 2 seconds for 60 seconds. If device not found, graceful container termination. On reconnection, cleanup ADB forward/reverse tunnels before relaunching scrcpy instances.
