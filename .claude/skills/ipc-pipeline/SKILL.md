---
name: ipc-pipeline
description: IPC named pipe lifecycle and audio transport between Capture Container and ML Worker. Use when working on services/stream_ingest/, the entrypoint.sh script, audio resampling in orchestrator.py, or any POSIX pipe / FFmpeg subprocess code.
---

# IPC Pipe Lifecycle (§4.A.1)

## Creation and open sequence

1. `mkfifo /tmp/ipc/audio_stream.raw` — remove and recreate if exists.
2. `exec 3<> /tmp/ipc/audio_stream.raw` — non-blocking open decoupling reader/writer.
3. scrcpy stdout → dd writing to fd 3.
4. ML Worker reads `/tmp/ipc/audio_stream.raw` when FFmpeg subprocess initializes.
5. Shutdown: close fd 3, delete pipe file.
6. Crash: pipe buffer fills, writes return EAGAIN, overflow silently discarded.

## scrcpy flags

`--audio-codec=raw` (force PCM), `--audio-buffer=30` (30ms), `--audio-dup` + `--no-audio-playback` (mirror without local playback).

## Audio format

Input: PCM signed 16-bit little-endian (s16le), 48 kHz, mono.
Output after resampling: PCM s16le, 16 kHz, mono.

## FFmpeg resampling command (§4.C.2)

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i /tmp/ipc/audio_stream.raw -ar 16000 -f s16le -ac 1 pipe:1
```

FFmpeg runs as a persistent subprocess. On unexpected exit, restart within 1 second.

## USB reconnection (§12.2 Module A)

Poll every 2 seconds for 60 seconds. If device not found, graceful container termination.
