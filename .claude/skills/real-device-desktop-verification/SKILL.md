---
name: real-device-desktop-verification
description: Debug and verify the v4 desktop runtime against a connected Android device using real ADB/scrcpy observations. Use when investigating desktop capture, Android playback state, Operator Console open/close behavior, CLI E2E checks, or hardware-dependent regressions that mocks cannot prove.
---

# Real-Device Desktop Verification

## When to invoke

Use this skill when a change depends on behavior from a real Android device or the Windows desktop process graph, especially:

- `capture_supervisor`, scrcpy, ADB, or transient capture file lifecycle changes.
- Desktop app launch/close behavior (`python -m services.desktop_app`).
- Operator API runtime checks (`python -m services.desktop_app --operator-api`).
- CLI E2E checks that need the loopback API backed by the real ProcessGraph.
- Playback, audio capture, visual capture, or process-teardown regressions.

Do not rely on this skill as a replacement for unit tests. Use it to establish real-world behavior, then add the smallest automated coverage that can guard the confirmed code path.

## Safety constraints

- Do not guess patches for Android media behavior. Reproduce the state transition with real ADB/scrcpy observations first.
- Do not add `--audio-dup` to the desktop capture audio command. It preserves Android playback but can produce silent captured audio for the app.
- Do not use post-teardown media key events as a fix for playback-stop bugs. They mask the cause and can alter operator/device state.
- Do not run destructive ADB commands, factory resets, app data clears, package uninstalls, or broad process kills without explicit user approval.
- Treat a paused/still source video as an invalid baseline for capture/playback-preservation conclusions.
- Prefer app code paths and project resolvers (`find_executable`) over shell PATH assumptions for `adb`, `scrcpy`, and `ffmpeg`.

## Baseline observations

Before launching the desktop runtime, record all three dimensions:

1. **Foreground app** — `adb shell dumpsys activity activities`; confirm the target app is resumed.
2. **Audio state** — `adb shell pidof <package>` plus `adb shell dumpsys audio`; confirm the target app has an `AudioPlaybackConfiguration` with `state:started` when testing playback preservation.
3. **Visual motion** — take multiple `adb exec-out screencap -p` samples over time and compare hashes; more than one unique hash indicates visible motion.

For TikTok-specific checks, the package is usually `com.zhiliaoapp.musically`. Do not infer TikTok playback from unrelated media sessions such as YouTube Music.

## Reusable probe skeleton

Use Python scripts for repeatable checks instead of ad-hoc shell pipelines. Resolve tools the same way the app does:

```python
from services.desktop_app.os_adapter import find_executable

ADB = find_executable("adb", env_override="LSIE_ADB_PATH")
SCRCPY = find_executable("scrcpy", env_override="LSIE_SCRCPY_PATH")
FFMPEG = find_executable("ffmpeg", env_override="LSIE_FFMPEG_PATH")
```

Core helpers:

- `foreground()` — parse `dumpsys activity activities` for the target package.
- `audio_started()` — parse `dumpsys audio` for target app PID lines with `state:started`.
- `visual_unique()` — hash 3-5 screenshots with a short delay between samples.
- `ensure_playing()` — only for manual test setup; if baseline is paused, tap once and re-check. Do not use this as the production fix.

## Desktop open/close verification

For full GUI checks:

1. Ensure the target app is playing and visually moving.
2. Launch `uv run python -m services.desktop_app` with stdout/stderr redirected to a temp log.
3. Wait for graph startup and capture stabilization.
4. Re-check foreground app, audio state, and visual motion.
5. Close the Operator Console normally by posting `WM_CLOSE` to the `LSIE-MLF Operator Console` window. Avoid force-killing the desktop process for normal-close evidence.
6. Re-check at multiple intervals, e.g. 3s, 8s, 15s, 25s, and 40s after close.
7. Confirm the desktop process exits with code `0` and the log has no traceback or retained capture artifact error.

For operator API runtime checks, use `uv run python -m services.desktop_app --operator-api` and verify no PySide Operator Console window appears while CLI/API calls work.

## Component isolation pattern

When the full desktop close changes device state, isolate Android-facing components one at a time with the same flags the app uses:

- Audio scrcpy:
  - `--no-video`
  - `--no-playback`
  - `--no-window`
  - `--audio-codec=raw`
  - `--audio-source=playback`
  - `--audio-buffer=30`
  - `--record=<audio_stream.wav>`
  - `--record-format=wav`
  - `--port=27100:27199`
- Video scrcpy:
  - `--no-audio`
  - `--no-playback`
  - `--no-window`
  - `--video-codec=h264`
  - `--max-fps=30`
  - `--record=<video_stream.mkv>`
  - `--record-format=mkv`
  - `--port=27200:27299`
- GPU live visual stream:
  - `adb exec-out screenrecord --output-format=h264 --time-limit=180 --size=540x960 --bit-rate=2000000 -`
  - decode through ffmpeg if needed.

For each component, record `before`, `during`, and `after teardown` audio/motion state. Only treat a component as causal if the baseline was playing/moving and the state transition occurs immediately after that component's teardown.

## Known Android media teardown finding

For scrcpy 3.3.4 on the tested Pixel device, terminating the full audio scrcpy/ADB process tree while using `--audio-source=playback` paused TikTok playback. Terminating only the scrcpy root process and allowing its ADB/server child to unwind preserved playback. Preserve this distinction when changing `SupervisedProcess` or `capture_supervisor` shutdown behavior.

## Evidence to report

When reporting a real-device verification result, include:

- Device/app baseline: package, foreground activity, audio `state`, visual hash uniqueness.
- Runtime command used: full GUI or operator API runtime.
- Close/teardown method: normal `WM_CLOSE`, API shutdown, root-only process terminate, full tree terminate, etc.
- State checkpoints: before launch, after launch, after close/teardown intervals, after process exit.
- Desktop exit code and whether logs contained tracebacks or retained capture artifacts.
- Any invalidated run and why, e.g. baseline was paused or the app was force-killed.

## Follow-up expectations

After confirming root cause:

1. Make the smallest code change that preserves the observed good path.
2. Add focused tests around the code contract even if the real-device behavior itself cannot be unit-tested.
3. Run focused unit/static checks.
4. Repeat the real-device verification before claiming the bug is fixed.
