# Synthetic capture replay fixtures

This directory is reserved for synthetic capture fixtures used by the worker
pipeline replay path. Fixtures are generated, not hand-authored, so CI can test
capture-dependent orchestration without a phone, USB, ADB, scrcpy, PyAV, live
IPC pipes, network TTS, or raw human biometric recordings.

## Generate a fixture

Use the deterministic generator from the repository root. Canonical and CI
fixtures should use the embedded speech backend; it is the CLI default and is
shown explicitly below so the backend contract is visible in logs and scripts:

```bash
python scripts/generate_capture_fixture.py tests/fixtures/capture/smoke \
  --segments 3 \
  --segment-duration-s 3.0 \
  --width 320 \
  --height 240 \
  --seed 1234 \
  --speech-backend embedded \
  --overwrite
```

For a production-like local replay artifact, keep the same deterministic speech
backend contract and increase only the duration/resolution, for example:

```bash
python scripts/generate_capture_fixture.py tests/fixtures/capture/baseline_30s \
  --segments 1 \
  --segment-duration-s 30.0 \
  --width 320 \
  --height 240 \
  --seed 1234 \
  --speech-backend embedded \
  --overwrite
```

The intended later post-merge baseline artifact should use synthetic-only media,
30 FPS video, 48 kHz mono PCM WAV audio, H.264-in-MKV video, the default neutral
mouth ratio (`0.55`), the default AU12 response (`tanh(6.0 * 0.16)`), and the
embedded speech backend metadata recorded by the generator. If the artifact
becomes large, store it outside git or via the repository's future LFS policy;
this task intentionally adds only the generator and tests.

## Fixture contents

Each fixture directory must contain exactly the replay contract files:

- `video.mkv` — H.264-compatible MKV at 30 FPS. The content is a deterministic
  synthetic-only frontal face animation designed to be accepted by the
  production `packages.ml_core.face_mesh.FaceMeshProcessor` path. Frames before
  each scripted stimulus hold neutral lip-corner geometry; frames after each
  stimulus widen the lip corners to generate an AU12 response. No metadata side
  channel or replay-only landmark injection is required.
- `audio.wav` — 48 kHz, mono, signed 16-bit little-endian PCM WAV. The audio is
  deterministic offline lexical speech synthesized from the literal
  `expected_greeting_text` values and placed at each scripted stimulus offset.
  By default, the generator uses its embedded `embedded-formant-phoneme-v1`
  synthesizer, which is deterministic and independent of host `PATH`. The
  optional `--speech-backend espeak-ng` path is an explicit opt-in only: it
  requires `espeak-ng` on `PATH`, fails clearly when unavailable, records the
  resolved binary/version metadata, and may produce different bytes across
  `espeak-ng` or FFmpeg versions. Neither backend calls network services or
  falls back to tone/noise placeholders.
- `stimulus_script.json` — JSON metadata consumed by `ReplayCaptureSource` and
  asserted by tests. Each row in `stimuli` contains:
  - `segment_index`
  - `stimulus_offset_s`
  - `expected_arm_id`
  - `expected_greeting_text`
  - `expected_peak_au12`
  - `expected_semantic_match`

The JSON also records media parameters (`fps`, `duration_s`, video dimensions,
audio format, `audio_synthesis`, `speech_backend`, and `segment_duration_s`) so
the replay source can validate the fixture before decoding. `speech_backend`
records the requested and used backend plus a stable embedded identifier/version
for canonical fixtures; explicit `espeak-ng` runs also record the detected binary
path, version, voice, gap, and FFmpeg conversion command metadata.

## Determinism expectations

Identical generator inputs using the embedded backend should produce
byte-identical `video.mkv`, `audio.wav`, and `stimulus_script.json` on the same
ffmpeg/libx264 version. The generator removes metadata, uses a fixed frame
cadence, writes canonical JSON, synthesizes speech deterministically from the
literal fixture text, and encodes with single-threaded bit-exact ffmpeg settings
to minimize nondeterminism.

There is no implicit `espeak-ng` auto-detection: installing or removing an
`espeak-ng` binary from `PATH` does not change canonical fixture bytes unless
`--speech-backend espeak-ng` is explicitly requested. For that explicit opt-in,
fixture bytes are tied to the resolved offline speech toolchain and the recorded
`speech_backend` metadata should travel with the artifact.

The integration tests generate temporary fixtures with `--speech-backend
embedded` and compare independent outputs with the same inputs before exercising
the orchestrator replay path.

## Replay audio contract

Live capture delivers continuous 48 kHz mono s16le PCM to the Orchestrator
Container, where `AudioResampler` uses FFmpeg to emit 16 kHz mono s16le PCM for
segment assembly. Replay fixtures keep the same 48 kHz WAV input contract.
Because `Orchestrator` binds `ReplayCaptureSource` as the audio-read surface in
replay mode, `ReplayCaptureSource` invokes FFmpeg internally with the same raw
48 kHz -> 16 kHz mono s16le contract and then serves cadence-correct 16 kHz
chunks. Replay must not use array stride decimation or any non-FFmpeg resampler.

## Using a fixture in the orchestrator

Set `REPLAY_CAPTURE_FIXTURE` to a fixture directory before constructing the
orchestrator:

```bash
REPLAY_CAPTURE_FIXTURE=tests/fixtures/capture/smoke \
python -m services.worker.run_orchestrator
```

When the variable is set, `Orchestrator` binds `ReplayCaptureSource` to both the
video surface (`get_latest_frame()`) and the audio-read surface (`read_chunk()`).
The live path remains the default when `REPLAY_CAPTURE_FIXTURE` is unset.

For fast deterministic tests, `REPLAY_CAPTURE_REALTIME=0` disables sleeps while
preserving the same source methods. Leave it unset for wall-clock replay cadence.
The replay path is a no-biometric/no-hardware guarantee: it never requires ADB,
USB device access, scrcpy launch logic, or live capture IPC setup.
