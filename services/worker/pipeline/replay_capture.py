"""Replay capture source for deterministic synthetic capture fixtures.

``ReplayCaptureSource`` is a drop-in opt-in source for the orchestrator's
capture surfaces:

* video: ``start()``, ``get_latest_frame()``, ``stop()``, ``is_running``
* audio: ``start()``, ``read_chunk(num_bytes)``, ``stop()``

The fixture itself remains close to the live capture contract: H.264 video in an
MKV container and 48 kHz mono PCM s16le WAV audio.  Because the current
orchestrator binds replay as the object behind ``audio_resampler``, this source
runs the same FFmpeg 48 kHz raw PCM -> 16 kHz mono s16le contract internally and
serves the resampled bytes at live cadence.  It never performs stride/array
sample decimation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FIXTURE_VIDEO_NAME: str = "video.mkv"
FIXTURE_AUDIO_NAME: str = "audio.wav"
FIXTURE_SCRIPT_NAME: str = "stimulus_script.json"
REPLAY_FPS: int = 30
FIXTURE_AUDIO_SAMPLE_RATE_HZ: int = 48_000
ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ: int = 16_000
SAMPLE_WIDTH_BYTES: int = 2
CHANNELS: int = 1

FrameArray = npt.NDArray[np.uint8]


class ReplayCaptureSource:
    """Synthetic fixture-backed capture source for orchestrator replay mode.

    Args:
        fixture_dir: Directory containing ``video.mkv``, ``audio.wav``, and
            ``stimulus_script.json``.
        realtime: When true, audio reads sleep until the requested bytes would
            be available on a real 16 kHz stream and video frames are selected
            from wall-clock elapsed time. Tests may pass false for fast
            deterministic stepping while keeping the same public surface.
        ffmpeg_bin: ffmpeg binary used for H.264/MKV decoding and replay audio
            resampling.
    """

    def __init__(
        self,
        fixture_dir: str | Path,
        *,
        realtime: bool = True,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.fixture_dir = Path(fixture_dir)
        self.video_path = self.fixture_dir / FIXTURE_VIDEO_NAME
        self.audio_path = self.fixture_dir / FIXTURE_AUDIO_NAME
        self.script_path = self.fixture_dir / FIXTURE_SCRIPT_NAME
        self.ffmpeg_bin = ffmpeg_bin
        self.realtime = realtime

        self._validate_fixture_files()
        self.script: dict[str, Any] = json.loads(self.script_path.read_text(encoding="utf-8"))
        self._validate_script()

        self.fps: int = int(self.script["fps"])
        self.width: int = int(self.script["video_width"])
        self.height: int = int(self.script["video_height"])
        self.duration_s: float = float(self.script["duration_s"])
        self.segment_duration_s: float = float(self.script["segment_duration_s"])
        self.stimuli: list[dict[str, Any]] = list(self.script["stimuli"])

        self._frames: FrameArray | None = None
        self._audio_16k: bytes | None = None
        self._audio_cursor: int = 0
        self._running: bool = False
        self._start_monotonic: float | None = None
        self._manual_elapsed_s: float | None = None
        self._sequential_frame_index: int = 0

    def _validate_fixture_files(self) -> None:
        """Raise a clear error if any required fixture artifact is missing."""
        missing = [
            path.name
            for path in (self.video_path, self.audio_path, self.script_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Replay fixture {self.fixture_dir} is missing required files: {', '.join(missing)}"
            )

    def _validate_script(self) -> None:
        """Validate the replay fixture metadata contract."""
        required_top_level = {
            "fps",
            "duration_s",
            "segment_duration_s",
            "video_width",
            "video_height",
            "audio_sample_rate_hz",
            "audio_sample_width_bytes",
            "audio_channels",
            "stimuli",
        }
        missing = sorted(required_top_level - set(self.script))
        if missing:
            raise ValueError(f"stimulus_script.json missing keys: {', '.join(missing)}")
        if int(self.script["fps"]) != REPLAY_FPS:
            raise ValueError(f"Replay video must be {REPLAY_FPS} fps")
        if int(self.script["audio_sample_rate_hz"]) != FIXTURE_AUDIO_SAMPLE_RATE_HZ:
            raise ValueError(f"Replay audio.wav must be {FIXTURE_AUDIO_SAMPLE_RATE_HZ} Hz")
        if int(self.script["audio_sample_width_bytes"]) != SAMPLE_WIDTH_BYTES:
            raise ValueError("Replay audio.wav must be signed 16-bit PCM")
        if int(self.script["audio_channels"]) != CHANNELS:
            raise ValueError("Replay audio.wav must be mono")
        if not isinstance(self.script["stimuli"], list) or not self.script["stimuli"]:
            raise ValueError("stimulus_script.json must contain at least one stimulus entry")

        required_stimulus_keys = {
            "segment_index",
            "stimulus_offset_s",
            "expected_arm_id",
            "expected_greeting_text",
            "expected_peak_au12",
            "expected_semantic_match",
        }
        for index, stimulus in enumerate(self.script["stimuli"]):
            missing_stimulus = sorted(required_stimulus_keys - set(stimulus))
            if missing_stimulus:
                raise ValueError(
                    "stimulus_script.json stimulus "
                    f"{index} missing keys: {', '.join(missing_stimulus)}"
                )

    @property
    def is_running(self) -> bool:
        """Whether the replay source has been started and not yet stopped."""
        return self._running

    @property
    def audio_duration_s(self) -> float:
        """Duration of the 16 kHz orchestrator audio stream."""
        if self._audio_16k is None:
            return self.duration_s
        return len(self._audio_16k) / (ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES)

    def start(self) -> None:
        """Load fixture media and start wall-clock replay cadence."""
        if self._running:
            return
        if self._frames is None:
            self._frames = self._load_video_frames()
        if self._audio_16k is None:
            self._audio_16k = self._load_audio_16k()
        self._audio_cursor = 0
        self._sequential_frame_index = 0
        self._manual_elapsed_s = None
        self._start_monotonic = time.monotonic()
        self._running = True
        logger.info("Replay capture started from fixture %s", self.fixture_dir)

    def stop(self) -> None:
        """Stop replay; cached media remains available for later restarts."""
        self._running = False
        self._start_monotonic = None
        self._manual_elapsed_s = None
        logger.info("Replay capture stopped")

    def read_chunk(self, num_bytes: int) -> bytes:
        """Read continuous 16 kHz mono s16le PCM bytes for the orchestrator.

        The source fixture is 48 kHz WAV.  This method mirrors the live
        ``AudioResampler.read_chunk`` output consumed by ``Orchestrator.run`` by
        first feeding fixture PCM through FFmpeg's 48 kHz -> 16 kHz resampler.
        """
        if num_bytes <= 0:
            return b""
        if not self._running:
            self.start()
        if self._audio_16k is None:
            return b""

        if self.realtime:
            self._sleep_until_audio_available(self._audio_cursor + num_bytes)

        if self._audio_cursor >= len(self._audio_16k):
            return b""
        end = min(self._audio_cursor + num_bytes, len(self._audio_16k))
        chunk = self._audio_16k[self._audio_cursor : end]
        self._audio_cursor = end
        return chunk

    def get_latest_frame(self) -> FrameArray | None:
        """Return the synthetic BGR frame corresponding to replay elapsed time."""
        if not self._running:
            return None
        if self._frames is None or len(self._frames) == 0:
            return None

        if self.realtime:
            start = self._start_monotonic if self._start_monotonic is not None else time.monotonic()
            elapsed_s = max(0.0, time.monotonic() - start)
            frame_index = int(elapsed_s * self.fps)
        elif self._manual_elapsed_s is not None:
            frame_index = int(max(0.0, self._manual_elapsed_s) * self.fps)
        else:
            frame_index = self._sequential_frame_index
            self._sequential_frame_index += 1

        frame_index = min(max(frame_index, 0), len(self._frames) - 1)
        return cast(FrameArray, self._frames[frame_index])

    def seek(self, elapsed_s: float, *, update_audio: bool = False) -> None:
        """Set non-realtime video elapsed time for deterministic tests.

        ``update_audio`` is false by default so tests can step video frames while
        still reading audio sequentially through ``read_chunk``.
        """
        clamped = min(max(elapsed_s, 0.0), self.duration_s)
        self._manual_elapsed_s = clamped
        self._sequential_frame_index = int(clamped * self.fps)
        if update_audio:
            byte_offset = int(clamped * ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES)
            if self._audio_16k is not None:
                byte_offset = min(byte_offset, len(self._audio_16k))
            self._audio_cursor = max(0, byte_offset)

    def elapsed_for_stimulus(self, stimulus: dict[str, Any]) -> float:
        """Return absolute fixture elapsed seconds for a stimulus metadata row."""
        return (int(stimulus["segment_index"]) * self.segment_duration_s) + float(
            stimulus["stimulus_offset_s"]
        )

    def _sleep_until_audio_available(self, target_cursor: int) -> None:
        """Block until ``target_cursor`` bytes should be available by wall clock."""
        if self._start_monotonic is None:
            return
        target_elapsed_s = target_cursor / (ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES)
        deadline = self._start_monotonic + min(target_elapsed_s, self.audio_duration_s)
        remaining_s = deadline - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)

    def _load_video_frames(self) -> FrameArray:
        """Decode H.264/MKV fixture video to BGR numpy frames via ffmpeg."""
        cmd = [
            self.ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, bufsize=10**8)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed to decode replay video: {stderr}")

        frame_bytes = self.width * self.height * 3
        if frame_bytes <= 0:
            raise ValueError("Replay fixture has invalid video dimensions")
        if len(result.stdout) % frame_bytes != 0:
            raise ValueError("Decoded replay video byte count is not divisible by frame dimensions")
        frame_count = len(result.stdout) // frame_bytes
        if frame_count == 0:
            raise ValueError("Replay fixture video decoded zero frames")
        expected_frames = int(round(self.duration_s * self.fps))
        if frame_count != expected_frames:
            logger.warning(
                "Replay fixture decoded %d frames, expected %d from metadata",
                frame_count,
                expected_frames,
            )
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
            (frame_count, self.height, self.width, 3)
        )

    def _read_fixture_pcm_48k(self) -> bytes:
        """Read and validate the fixture WAV as raw 48 kHz mono s16le PCM."""
        with wave.open(str(self.audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if channels != CHANNELS:
            raise ValueError("Replay audio.wav must be mono")
        if sample_width != SAMPLE_WIDTH_BYTES:
            raise ValueError("Replay audio.wav must be signed 16-bit PCM")
        if sample_rate != FIXTURE_AUDIO_SAMPLE_RATE_HZ:
            raise ValueError(f"Replay audio.wav must be {FIXTURE_AUDIO_SAMPLE_RATE_HZ} Hz")
        return raw

    def _load_audio_16k(self) -> bytes:
        """Load fixture WAV and resample through FFmpeg's live 48 kHz -> 16 kHz path."""
        raw_48k = self._read_fixture_pcm_48k()
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(FIXTURE_AUDIO_SAMPLE_RATE_HZ),
            "-ac",
            str(CHANNELS),
            "-i",
            "pipe:0",
            "-ar",
            str(ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ),
            "-f",
            "s16le",
            "-ac",
            str(CHANNELS),
            "pipe:1",
        ]
        result = subprocess.run(cmd, input=raw_48k, check=False, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed to resample replay audio: {stderr}")
        if len(result.stdout) % (SAMPLE_WIDTH_BYTES * CHANNELS) != 0:
            raise ValueError("FFmpeg replay resampler emitted a partial PCM sample")

        expected_bytes = int(round(self.duration_s * ORCHESTRATOR_AUDIO_SAMPLE_RATE_HZ))
        expected_bytes *= SAMPLE_WIDTH_BYTES * CHANNELS
        if abs(len(result.stdout) - expected_bytes) > SAMPLE_WIDTH_BYTES * CHANNELS:
            logger.warning(
                "Replay audio resampled to %d bytes, expected approximately %d bytes",
                len(result.stdout),
                expected_bytes,
            )
        return result.stdout
