#!/usr/bin/env python3
"""Generate deterministic synthetic capture fixtures for replay tests.

The CLI writes three files into a requested fixture directory:

* ``video.mkv``: 30 FPS H.264-in-MKV synthetic face animation.
* ``audio.wav``: 48 kHz mono PCM s16le WAV with deterministic lexical speech.
* ``stimulus_script.json``: ground-truth stimulus and expected AU12 metadata.

The media is synthetic-only and intentionally small enough for CI fixtures while
remaining byte-compatible with the worker's capture ingestion contracts.  Speech
is synthesized offline from the literal ``expected_greeting_text`` values; the
generator never calls network TTS and never falls back to tone/noise placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FPS: int = 30
AUDIO_SAMPLE_RATE_HZ: int = 48_000
OUTPUT_AUDIO_SAMPLE_RATE_HZ: int = 16_000
AUDIO_CHANNELS: int = 1
AUDIO_SAMPLE_WIDTH_BYTES: int = 2
NEUTRAL_MOUTH_RATIO: float = 0.55
PEAK_MOUTH_DEVIATION: float = 0.16
AU12_ALPHA: float = 6.0
SPEECH_PEAK_AMPLITUDE: float = 0.34
SPEECH_BACKEND_EMBEDDED: str = "embedded"
SPEECH_BACKEND_ESPEAK_NG: str = "espeak-ng"
SPEECH_BACKEND_CHOICES: tuple[str, str] = (
    SPEECH_BACKEND_EMBEDDED,
    SPEECH_BACKEND_ESPEAK_NG,
)
EMBEDDED_SPEECH_BACKEND_ID: str = "embedded-formant-phoneme-v1"
EMBEDDED_SPEECH_BACKEND_VERSION: str = "1"

Frame = npt.NDArray[np.uint8]

GREETINGS: tuple[tuple[str, str], ...] = (
    ("simple_hello", "Hello! Just joined, happy to be here!"),
    ("warm_welcome", "Hey! Thanks for streaming, you're awesome!"),
    ("direct_question", "Hi! What's the best advice you've gotten today?"),
    ("compliment_content", "Love the energy on this stream! How long have you been live?"),
)

FormantSpec = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]

# Compact English formant table used by the embedded deterministic synthesizer.
# This default backend keeps canonical fixture generation self-contained while
# still producing phoneme-level lexical speech rather than a non-linguistic tone
# or noise proxy.
_FORMANT_SPECS: dict[str, FormantSpec] = {
    "IY": ((280.0, 2250.0, 3000.0), (70.0, 150.0, 220.0), (1.0, 0.58, 0.25)),
    "IH": ((390.0, 1990.0, 2550.0), (80.0, 150.0, 210.0), (1.0, 0.52, 0.22)),
    "EH": ((530.0, 1840.0, 2480.0), (90.0, 160.0, 220.0), (1.0, 0.54, 0.24)),
    "AE": ((660.0, 1720.0, 2410.0), (95.0, 170.0, 230.0), (1.0, 0.50, 0.22)),
    "AH": ((640.0, 1190.0, 2390.0), (95.0, 150.0, 210.0), (1.0, 0.45, 0.22)),
    "AA": ((730.0, 1090.0, 2440.0), (100.0, 150.0, 220.0), (1.0, 0.42, 0.20)),
    "AO": ((570.0, 840.0, 2410.0), (90.0, 120.0, 220.0), (1.0, 0.38, 0.20)),
    "OW": ((500.0, 880.0, 2570.0), (85.0, 120.0, 220.0), (1.0, 0.40, 0.18)),
    "UH": ((440.0, 1020.0, 2240.0), (80.0, 130.0, 210.0), (1.0, 0.42, 0.18)),
    "UW": ((320.0, 920.0, 2200.0), (75.0, 120.0, 210.0), (1.0, 0.36, 0.18)),
    "ER": ((490.0, 1350.0, 1690.0), (85.0, 160.0, 180.0), (1.0, 0.44, 0.26)),
    "EY": ((430.0, 2050.0, 2850.0), (80.0, 150.0, 220.0), (1.0, 0.54, 0.22)),
    "AY": ((660.0, 1730.0, 2600.0), (95.0, 170.0, 230.0), (1.0, 0.50, 0.22)),
    "OY": ((570.0, 1200.0, 2600.0), (90.0, 150.0, 230.0), (1.0, 0.46, 0.22)),
    "M": ((280.0, 1050.0, 2200.0), (80.0, 180.0, 260.0), (0.75, 0.20, 0.08)),
    "N": ((300.0, 1450.0, 2550.0), (80.0, 180.0, 260.0), (0.72, 0.24, 0.10)),
    "NG": ((280.0, 1650.0, 2600.0), (80.0, 200.0, 280.0), (0.68, 0.20, 0.08)),
    "L": ((360.0, 1300.0, 2600.0), (80.0, 160.0, 240.0), (0.85, 0.35, 0.14)),
    "R": ((420.0, 1180.0, 1600.0), (85.0, 160.0, 190.0), (0.82, 0.36, 0.20)),
    "W": ((330.0, 900.0, 2200.0), (80.0, 130.0, 220.0), (0.76, 0.30, 0.12)),
    "Y": ((300.0, 2100.0, 2950.0), (70.0, 150.0, 220.0), (0.78, 0.48, 0.18)),
    "V": ((500.0, 1500.0, 2500.0), (120.0, 220.0, 320.0), (0.45, 0.22, 0.08)),
    "Z": ((500.0, 1700.0, 2900.0), (130.0, 240.0, 360.0), (0.42, 0.24, 0.10)),
    "JH": ((500.0, 1900.0, 2750.0), (120.0, 230.0, 330.0), (0.45, 0.30, 0.12)),
}

_PHONEME_DURATIONS_S: dict[str, float] = {
    "SIL_SHORT": 0.075,
    "SIL_MED": 0.140,
    "SIL_WORD": 0.030,
    "P": 0.055,
    "T": 0.050,
    "K": 0.055,
    "B": 0.055,
    "D": 0.050,
    "G": 0.055,
    "CH": 0.075,
    "JH": 0.075,
    "S": 0.065,
    "Z": 0.065,
    "SH": 0.075,
    "TH": 0.070,
    "F": 0.070,
    "V": 0.070,
    "HH": 0.055,
    "M": 0.070,
    "N": 0.065,
    "NG": 0.075,
    "L": 0.070,
    "R": 0.075,
    "W": 0.060,
    "Y": 0.055,
}

_WORD_PHONEMES: dict[str, tuple[str, ...]] = {
    "hello": ("HH", "EH", "L", "OW"),
    "just": ("JH", "AH", "S", "T"),
    "joined": ("JH", "OY", "N", "D"),
    "happy": ("HH", "AE", "P", "IY"),
    "to": ("T", "UW"),
    "be": ("B", "IY"),
    "here": ("HH", "IY", "R"),
    "hey": ("HH", "EY"),
    "thanks": ("TH", "AE", "NG", "K", "S"),
    "for": ("F", "AO", "R"),
    "streaming": ("S", "T", "R", "IY", "M", "IH", "NG"),
    "youre": ("Y", "UH", "R"),
    "awesome": ("AO", "S", "AH", "M"),
    "hi": ("HH", "AY"),
    "whats": ("W", "AH", "T", "S"),
    "the": ("TH", "AH"),
    "best": ("B", "EH", "S", "T"),
    "advice": ("AE", "D", "V", "AY", "S"),
    "youve": ("Y", "UW", "V"),
    "gotten": ("G", "AA", "T", "AH", "N"),
    "today": ("T", "AH", "D", "EY"),
    "love": ("L", "AH", "V"),
    "energy": ("EH", "N", "ER", "JH", "IY"),
    "on": ("AA", "N"),
    "this": ("TH", "IH", "S"),
    "stream": ("S", "T", "R", "IY", "M"),
    "how": ("HH", "AW"),
    "long": ("L", "AO", "NG"),
    "have": ("HH", "AE", "V"),
    "you": ("Y", "UW"),
    "been": ("B", "IH", "N"),
    "live": ("L", "AY", "V"),
}

_LETTER_PHONEMES: dict[str, tuple[str, ...]] = {
    "a": ("AH",),
    "b": ("B", "IY"),
    "c": ("K", "IY"),
    "d": ("D", "IY"),
    "e": ("IY",),
    "f": ("EH", "F"),
    "g": ("G", "IY"),
    "h": ("HH", "EY", "CH"),
    "i": ("AY",),
    "j": ("JH", "EY"),
    "k": ("K", "EY"),
    "l": ("EH", "L"),
    "m": ("EH", "M"),
    "n": ("EH", "N"),
    "o": ("OW",),
    "p": ("P", "IY"),
    "q": ("K", "Y", "UW"),
    "r": ("AA", "R"),
    "s": ("EH", "S"),
    "t": ("T", "IY"),
    "u": ("Y", "UW"),
    "v": ("V", "IY"),
    "w": ("D", "AH", "B", "L", "Y", "UW"),
    "x": ("EH", "K", "S"),
    "y": ("W", "AY"),
    "z": ("Z", "IY"),
}

_FRICATIVE_CENTER_HZ: dict[str, float] = {
    "S": 6200.0,
    "Z": 5600.0,
    "SH": 3400.0,
    "TH": 4200.0,
    "F": 5200.0,
    "V": 4600.0,
    "HH": 2500.0,
    "CH": 3600.0,
}

_PLOSIVES: set[str] = {"P", "T", "K", "B", "D", "G"}


def _smoothstep(value: float) -> float:
    """Return a clamped cubic smoothstep in [0, 1]."""
    x = min(max(value, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _draw_disk(
    frame: Frame, cx: float, cy: float, radius: float, color: tuple[int, int, int]
) -> None:
    """Draw a filled RGB disk into ``frame``."""
    height, width, _ = frame.shape
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(width - 1, int(math.ceil(cx + radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(height - 1, int(math.ceil(cy + radius)))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = ((xx - cx) ** 2) + ((yy - cy) ** 2) <= radius**2
    region = frame[y0 : y1 + 1, x0 : x1 + 1]
    region[mask] = color


def _fill_ellipse(
    frame: Frame,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled RGB ellipse into ``frame``."""
    height, width, _ = frame.shape
    x0 = max(0, int(math.floor(cx - rx)))
    x1 = min(width - 1, int(math.ceil(cx + rx)))
    y0 = max(0, int(math.floor(cy - ry)))
    y1 = min(height - 1, int(math.ceil(cy + ry)))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (((xx - cx) / max(rx, 1.0)) ** 2) + (((yy - cy) / max(ry, 1.0)) ** 2) <= 1.0
    region = frame[y0 : y1 + 1, x0 : x1 + 1]
    region[mask] = color


def _draw_line(
    frame: Frame,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a thick RGB line by stamping small disks along the segment."""
    steps = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))))
    for step in range(steps + 1):
        frac = step / steps
        _draw_disk(frame, x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, radius, color)


def _build_stimulus_script(
    *,
    segments: int,
    segment_duration_s: float,
    stimulus_offset_s: float,
    width: int,
    height: int,
    seed: int,
    speech_backend_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic fixture metadata and per-segment stimulus rows."""
    expected_peak = math.tanh(AU12_ALPHA * PEAK_MOUTH_DEVIATION)
    stimuli: list[dict[str, Any]] = []
    for segment_index in range(segments):
        arm_id, greeting = GREETINGS[(seed + segment_index) % len(GREETINGS)]
        stimuli.append(
            {
                "segment_index": segment_index,
                "stimulus_offset_s": round(stimulus_offset_s, 6),
                "expected_arm_id": arm_id,
                "expected_greeting_text": greeting,
                "expected_peak_au12": round(expected_peak, 6),
                "expected_semantic_match": True,
            }
        )

    return {
        "version": 1,
        "generator": "scripts/generate_capture_fixture.py",
        "seed": seed,
        "fps": FPS,
        "duration_s": round(segments * segment_duration_s, 6),
        "segment_duration_s": round(segment_duration_s, 6),
        "video_container": "mkv",
        "video_codec": "h264",
        "video_width": width,
        "video_height": height,
        "audio_sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
        "audio_sample_width_bytes": AUDIO_SAMPLE_WIDTH_BYTES,
        "audio_channels": AUDIO_CHANNELS,
        "audio_synthesis": (
            "deterministic_offline_lexical_speech:"
            f"{speech_backend_metadata['used']}"
        ),
        "speech_backend": dict(speech_backend_metadata),
        "orchestrator_audio_sample_rate_hz": OUTPUT_AUDIO_SAMPLE_RATE_HZ,
        "neutral_mouth_ratio": round(NEUTRAL_MOUTH_RATIO, 6),
        "peak_mouth_ratio": round(NEUTRAL_MOUTH_RATIO + PEAK_MOUTH_DEVIATION, 6),
        "au12_alpha": AU12_ALPHA,
        "stimuli": stimuli,
    }


def _segment_activation(elapsed_s: float, script: dict[str, Any]) -> float:
    """Return synthetic AU12 activation for a fixture timestamp."""
    segment_duration_s = float(script["segment_duration_s"])
    stimuli = script["stimuli"]
    segment_index = min(int(elapsed_s // segment_duration_s), len(stimuli) - 1)
    rel_s = elapsed_s - (segment_index * segment_duration_s)
    stimulus_offset_s = float(stimuli[segment_index]["stimulus_offset_s"])
    rise_s = min(0.45, max(0.12, segment_duration_s * 0.35))
    return _smoothstep((rel_s - stimulus_offset_s) / rise_s)


def _render_frame(width: int, height: int, frame_index: int, script: dict[str, Any]) -> Frame:
    """Render one synthetic RGB face frame for production FaceMesh detection."""
    elapsed_s = frame_index / FPS
    activation = _segment_activation(elapsed_s, script)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (24, 27, 42)
    scan_y = int((elapsed_s * 17.0) % max(height, 1))
    frame[max(0, scan_y - 1) : min(height, scan_y + 2), :, :] = (29, 33, 53)

    bob = math.sin(2.0 * math.pi * elapsed_s * 0.45) * height * 0.008
    cx = width * 0.5
    cy = height * 0.50 + bob
    face_rx = width * 0.255
    face_ry = height * 0.360

    # Neck, ears, hair, and shaded skin keep the procedural subject synthetic
    # while giving MediaPipe's detector a natural frontal-face silhouette.
    _fill_ellipse(frame, cx, cy + face_ry * 0.96, face_rx * 0.44, face_ry * 0.44, (183, 132, 98))
    _fill_ellipse(frame, cx - face_rx * 1.02, cy, face_rx * 0.16, face_ry * 0.24, (207, 154, 116))
    _fill_ellipse(frame, cx + face_rx * 1.02, cy, face_rx * 0.16, face_ry * 0.24, (207, 154, 116))
    _fill_ellipse(frame, cx, cy, face_rx, face_ry, (222, 176, 136))
    _fill_ellipse(frame, cx, cy + face_ry * 0.08, face_rx * 0.82, face_ry * 0.78, (233, 190, 150))
    _fill_ellipse(frame, cx, cy - face_ry * 0.86, face_rx * 0.92, face_ry * 0.28, (57, 37, 31))
    _fill_ellipse(
        frame,
        cx - face_rx * 0.22,
        cy - face_ry * 0.72,
        face_rx * 0.72,
        face_ry * 0.18,
        (67, 43, 34),
    )
    _fill_ellipse(
        frame,
        cx + face_rx * 0.40,
        cy - face_ry * 0.65,
        face_rx * 0.58,
        face_ry * 0.16,
        (63, 41, 34),
    )

    # Soft synthetic mesh/skin texture; deliberately low contrast so it does not
    # act as a side-channel landmark injection.
    mesh_color = (201, 151, 118)
    for row in range(-3, 4):
        row_y = cy + row * face_ry * 0.16
        row_rx = face_rx * math.sqrt(max(0.0, 1.0 - ((row_y - cy) / max(face_ry, 1.0)) ** 2))
        for col in range(-4, 5):
            dot_x = cx + (col / 4.8) * row_rx
            dot_y = row_y + math.sin(col + elapsed_s * 2.0) * 0.5
            _draw_disk(frame, dot_x, dot_y, max(0.9, width * 0.0035), mesh_color)

    iod_px = width * 0.34
    left_eye_x = cx - iod_px * 0.5
    right_eye_x = cx + iod_px * 0.5
    eye_y = cy - face_ry * 0.20
    eye_rx = max(4.0, width * 0.040)
    eye_ry = max(2.5, height * 0.020)

    for eye_x in (left_eye_x, right_eye_x):
        _fill_ellipse(frame, eye_x, eye_y, eye_rx, eye_ry, (245, 241, 230))
        _fill_ellipse(
            frame, eye_x + eye_rx * 0.08, eye_y, eye_ry * 0.78, eye_ry * 0.90, (66, 81, 97)
        )
        _fill_ellipse(
            frame, eye_x + eye_rx * 0.08, eye_y, eye_ry * 0.42, eye_ry * 0.52, (18, 24, 32)
        )
        _draw_line(
            frame,
            eye_x - eye_rx * 0.96,
            eye_y - eye_ry * 1.45,
            eye_x + eye_rx * 0.96,
            eye_y - eye_ry * 1.70,
            max(0.9, width * 0.0035),
            (62, 42, 35),
        )
        _draw_line(
            frame,
            eye_x - eye_rx * 0.95,
            eye_y + eye_ry * 0.08,
            eye_x + eye_rx * 0.95,
            eye_y + eye_ry * 0.05,
            max(0.8, width * 0.0027),
            (96, 58, 50),
        )

    nose_x = cx
    nose_top = cy - face_ry * 0.06
    nose_tip_y = cy + face_ry * 0.17
    _draw_line(
        frame,
        nose_x - width * 0.010,
        nose_top,
        nose_x - width * 0.022,
        nose_tip_y,
        1.1,
        (178, 120, 98),
    )
    _draw_line(
        frame,
        nose_x + width * 0.008,
        nose_top,
        nose_x + width * 0.026,
        nose_tip_y,
        1.0,
        (197, 142, 113),
    )
    _fill_ellipse(frame, nose_x, nose_tip_y, width * 0.030, height * 0.014, (203, 145, 113))
    _fill_ellipse(
        frame, nose_x - width * 0.018, nose_tip_y + height * 0.004, 1.6, 0.9, (91, 54, 52)
    )
    _fill_ellipse(
        frame, nose_x + width * 0.018, nose_tip_y + height * 0.004, 1.6, 0.9, (91, 54, 52)
    )

    mouth_ratio = NEUTRAL_MOUTH_RATIO + (PEAK_MOUTH_DEVIATION * activation)
    mouth_width_px = iod_px * mouth_ratio
    mouth_y = cy + face_ry * 0.33 - activation * height * 0.020
    left_mouth_x = cx - mouth_width_px * 0.5
    right_mouth_x = cx + mouth_width_px * 0.5
    smile_depth = height * (0.014 + 0.030 * activation)
    mid_y = mouth_y + smile_depth
    lip_radius = max(1.2, width * 0.006)

    # Draw lips as a real image feature that FaceMesh can follow. The geometric
    # contract remains the scripted lip-corner widening: neutral ratio 0.55 plus
    # a 0.16 deviation at peak, yielding tanh(alpha * deviation).
    _draw_line(frame, left_mouth_x, mouth_y, cx, mid_y, lip_radius, (126, 47, 58))
    _draw_line(frame, cx, mid_y, right_mouth_x, mouth_y, lip_radius, (126, 47, 58))
    _draw_line(
        frame,
        left_mouth_x + mouth_width_px * 0.08,
        mouth_y - height * 0.006,
        cx,
        mid_y - height * (0.012 + 0.010 * activation),
        max(0.9, lip_radius * 0.72),
        (164, 67, 76),
    )
    _draw_line(
        frame,
        cx,
        mid_y - height * (0.012 + 0.010 * activation),
        right_mouth_x - mouth_width_px * 0.08,
        mouth_y - height * 0.006,
        max(0.9, lip_radius * 0.72),
        (164, 67, 76),
    )
    if activation > 0.25:
        _fill_ellipse(
            frame,
            cx,
            mid_y - height * 0.004,
            mouth_width_px * 0.24,
            max(1.0, height * 0.016 * activation),
            (241, 235, 214),
        )
        _draw_line(
            frame,
            cx - mouth_width_px * 0.19,
            mid_y - height * 0.001,
            cx + mouth_width_px * 0.19,
            mid_y - height * 0.001,
            0.7,
            (197, 190, 176),
        )

    _fill_ellipse(
        frame,
        cx - face_rx * 0.44,
        cy + face_ry * 0.18,
        face_rx * 0.16,
        face_ry * 0.07,
        (231, 157, 138),
    )
    _fill_ellipse(
        frame,
        cx + face_rx * 0.44,
        cy + face_ry * 0.18,
        face_rx * 0.16,
        face_ry * 0.07,
        (231, 157, 138),
    )

    return frame


def _encode_video(video_path: Path, script: dict[str, Any], ffmpeg_bin: str) -> None:
    """Encode deterministic RGB frames as H.264 in an MKV container."""
    width = int(script["video_width"])
    height = int(script["video_height"])
    frame_count = int(round(float(script["duration_s"]) * FPS))
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+bitexact",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-g",
        str(FPS),
        "-keyint_min",
        str(FPS),
        "-x264-params",
        "scenecut=0:open-gop=0:threads=1",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-bitexact",
        str(video_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not created")
    try:
        for frame_index in range(frame_count):
            process.stdin.write(_render_frame(width, height, frame_index, script).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        process.wait()
        raise RuntimeError(f"ffmpeg terminated while encoding video: {stderr}") from exc

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed to encode video.mkv: {stderr}")


def _text_code(text: str, seed: int) -> int:
    """Return a stable integer code for deterministic speech variation."""
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) + seed * 104_729


def _normalize_word(word: str) -> str:
    """Normalize a phrase token for the exact fixture lexicon."""
    return "".join(char for char in word.lower() if char.isalnum())


def _phonemes_for_word(word: str) -> tuple[str, ...]:
    """Return phonemes for a known greeting word or a deterministic spelling fallback."""
    normalized = _normalize_word(word)
    if not normalized:
        return ()
    if normalized in _WORD_PHONEMES:
        return _WORD_PHONEMES[normalized]

    phonemes: list[str] = []
    for char in normalized:
        phonemes.extend(_LETTER_PHONEMES.get(char, ("AH",)))
    return tuple(phonemes)


def _phoneme_plan(text: str) -> list[str]:
    """Map literal greeting text to a deterministic phoneme and pause plan."""
    plan: list[str] = []
    current: list[str] = []

    def flush_word() -> None:
        if not current:
            return
        phonemes = _phonemes_for_word("".join(current))
        if phonemes:
            if plan and not plan[-1].startswith("SIL"):
                plan.append("SIL_WORD")
            plan.extend(phonemes)
        current.clear()

    for char in text:
        if char.isalnum() or char == "'":
            current.append(char)
            continue
        flush_word()
        if char in ",;:":
            plan.append("SIL_SHORT")
        elif char in ".!?":
            plan.append("SIL_MED")
        elif char.isspace() and plan and not plan[-1].startswith("SIL"):
            plan.append("SIL_WORD")
    flush_word()

    while plan and plan[0].startswith("SIL"):
        plan.pop(0)
    while plan and plan[-1].startswith("SIL"):
        plan.pop()
    return plan or ["AH"]


def _phoneme_duration_s(phoneme: str) -> float:
    """Return natural phoneme duration for embedded lexical synthesis."""
    if phoneme in _PHONEME_DURATIONS_S:
        return _PHONEME_DURATIONS_S[phoneme]
    if phoneme in _FORMANT_SPECS:
        return 0.095
    return 0.070


def _deterministic_noise(sample_count: int, seed: int) -> npt.NDArray[np.float64]:
    """Return deterministic pseudo-noise in [-1, 1] without RNG state."""
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float64)
    indices = np.arange(sample_count, dtype=np.uint64)
    seed64 = np.uint64(seed & 0xFFFFFFFF)
    values = (indices * np.uint64(1_664_525) + seed64 + np.uint64(1_013_904_223)) & np.uint64(
        0xFFFFFFFF
    )
    return values.astype(np.float64) / 2_147_483_647.5 - 1.0


def _apply_short_envelope(
    signal: npt.NDArray[np.float64], attack_s: float, release_s: float
) -> npt.NDArray[np.float64]:
    """Apply deterministic attack/release ramps to a phoneme waveform."""
    if signal.size == 0:
        return signal
    result = signal.copy()
    attack = min(result.size, max(1, int(round(attack_s * AUDIO_SAMPLE_RATE_HZ))))
    release = min(result.size, max(1, int(round(release_s * AUDIO_SAMPLE_RATE_HZ))))
    result[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float64)
    result[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float64)
    return result


def _render_voiced_phoneme(
    phoneme: str,
    duration_s: float,
    f0_hz: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Render one vowel-like or voiced consonant phoneme."""
    sample_count = int(round(duration_s * AUDIO_SAMPLE_RATE_HZ))
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float64)

    formants, bandwidths, amplitudes = _FORMANT_SPECS.get(phoneme, _FORMANT_SPECS["AH"])
    t = np.arange(sample_count, dtype=np.float64) / AUDIO_SAMPLE_RATE_HZ
    vibrato = 1.0 + 0.010 * np.sin(2.0 * math.pi * 4.2 * t + (seed % 31) * 0.1)
    phase = np.cumsum((2.0 * math.pi * f0_hz * vibrato) / AUDIO_SAMPLE_RATE_HZ)
    signal = np.zeros(sample_count, dtype=np.float64)
    max_harmonic = max(4, min(48, int(5_200.0 / max(f0_hz, 1.0))))
    for harmonic in range(1, max_harmonic + 1):
        harmonic_hz = f0_hz * harmonic
        weight = 0.0
        for formant_hz, bandwidth_hz, amplitude in zip(
            formants, bandwidths, amplitudes, strict=True
        ):
            weight += amplitude * math.exp(-0.5 * ((harmonic_hz - formant_hz) / bandwidth_hz) ** 2)
        signal += (weight / (harmonic**0.72)) * np.sin(harmonic * phase)

    breath = 0.012 * _deterministic_noise(sample_count, seed + 7)
    signal = signal + breath
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0.0:
        signal = signal / peak
    return _apply_short_envelope(signal, 0.010, 0.018)


def _render_fricative_phoneme(
    phoneme: str,
    duration_s: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Render deterministic fricative or aspirate speech noise."""
    sample_count = int(round(duration_s * AUDIO_SAMPLE_RATE_HZ))
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(sample_count, dtype=np.float64) / AUDIO_SAMPLE_RATE_HZ
    center_hz = _FRICATIVE_CENTER_HZ.get(phoneme, 4_200.0)
    noise = _deterministic_noise(sample_count, seed)
    high_pass = np.empty_like(noise)
    high_pass[0] = noise[0]
    high_pass[1:] = noise[1:] - 0.94 * noise[:-1]
    whistle = np.sin(2.0 * math.pi * center_hz * t + (seed % 19) * 0.13)
    hiss = high_pass * (0.72 + 0.28 * whistle)
    if phoneme in {"Z", "V"}:
        hiss += 0.24 * _render_voiced_phoneme(phoneme, duration_s, 125.0 + (seed % 25), seed + 11)
    peak = float(np.max(np.abs(hiss))) if hiss.size else 0.0
    if peak > 0.0:
        hiss = hiss / peak
    return _apply_short_envelope(hiss, 0.006, 0.016) * 0.58


def _render_plosive_phoneme(
    phoneme: str,
    duration_s: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Render deterministic stop consonant burst and release."""
    sample_count = int(round(duration_s * AUDIO_SAMPLE_RATE_HZ))
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float64)
    signal = np.zeros(sample_count, dtype=np.float64)
    burst_count = min(sample_count, max(1, int(round(0.018 * AUDIO_SAMPLE_RATE_HZ))))
    burst = _deterministic_noise(burst_count, seed + 23)
    burst_peak = float(np.max(np.abs(burst))) if burst.size else 0.0
    if burst_peak > 0.0:
        burst = burst / burst_peak
    signal[:burst_count] = burst * 0.82
    if phoneme in {"B", "D", "G"} and sample_count > burst_count:
        tail_count = sample_count - burst_count
        tail = _render_voiced_phoneme(
            "AH", tail_count / AUDIO_SAMPLE_RATE_HZ, 115.0 + (seed % 20), seed
        )
        signal[burst_count:] += tail * 0.22
    return _apply_short_envelope(signal, 0.002, 0.018) * 0.62


def _render_embedded_phoneme(
    phoneme: str,
    duration_s: float,
    f0_hz: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Render one phoneme for embedded deterministic speech synthesis."""
    if phoneme.startswith("SIL"):
        return np.zeros(int(round(duration_s * AUDIO_SAMPLE_RATE_HZ)), dtype=np.float64)
    if phoneme in _PLOSIVES:
        return _render_plosive_phoneme(phoneme, duration_s, seed)
    if phoneme in _FRICATIVE_CENTER_HZ or phoneme == "CH":
        return _render_fricative_phoneme(phoneme, duration_s, seed)
    return _render_voiced_phoneme(phoneme, duration_s, f0_hz, seed)


def _fit_waveform_duration(
    waveform: npt.NDArray[np.float64],
    duration_s: float,
) -> npt.NDArray[np.float64]:
    """Pad or resample a waveform to an exact deterministic duration."""
    target_count = int(round(duration_s * AUDIO_SAMPLE_RATE_HZ))
    if target_count <= 0:
        return np.zeros(0, dtype=np.float64)
    if waveform.size == target_count:
        fitted = waveform.astype(np.float64, copy=True)
    elif waveform.size <= 1:
        fitted = np.zeros(target_count, dtype=np.float64)
    else:
        source_positions = np.arange(waveform.size, dtype=np.float64)
        target_positions = np.linspace(
            0.0, float(waveform.size - 1), target_count, dtype=np.float64
        )
        fitted = np.interp(target_positions, source_positions, waveform).astype(np.float64)

    fade_len = max(1, min(target_count // 12, int(0.030 * AUDIO_SAMPLE_RATE_HZ)))
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    fitted[:fade_len] *= fade
    fitted[-fade_len:] *= fade[::-1]
    return fitted


def _trim_silence(
    waveform: npt.NDArray[np.float64], threshold: float = 0.006
) -> npt.NDArray[np.float64]:
    """Trim leading/trailing synthesizer silence while keeping small pads."""
    if waveform.size == 0:
        return waveform
    active = np.flatnonzero(np.abs(waveform) > threshold)
    if active.size == 0:
        return waveform
    pad = int(round(0.035 * AUDIO_SAMPLE_RATE_HZ))
    start = max(0, int(active[0]) - pad)
    stop = min(waveform.size, int(active[-1]) + pad + 1)
    return waveform[start:stop]


def _normalize_speech_waveform(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Center and peak-normalize a speech waveform to fixture amplitude."""
    if waveform.size == 0:
        return waveform
    centered = waveform.astype(np.float64, copy=True)
    centered -= float(np.mean(centered))
    peak = float(np.max(np.abs(centered)))
    if peak > 0.0:
        centered = centered / peak
    return centered * SPEECH_PEAK_AMPLITUDE


def _embedded_lexical_speech_waveform(
    text: str,
    duration_s: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Create deterministic phoneme-level speech for one literal greeting."""
    plan = _phoneme_plan(text)
    natural_duration_s = sum(_phoneme_duration_s(phoneme) for phoneme in plan)
    scale = duration_s / max(natural_duration_s, 1e-6)
    scale = min(max(scale, 0.58), 1.35)
    code = _text_code(text, seed)
    base_f0 = 138.0 + float(code % 34)
    pieces: list[npt.NDArray[np.float64]] = []
    voiced_index = 0
    for index, phoneme in enumerate(plan):
        phoneme_duration_s = _phoneme_duration_s(phoneme) * scale
        if not phoneme.startswith("SIL"):
            voiced_index += 1
        phrase_progress = index / max(len(plan) - 1, 1)
        question_lift = 18.0 * phrase_progress if text.strip().endswith("?") else 0.0
        f0 = base_f0 + question_lift + 9.0 * math.sin(2.0 * math.pi * phrase_progress)
        pieces.append(
            _render_embedded_phoneme(phoneme, phoneme_duration_s, f0, code + voiced_index * 97)
        )

    waveform = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    waveform = _fit_waveform_duration(waveform, duration_s)
    return _normalize_speech_waveform(waveform)


def _probe_espeak_ng_version(espeak_bin: str) -> str:
    """Return compact espeak-ng version metadata when it can be detected."""
    try:
        result = subprocess.run(
            [espeak_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return "unknown"
    return output.splitlines()[0].strip() or "unknown"


def _resolve_speech_backend(speech_backend: str, ffmpeg_bin: str) -> dict[str, Any]:
    """Resolve explicit speech backend selection into reproducibility metadata."""
    if speech_backend == SPEECH_BACKEND_EMBEDDED:
        return {
            "requested": SPEECH_BACKEND_EMBEDDED,
            "used": SPEECH_BACKEND_EMBEDDED,
            "identifier": EMBEDDED_SPEECH_BACKEND_ID,
            "version": EMBEDDED_SPEECH_BACKEND_VERSION,
            "deterministic": True,
        }

    if speech_backend == SPEECH_BACKEND_ESPEAK_NG:
        espeak_bin = shutil.which("espeak-ng")
        if espeak_bin is None:
            raise FileNotFoundError(
                "--speech-backend espeak-ng requires espeak-ng to be installed on PATH"
            )
        return {
            "requested": SPEECH_BACKEND_ESPEAK_NG,
            "used": SPEECH_BACKEND_ESPEAK_NG,
            "identifier": "espeak-ng",
            "version": _probe_espeak_ng_version(espeak_bin),
            "binary_path": espeak_bin,
            "voice": "en-us",
            "gap": "2",
            "ffmpeg_bin": ffmpeg_bin,
            "deterministic": False,
        }

    raise ValueError(f"Unsupported --speech-backend value: {speech_backend!r}")


def _espeak_ng_waveform(
    text: str,
    duration_s: float,
    ffmpeg_bin: str,
    espeak_bin: str,
) -> npt.NDArray[np.float64]:
    """Synthesize text with explicitly requested offline espeak-ng."""
    word_count = max(1, len([word for word in text.replace("'", "").split() if word.strip()]))
    speech_time_s = max(0.35, duration_s * 0.88)
    words_per_minute = int(min(390, max(145, round((word_count / speech_time_s) * 60.0))))
    espeak_cmd = [
        espeak_bin,
        "--stdout",
        "-v",
        "en-us",
        "-s",
        str(words_per_minute),
        "-g",
        "2",
        text,
    ]
    try:
        espeak_result = subprocess.run(espeak_cmd, check=False, capture_output=True)
    except OSError as exc:
        raise FileNotFoundError(
            "--speech-backend espeak-ng requires the resolved espeak-ng binary "
            f"to be executable: {espeak_bin}"
        ) from exc
    if espeak_result.returncode != 0 or not espeak_result.stdout:
        stderr = espeak_result.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr or "no audio data produced"
        raise RuntimeError(f"espeak-ng failed to synthesize fixture speech: {detail}")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ar",
        str(AUDIO_SAMPLE_RATE_HZ),
        "-ac",
        str(AUDIO_CHANNELS),
        "-f",
        "s16le",
        "pipe:1",
    ]
    ffmpeg_result = subprocess.run(
        ffmpeg_cmd,
        input=espeak_result.stdout,
        check=False,
        capture_output=True,
    )
    if ffmpeg_result.returncode != 0 or not ffmpeg_result.stdout:
        stderr = ffmpeg_result.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr or "no PCM data produced"
        raise RuntimeError(f"ffmpeg failed to convert espeak-ng speech: {detail}")

    samples = np.frombuffer(ffmpeg_result.stdout, dtype="<i2").astype(np.float64) / 32_768.0
    samples = _trim_silence(samples)
    samples = _fit_waveform_duration(samples, duration_s)
    return _normalize_speech_waveform(samples)


def _lexical_speech_waveform(
    text: str,
    duration_s: float,
    seed: int,
    ffmpeg_bin: str,
    speech_backend_metadata: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Create deterministic offline lexical speech for one greeting."""
    speech_backend = str(speech_backend_metadata["used"])
    if speech_backend == SPEECH_BACKEND_EMBEDDED:
        return _embedded_lexical_speech_waveform(text, duration_s, seed)
    if speech_backend == SPEECH_BACKEND_ESPEAK_NG:
        espeak_bin = str(speech_backend_metadata.get("binary_path") or "")
        if not espeak_bin:
            raise RuntimeError("espeak-ng backend metadata is missing binary_path")
        return _espeak_ng_waveform(text, duration_s, ffmpeg_bin, espeak_bin)
    raise ValueError(f"Unsupported speech backend in stimulus_script.json: {speech_backend!r}")


def _speech_duration_s(text: str, available_s: float) -> float:
    """Choose a deterministic spoken duration that fits the scripted segment."""
    words = [word for word in text.split() if word]
    natural_s = max(0.85, min(3.60, 0.34 * len(words) + 0.006 * len(text)))
    return min(max(0.24, available_s), natural_s)


def _write_audio(audio_path: Path, script: dict[str, Any], ffmpeg_bin: str) -> None:
    """Write deterministic 48 kHz mono PCM s16le WAV audio."""
    duration_s = float(script["duration_s"])
    sample_count = int(round(duration_s * AUDIO_SAMPLE_RATE_HZ))
    audio = np.zeros(sample_count, dtype=np.float64)
    segment_duration_s = float(script["segment_duration_s"])
    seed = int(script["seed"])

    for stimulus in script["stimuli"]:
        segment_index = int(stimulus["segment_index"])
        segment_start_s = segment_index * segment_duration_s
        stimulus_s = segment_start_s + float(stimulus["stimulus_offset_s"])
        available_s = max(0.05, segment_start_s + segment_duration_s - stimulus_s - 0.06)
        greeting_text = str(stimulus["expected_greeting_text"])
        greeting_duration_s = _speech_duration_s(greeting_text, available_s)
        waveform = _lexical_speech_waveform(
            greeting_text,
            greeting_duration_s,
            seed + segment_index,
            ffmpeg_bin,
            script["speech_backend"],
        )
        start = int(round(stimulus_s * AUDIO_SAMPLE_RATE_HZ))
        stop = min(sample_count, start + waveform.size)
        if stop > start:
            audio[start:stop] += waveform[: stop - start]

    pcm = np.clip(audio * 30_000.0, -32_768, 32_767).astype("<i2")
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(AUDIO_SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm.tobytes())


def _write_stimulus_script(script_path: Path, script: dict[str, Any]) -> None:
    """Write canonical JSON with stable ordering and trailing newline."""
    script_path.write_text(json.dumps(script, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic capture replay fixtures.",
    )
    parser.add_argument("fixture_dir", type=Path, help="Output fixture directory")
    parser.add_argument("--segments", type=int, default=3, help="Number of scripted segments")
    parser.add_argument(
        "--segment-duration-s",
        type=float,
        default=30.0,
        help="Duration of each scripted segment in seconds",
    )
    parser.add_argument(
        "--stimulus-offset-s",
        type=float,
        default=None,
        help="Stimulus offset within each segment; defaults to min(2s, 25%% of segment)",
    )
    parser.add_argument("--width", type=int, default=320, help="Synthetic video width in pixels")
    parser.add_argument("--height", type=int, default=240, help="Synthetic video height in pixels")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Deterministic greeting rotation seed"
    )
    parser.add_argument(
        "--speech-backend",
        choices=SPEECH_BACKEND_CHOICES,
        default=SPEECH_BACKEND_EMBEDDED,
        help=(
            "Speech synthesis backend for audio.wav; default is the embedded "
            "deterministic synthesizer"
        ),
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixture files")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> float:
    if args.segments < 1:
        raise ValueError("--segments must be >= 1")
    if args.segment_duration_s < 0.5:
        raise ValueError("--segment-duration-s must be >= 0.5")
    if args.width < 96 or args.height < 72:
        raise ValueError("--width/--height must be at least 96x72")
    stimulus_offset_s = (
        min(2.0, max(0.20, args.segment_duration_s * 0.25))
        if args.stimulus_offset_s is None
        else float(args.stimulus_offset_s)
    )
    if stimulus_offset_s <= 0.0 or stimulus_offset_s >= args.segment_duration_s:
        raise ValueError("stimulus offset must be within each segment")
    if stimulus_offset_s > args.segment_duration_s - 0.10:
        raise ValueError("stimulus offset leaves no room for post-stimulus response")
    return stimulus_offset_s


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    try:
        args = _parse_args(argv)
        stimulus_offset_s = _validate_args(args)
        speech_backend_metadata = _resolve_speech_backend(
            str(args.speech_backend),
            str(args.ffmpeg_bin),
        )
        fixture_dir: Path = args.fixture_dir
        fixture_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [
            fixture_dir / "video.mkv",
            fixture_dir / "audio.wav",
            fixture_dir / "stimulus_script.json",
        ]
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite:
            existing_names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing fixture files: {existing_names}")
        for path in existing:
            path.unlink()

        script = _build_stimulus_script(
            segments=args.segments,
            segment_duration_s=args.segment_duration_s,
            stimulus_offset_s=stimulus_offset_s,
            width=args.width,
            height=args.height,
            seed=args.seed,
            speech_backend_metadata=speech_backend_metadata,
        )
        _encode_video(fixture_dir / "video.mkv", script, str(args.ffmpeg_bin))
        _write_audio(fixture_dir / "audio.wav", script, str(args.ffmpeg_bin))
        _write_stimulus_script(fixture_dir / "stimulus_script.json", script)
    except Exception as exc:
        print(f"generate_capture_fixture: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
