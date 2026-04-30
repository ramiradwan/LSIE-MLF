"""
Video Capture — §4.D.2 / Gap G-03 Remediation

In-memory H.264 → numpy.ndarray bridge via PyAV for feeding
MediaPipe FaceMesh from the video IPC Pipe.

The Capture Container writes H.264 video in MKV streaming container
to /tmp/ipc/video_stream.mkv (an IPC Pipe). This module opens
that pipe via PyAV's libavformat bindings and decodes frames directly
into memory-view objects cast to numpy arrays — zero disk I/O.

§5.2 — Transient Data tier: raw video frames exist only in volatile
memory during active processing. No persistence.

Architecture notes (from Stage 2 Deployment Research):
  - PyAV bypasses subprocess overhead by binding directly to FFmpeg's
    C libraries (libavformat, libavcodec)
  - MKV is a streaming container (no seek required), safe for IPC Pipes
  - Decoded frames are BGR24 numpy arrays ready for cv2.cvtColor → RGB
    as required by MediaPipe FaceMesh (§4.D.2)
  - Frame jitter must be minimized to prevent IOD fluctuation in AU12
    normalization (§7A.3) and Thompson Sampling posterior corruption
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# §4.A.1 — Video IPC Pipe path (shared Docker volume)
VIDEO_PIPE_PATH: str = "/tmp/ipc/video_stream.mkv"

# Maximum buffered frames — discard oldest if processing falls behind
# §12 Queue overload: deque eviction prevents unbounded memory growth
MAX_FRAME_BUFFER: int = 5

# Retry delay when the video IPC Pipe is not yet available
VIDEO_PIPE_RETRY_DELAY: float = 2.0
VIDEO_PIPE_MAX_RETRIES: int = 30


class VideoCapture:
    """
    §4.D.2 / Gap G-03 — In-memory video frame capture via PyAV.

    Opens the video IPC Pipe (MKV container over mkfifo pipe) and
    decodes H.264 frames into numpy arrays for MediaPipe FaceMesh.

    Runs a background thread to continuously decode frames into a
    bounded deque. The orchestrator's segment assembly loop calls
    get_latest_frame() synchronously to retrieve the most recent
    decoded frame for AU12 processing.

    Thread safety: deque with maxlen handles producer/consumer without
    explicit locking (CPython GIL + atomic append/pop).
    """

    def __init__(self, pipe_path: str = VIDEO_PIPE_PATH) -> None:
        self._pipe_path = pipe_path
        self._frame_buffer: deque[npt.NDArray[np.uint8]] = deque(maxlen=MAX_FRAME_BUFFER)
        self._container: Any = None  # PyAV container (lazy-opened)
        self._thread: threading.Thread | None = None
        self._running: bool = False

    def start(self) -> None:
        """
        Start the background video decode thread.

        The thread opens the video IPC Pipe via PyAV and continuously
        decodes H.264 frames into the bounded frame buffer.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._decode_loop,
            name="video-capture",
            daemon=True,
        )
        self._thread.start()
        logger.info("Video capture thread started (pipe: %s)", self._pipe_path)

    def _open_container(self) -> Any:
        """
        Open the video IPC Pipe via PyAV.

        Retries if the pipe is not yet available (Capture Container
        may start after the Worker). §12 Hardware loss — poll with
        retry before giving up.

        Returns:
            PyAV container object, or None on failure.
        """
        import av

        for attempt in range(VIDEO_PIPE_MAX_RETRIES):
            try:
                # PyAV opens the IPC Pipe and reads MKV stream headers
                # format=None lets PyAV auto-detect MKV from the pipe
                container = av.open(  # type: ignore[attr-defined]
                    self._pipe_path,
                    format=None,
                    options={"fflags": "nobuffer", "flags": "low_delay"},
                )
                logger.info("Video IPC Pipe opened via PyAV (attempt %d)", attempt + 1)
                return container
            except Exception as exc:
                logger.warning(
                    "Video IPC Pipe not ready (attempt %d/%d): %s",
                    attempt + 1,
                    VIDEO_PIPE_MAX_RETRIES,
                    exc,
                )
                time.sleep(VIDEO_PIPE_RETRY_DELAY)

        logger.error("Failed to open video IPC Pipe after %d attempts", VIDEO_PIPE_MAX_RETRIES)
        return None

    def _decode_loop(self) -> None:
        """
        Background decode loop: read MKV from pipe, decode H.264,
        push BGR numpy frames into the bounded deque.

        §5.2 — Frames exist only in volatile memory (numpy arrays).
        §12 Worker crash — restart on decode errors.
        """
        while self._running:
            try:
                container = self._open_container()
                if container is None:
                    logger.error("Video capture: giving up on pipe connection")
                    self._running = False
                    return

                self._container = container
                video_stream = container.streams.video[0]
                # Low-latency decode: don't buffer ahead
                video_stream.thread_type = "AUTO"

                for frame in container.decode(video_stream):
                    if not self._running:
                        break

                    # §4.D.2 — Convert to BGR numpy array for MediaPipe
                    # PyAV frame.to_ndarray() with format="bgr24" produces
                    # (H, W, 3) uint8 array — exactly what cv2/MediaPipe expects
                    bgr_array: npt.NDArray[np.uint8] = frame.to_ndarray(format="bgr24")

                    # §12 Queue overload — deque maxlen auto-evicts oldest
                    self._frame_buffer.append(bgr_array)

            except Exception as exc:
                logger.error("Video decode error: %s", exc, exc_info=True)
                # §12 Worker crash — close and retry
                if self._container is not None:
                    with contextlib.suppress(Exception):
                        self._container.close()
                    self._container = None
                time.sleep(1.0)

        # Cleanup on exit
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()
            self._container = None

    def get_latest_frame(self) -> npt.NDArray[np.uint8] | None:
        """
        Retrieve the most recent decoded video frame.

        Called by the Orchestrator during segment assembly to attach
        frame data for AU12 processing in Module D.

        Returns:
            BGR numpy array (H, W, 3) or None if no frames available.
            Older buffered frames are discarded — only the latest matters
            for FaceMesh temporal consistency (§4.D.2).
        """
        if not self._frame_buffer:
            return None

        # Consume all but the latest frame (temporal consistency)
        latest: npt.NDArray[np.uint8] | None = None
        while self._frame_buffer:
            try:
                latest = self._frame_buffer.popleft()
            except IndexError:
                break
        return latest

    def stop(self) -> None:
        """Stop the video capture thread and release resources."""
        self._running = False
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()
            self._container = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._frame_buffer.clear()
        logger.info("Video capture stopped")

    @property
    def is_running(self) -> bool:
        """Check if the video capture thread is active."""
        return self._running and self._thread is not None and self._thread.is_alive()
