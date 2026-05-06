"""Retained server/cloud video IPC decoder.

In-memory H.264 → numpy.ndarray bridge via PyAV for feeding MediaPipe FaceMesh
from the retained video IPC Pipe. Raw video frames exist only in volatile memory
during active processing and are not persisted.
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

VIDEO_PIPE_PATH: str = "/tmp/ipc/video_stream.mkv"
MAX_FRAME_BUFFER: int = 5
VIDEO_PIPE_RETRY_DELAY: float = 2.0
VIDEO_PIPE_MAX_RETRIES: int = 30


class VideoCapture:
    """In-memory retained video frame capture via PyAV."""

    def __init__(self, pipe_path: str = VIDEO_PIPE_PATH) -> None:
        self._pipe_path = pipe_path
        self._frame_buffer: deque[npt.NDArray[np.uint8]] = deque(maxlen=MAX_FRAME_BUFFER)
        self._container: Any = None  # PyAV container (lazy-opened)
        self._thread: threading.Thread | None = None
        self._running: bool = False

    def start(self) -> None:
        """Start the background retained video decode thread."""
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
        """Open the retained video stream via PyAV."""
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
