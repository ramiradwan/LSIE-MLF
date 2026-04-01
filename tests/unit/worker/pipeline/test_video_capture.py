"""
Tests for services/worker/pipeline/video_capture.py — Phase 3.3 gap-fix coverage.

Verifies the VideoCapture in-memory H.264 → numpy bridge:
  §4.D.2 — Face Mesh video frame input
  §5.2 — Transient Data: frames in volatile memory only
  §12 — Queue overload: deque eviction, crash recovery
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from services.worker.pipeline.video_capture import MAX_FRAME_BUFFER, VIDEO_PIPE_PATH, VideoCapture


def _make_frame(value: int = 0) -> npt.NDArray[np.uint8]:
    """Create a small BGR test frame (10x10 pixels)."""
    return np.full((10, 10, 3), value, dtype=np.uint8)


class TestVideoInit:
    """VideoCapture.__init__: default configuration."""

    def test_default_pipe_path(self) -> None:
        """Default pipe path matches §4.A.1 video IPC Pipe."""
        vc = VideoCapture()
        assert vc._pipe_path == VIDEO_PIPE_PATH

    def test_custom_pipe_path(self) -> None:
        """Custom pipe path is accepted."""
        vc = VideoCapture(pipe_path="/tmp/custom.mkv")
        assert vc._pipe_path == "/tmp/custom.mkv"

    def test_buffer_maxlen(self) -> None:
        """Frame buffer is bounded by MAX_FRAME_BUFFER (§12 Queue overload)."""
        vc = VideoCapture()
        assert isinstance(vc._frame_buffer, deque)
        assert vc._frame_buffer.maxlen == MAX_FRAME_BUFFER

    def test_initial_state(self) -> None:
        """Not running before start() is called."""
        vc = VideoCapture()
        assert vc._running is False
        assert vc._thread is None
        assert vc._container is None


class TestGetLatestFrame:
    """VideoCapture.get_latest_frame: retrieve most recent decoded frame."""

    def test_empty_buffer_returns_none(self) -> None:
        """Returns None when no frames are available."""
        vc = VideoCapture()
        assert vc.get_latest_frame() is None

    def test_returns_latest_frame(self) -> None:
        """Returns the most recently buffered frame."""
        vc = VideoCapture()
        vc._frame_buffer.append(_make_frame(1))
        vc._frame_buffer.append(_make_frame(2))
        vc._frame_buffer.append(_make_frame(3))

        latest = vc.get_latest_frame()
        assert latest is not None
        # The last frame appended (value 3) should be the latest returned
        assert int(latest[0, 0, 0]) == 3

    def test_drains_buffer(self) -> None:
        """Buffer is empty after get_latest_frame consumes all frames."""
        vc = VideoCapture()
        vc._frame_buffer.append(_make_frame(1))
        vc._frame_buffer.append(_make_frame(2))

        vc.get_latest_frame()
        assert len(vc._frame_buffer) == 0

    def test_single_frame(self) -> None:
        """Works correctly with exactly one frame in buffer."""
        vc = VideoCapture()
        frame = _make_frame(42)
        vc._frame_buffer.append(frame)

        result = vc.get_latest_frame()
        assert result is not None
        assert int(result[0, 0, 0]) == 42
        assert len(vc._frame_buffer) == 0


class TestIsRunning:
    """VideoCapture.is_running property."""

    def test_false_before_start(self) -> None:
        """is_running is False before start()."""
        vc = VideoCapture()
        assert vc.is_running is False

    def test_false_when_running_but_no_thread(self) -> None:
        """is_running is False if _running=True but thread is None."""
        vc = VideoCapture()
        vc._running = True
        assert vc.is_running is False

    def test_false_after_stop(self) -> None:
        """is_running is False after stop()."""
        vc = VideoCapture()
        vc._running = True
        vc.stop()
        assert vc.is_running is False


class TestStop:
    """VideoCapture.stop: resource cleanup."""

    def test_sets_running_false(self) -> None:
        """stop() sets _running to False."""
        vc = VideoCapture()
        vc._running = True
        vc.stop()
        assert vc._running is False

    def test_clears_buffer(self) -> None:
        """stop() empties the frame buffer."""
        vc = VideoCapture()
        vc._frame_buffer.append(_make_frame(1))
        vc._frame_buffer.append(_make_frame(2))
        vc.stop()
        assert len(vc._frame_buffer) == 0

    def test_closes_container(self) -> None:
        """stop() closes the PyAV container if open."""
        from unittest.mock import MagicMock

        vc = VideoCapture()
        mock_container = MagicMock()
        vc._container = mock_container
        vc.stop()
        mock_container.close.assert_called_once()
        assert vc._container is None

    def test_nulls_thread(self) -> None:
        """stop() joins and nulls the thread."""
        from unittest.mock import MagicMock

        vc = VideoCapture()
        mock_thread = MagicMock()
        vc._thread = mock_thread
        vc.stop()
        mock_thread.join.assert_called_once_with(timeout=5)
        assert vc._thread is None

    def test_safe_when_already_stopped(self) -> None:
        """stop() is safe to call multiple times."""
        vc = VideoCapture()
        vc.stop()
        vc.stop()  # Should not raise


class TestDequeEviction:
    """§12 Queue overload — deque maxlen evicts oldest frames."""

    def test_buffer_evicts_oldest(self) -> None:
        """Oldest frames are evicted when buffer is full."""
        vc = VideoCapture()
        for i in range(MAX_FRAME_BUFFER + 3):
            vc._frame_buffer.append(_make_frame(i))

        assert len(vc._frame_buffer) == MAX_FRAME_BUFFER
        # The oldest frames (0, 1, 2) should have been evicted
        oldest_remaining = int(vc._frame_buffer[0][0, 0, 0])
        assert oldest_remaining == 3
