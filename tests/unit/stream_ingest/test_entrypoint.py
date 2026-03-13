"""
Tests for services/stream_ingest/entrypoint.sh — Phase 5.1 validation.

Validates entrypoint script structure against §4.A.1 IPC pipe lifecycle
and §12 error handling. Uses static analysis since the script requires
a real Android USB device to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ENTRYPOINT_PATH = Path("services/stream_ingest/entrypoint.sh")
DOCKERFILE_PATH = Path("services/stream_ingest/Dockerfile")


@pytest.fixture()
def entrypoint_content() -> str:
    """Read entrypoint.sh content."""
    return ENTRYPOINT_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def dockerfile_content() -> str:
    """Read Dockerfile content."""
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


class TestEntrypointStructure:
    """§4.A.1 — IPC pipe lifecycle validation."""

    def test_shebang(self, entrypoint_content: str) -> None:
        """Script has proper bash shebang."""
        assert entrypoint_content.startswith("#!/usr/bin/env bash")

    def test_set_euo_pipefail(self, entrypoint_content: str) -> None:
        """Script uses strict error handling."""
        assert "set -euo pipefail" in entrypoint_content

    def test_ipc_pipe_path(self, entrypoint_content: str) -> None:
        """§4.A.1 — IPC Pipe at /tmp/ipc/audio_stream.raw."""
        assert 'IPC_PIPE="/tmp/ipc/audio_stream.raw"' in entrypoint_content

    def test_mkfifo_creates_pipe(self, entrypoint_content: str) -> None:
        """§4.A.1 step 1 — Creates named pipe with mkfifo."""
        assert "mkfifo" in entrypoint_content

    def test_fd3_open(self, entrypoint_content: str) -> None:
        """§4.A.1 step 2 — Non-blocking open with exec 3<>."""
        assert 'exec 3<>"$IPC_PIPE"' in entrypoint_content

    def test_scrcpy_audio_codec_raw(self, entrypoint_content: str) -> None:
        """§4.A.1 — scrcpy uses --audio-codec=raw for PCM s16le 48kHz."""
        assert "--audio-codec=raw" in entrypoint_content

    def test_scrcpy_audio_buffer(self, entrypoint_content: str) -> None:
        """§4.A.1 — scrcpy uses --audio-buffer=30 (30ms)."""
        assert "--audio-buffer=" in entrypoint_content

    def test_scrcpy_audio_dup(self, entrypoint_content: str) -> None:
        """§4.A.1 — scrcpy uses --audio-dup to mirror audio."""
        assert "--audio-dup" in entrypoint_content

    def test_scrcpy_no_audio_playback(self, entrypoint_content: str) -> None:
        """§4.A.1 — scrcpy uses --no-audio-playback."""
        assert "--no-audio-playback" in entrypoint_content

    def test_dd_pipes_to_fd3(self, entrypoint_content: str) -> None:
        """§4.A.1 steps 3–4 — dd pipes scrcpy stdout to fd 3."""
        assert "dd" in entrypoint_content
        assert "fd/3" in entrypoint_content

    def test_cleanup_closes_fd3(self, entrypoint_content: str) -> None:
        """§4.A.1 step 4 — Cleanup closes fd 3."""
        assert "exec 3>&-" in entrypoint_content

    def test_cleanup_removes_pipe(self, entrypoint_content: str) -> None:
        """§4.A.1 step 4 — Cleanup removes pipe file."""
        assert 'rm -f "$IPC_PIPE"' in entrypoint_content

    def test_trap_sigterm_sigint(self, entrypoint_content: str) -> None:
        """Graceful shutdown on SIGTERM and SIGINT."""
        assert "trap cleanup SIGTERM SIGINT" in entrypoint_content


class TestEntrypointErrorHandling:
    """§12 — Error handling for Module A."""

    def test_usb_poll_interval(self, entrypoint_content: str) -> None:
        """§12 Hardware loss A — poll every 2 seconds."""
        assert "USB_RETRY_INTERVAL=2" in entrypoint_content

    def test_usb_poll_max(self, entrypoint_content: str) -> None:
        """§12 Hardware loss A — poll for 60 seconds max."""
        assert "USB_RETRY_MAX=60" in entrypoint_content

    def test_reconnect_loop(self, entrypoint_content: str) -> None:
        """§12 Hardware loss A — Restart capture after USB reconnection."""
        assert "while true" in entrypoint_content
        assert "wait_for_device" in entrypoint_content

    def test_scrcpy_pid_tracked(self, entrypoint_content: str) -> None:
        """Scrcpy PID tracked for cleanup."""
        assert "SCRCPY_PID" in entrypoint_content


class TestDockerfile:
    """§9.1 — Capture Container image validation."""

    def test_base_image(self, dockerfile_content: str) -> None:
        """§9.1 — Ubuntu 22.04 base image."""
        assert "FROM ubuntu:22.04" in dockerfile_content

    def test_adb_installed(self, dockerfile_content: str) -> None:
        """adb required for USB device communication."""
        assert "adb" in dockerfile_content

    def test_scrcpy_installed(self, dockerfile_content: str) -> None:
        """§9.1 — scrcpy v2.x installed from official release."""
        assert "scrcpy" in dockerfile_content
        assert "Genymobile/scrcpy" in dockerfile_content

    def test_ipc_directory(self, dockerfile_content: str) -> None:
        """§4.A.1 — /tmp/ipc directory created."""
        assert "mkdir -p /tmp/ipc" in dockerfile_content

    def test_entrypoint_set(self, dockerfile_content: str) -> None:
        """Entrypoint is the shell script."""
        assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile_content

    def test_entrypoint_copied(self, dockerfile_content: str) -> None:
        """§3.2 — Build context is monorepo root."""
        assert "COPY services/stream_ingest/entrypoint.sh" in dockerfile_content

    def test_entrypoint_executable(self, dockerfile_content: str) -> None:
        """Entrypoint has execute permission."""
        assert "chmod +x /entrypoint.sh" in dockerfile_content
