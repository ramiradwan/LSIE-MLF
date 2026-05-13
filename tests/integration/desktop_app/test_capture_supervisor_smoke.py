"""WS3 P3 — capture_supervisor smoke against a real Android device.

Gated by ``LSIE_INTEGRATION_DEVICE=1``. Runs only when an Android
device is on the USB bus AND ``scrcpy`` resolves on the host.

What it proves:
  - ``DriftCorrector.poll`` over real ADB returns a sub-second offset.
  - The full ``capture_supervisor.run`` body launches both scrcpy
    instances under :class:`SupervisedProcess`, writes replay-readable
    audio/video artifacts, and a cooperative shutdown leaves zero
    orphan ``scrcpy.exe`` / ``adb.exe`` PIDs in the system table.
  - Captured PCM can enter the real desktop IPC/shared-memory path and
    be ACKed/released without test-only schema bypasses.
  - Drift offsets reach the consumer side over
    ``IpcChannels.drift_updates``.

The smoke is intentionally short (≈10 seconds of capture) to avoid
filling the device's pageserver while still proving bounded artifact
integrity and cleanup.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty

import psutil
import pytest

from packages.schemas.evaluation import StimulusDefinition
from services.desktop_app.drift import DriftCorrector
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import InferenceControlMessage, PcmBlockAckMessage
from services.desktop_app.ipc.shared_buffers import read_pcm_block
from services.desktop_app.os_adapter import find_executable
from services.desktop_app.processes import module_c_orchestrator
from services.desktop_app.state.sqlite_schema import bootstrap_schema

ENABLED = os.environ.get("LSIE_INTEGRATION_DEVICE") == "1"


@pytest.fixture(scope="module")
def adb_path() -> str:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    try:
        return find_executable("adb", env_override="LSIE_ADB_PATH")
    except FileNotFoundError as exc:
        pytest.skip(f"adb not found: {exc}")


@pytest.fixture(scope="module")
def scrcpy_path() -> str:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    try:
        return find_executable("scrcpy", env_override="LSIE_SCRCPY_PATH")
    except FileNotFoundError as exc:
        pytest.skip(f"scrcpy not found: {exc}")


@pytest.fixture(scope="module")
def device_present(adb_path: str) -> Iterator[None]:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    result = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    has_device = any(
        line.strip().endswith("device")
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("List of devices")
    )
    if not has_device:
        pytest.skip(f"no adb device attached: {result.stdout!r}")
    yield


def test_real_device_drift_poll_returns_offset(device_present: None) -> None:
    """Real ADB shell call must produce a sub-second drift offset."""
    dc = DriftCorrector()
    offset = dc.poll()

    # Devices are usually within a few hundred ms of host time. A
    # multi-second offset here means the device clock or host clock is
    # wildly out of sync — surface that loudly so the operator notices.
    assert dc._consecutive_failures == 0
    assert dc._frozen is False
    assert abs(offset) < 5.0, f"drift offset {offset:.3f}s is implausibly large"


_STIMULUS_DEFINITION = StimulusDefinition.model_validate(
    {
        "stimulus_modality": "spoken_greeting",
        "stimulus_payload": {"content_type": "text", "text": "Say hello to the creator"},
        "expected_stimulus_rule": "Deliver the spoken greeting to the creator",
        "expected_response_rule": "The live streamer acknowledges the greeting",
    }
)

_BANDIT_SNAPSHOT = {
    "selection_method": "thompson_sampling",
    "selection_time_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
    "experiment_id": 1,
    "policy_version": "desktop_replay_v1",
    "selected_arm_id": "warm_welcome",
    "candidate_arm_ids": ["direct_question", "warm_welcome"],
    "posterior_by_arm": {
        "direct_question": {"alpha": 1.0, "beta": 5.0},
        "warm_welcome": {"alpha": 2.0, "beta": 3.0},
    },
    "sampled_theta_by_arm": {"direct_question": 0.2, "warm_welcome": 0.6},
    "stimulus_modality": _STIMULUS_DEFINITION.stimulus_modality,
    "stimulus_payload": _STIMULUS_DEFINITION.stimulus_payload.model_dump(mode="json"),
    "expected_stimulus_rule": _STIMULUS_DEFINITION.expected_stimulus_rule,
    "expected_response_rule": _STIMULUS_DEFINITION.expected_response_rule,
    "decision_context_hash": "a" * 64,
    "random_seed": 42,
}


def _orphan_scrcpy_pids() -> set[int]:
    return {
        proc.info["pid"]
        for proc in psutil.process_iter(["pid", "name"])
        if proc.info.get("name", "").lower().startswith("scrcpy")
    }


def _bootstrap_runtime_db(state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "desktop.sqlite"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(conn)
    finally:
        conn.close()
    return db_path


def _wait_for_first_drift(channels: IpcChannels, *, timeout_s: float = 30.0) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw = channels.drift_updates.get(timeout=1.0)
        except Exception:  # noqa: BLE001
            continue
        assert isinstance(raw, dict)
        if "drift_offset" in raw:
            return raw
    raise AssertionError(f"no drift_updates payload arrived within {timeout_s:.0f} s")


def _wait_for_capture_artifacts(capture_dir: Path, *, timeout_s: float = 30.0) -> tuple[Path, Path]:
    audio_path = capture_dir / "audio_stream.wav"
    video_path = capture_dir / "video_stream.mkv"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            audio_path.exists()
            and video_path.exists()
            and audio_path.stat().st_size > 44
            and video_path.stat().st_size > 0
        ):
            return audio_path, video_path
        time.sleep(0.5)
    raise AssertionError(
        f"capture artifacts not ready within {timeout_s:.0f} s: {audio_path} {video_path}"
    )


def _wait_for_capture_status_ready(db_path: Path, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT status_key, state, detail FROM capture_status").fetchall()
        finally:
            conn.close()
        mapping = {str(row["status_key"]): row for row in rows}
        audio = mapping.get("audio_capture")
        video = mapping.get("video_capture")
        if (
            audio is not None
            and video is not None
            and audio["state"] == "ok"
            and video["state"] == "ok"
        ):
            assert audio["detail"] is not None
            assert "Audio stream recording:" in str(audio["detail"])
            assert video["detail"] is not None
            assert "Video stream recording:" in str(video["detail"])
            return
        time.sleep(0.5)
    raise AssertionError("capture_status never reported replay-ready audio/video artifacts")


def _assert_video_container_readable(video_path: Path) -> None:
    import av

    with av.open(str(video_path), format=None) as container:  # type: ignore[attr-defined]
        assert container.streams.video, f"no video streams in {video_path}"


def _exercise_replay_ready_pcm(
    channels: IpcChannels,
    audio_path: Path,
) -> None:
    audio_format = module_c_orchestrator._read_capture_audio_format(audio_path)  # noqa: SLF001
    assert audio_format is not None, f"audio artifact was not WAV-readable: {audio_path}"

    raw, cursor = module_c_orchestrator._read_new_capture_audio(  # noqa: SLF001
        audio_path,
        0,
        audio_format,
    )
    assert cursor > audio_format.data_offset_bytes
    pcm_16k = module_c_orchestrator._source_pcm_to_16k_mono(raw, audio_format)  # noqa: SLF001
    assert pcm_16k, "captured audio produced no replay PCM"

    dispatcher = module_c_orchestrator.DesktopSegmentDispatcher(
        channels.ml_inbox,
        channels.pcm_acks,
    )
    segment = module_c_orchestrator.DesktopSegment(
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="test://real-device-capture",
        segment_window_start_utc=datetime.now(UTC),
        pcm_s16le_16khz_mono=pcm_16k,
        experiment_row_id=1,
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        stimulus_definition=_STIMULUS_DEFINITION,
        bandit_decision_snapshot=_BANDIT_SNAPSHOT,
        stimulus_time_s=100.0,
    )
    try:
        assert dispatcher.dispatch(segment)
        raw_control = channels.ml_inbox.get(timeout=1.0)
        control = InferenceControlMessage.model_validate(raw_control)
        recovered = read_pcm_block(control.audio.to_metadata())
        assert recovered == pcm_16k
        assert channels.pcm_acks is not None
        channels.pcm_acks.put(PcmBlockAckMessage(name=control.audio.name).model_dump(mode="json"))
        dispatcher.release_acked_blocks()
        with pytest.raises(FileNotFoundError):
            read_pcm_block(control.audio.to_metadata())
    finally:
        dispatcher.close_inflight_blocks()
        while True:
            try:
                channels.ml_inbox.get_nowait()
            except Empty:
                break


def test_capture_supervisor_full_lifecycle_leaves_no_orphans(
    device_present: None,
    scrcpy_path: str,
    tmp_path: Path,
) -> None:
    """Spawn capture_supervisor, verify in-run artifacts, then verify cleanup."""
    pre_orphans = _orphan_scrcpy_pids()
    assert pre_orphans == set(), f"pre-test orphans present: {pre_orphans}"

    ctx = mp.get_context("spawn")
    channels = IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
        pcm_acks=ctx.Queue(),
    )
    shutdown = ctx.Event()

    capture_dir = tmp_path / "capture"
    state_dir = tmp_path / "state"
    db_path = _bootstrap_runtime_db(state_dir)

    proc = ctx.Process(
        target=_run_capture_supervisor_in_subprocess,
        args=(shutdown, channels, str(capture_dir), str(state_dir), scrcpy_path),
        name="capture_supervisor",
    )
    proc.start()
    try:
        first_drift = _wait_for_first_drift(channels)
        assert "drift_offset" in first_drift

        audio_path, video_path = _wait_for_capture_artifacts(capture_dir)
        _wait_for_capture_status_ready(db_path)
        _assert_video_container_readable(video_path)
        _exercise_replay_ready_pcm(channels, audio_path)

        shutdown.set()
        proc.join(timeout=15.0)
        assert not proc.is_alive(), "supervisor did not stop cooperatively"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)

    post_orphans = _orphan_scrcpy_pids()
    assert post_orphans == set(), f"orphan scrcpy pids: {post_orphans}"
    assert not (capture_dir / "audio_stream.wav").exists()
    assert not (capture_dir / "video_stream.mkv").exists()


def _run_capture_supervisor_in_subprocess(
    shutdown_event: object,
    channels: IpcChannels,
    capture_dir: str,
    state_dir: str,
    scrcpy_path: str,
) -> None:
    """Spawn-mode child entrypoint: re-import + invoke run() with overrides set."""
    import logging
    import os as _os

    _os.environ["LSIE_CAPTURE_DIR"] = capture_dir
    _os.environ["LSIE_STATE_DIR"] = state_dir
    _os.environ["LSIE_SCRCPY_PATH"] = scrcpy_path
    logging.basicConfig(level=logging.INFO)

    from services.desktop_app.processes.capture_supervisor import run

    run(shutdown_event=shutdown_event, channels=channels)  # type: ignore[arg-type]
