"""WS3 P2 — IPC replay parity against the Gate 0 corpus.

Drives the v4.0 IPC dispatch end-to-end on a single host (no
multiprocessing spawn) and proves byte-identical reconstruction of the
``InferenceHandoffPayload`` body that crosses the orchestrator →
gpu_ml_worker boundary. The producer side wraps
``services.worker.pipeline.orchestrator.Orchestrator._dispatch_payload``;
the consumer side mirrors what
``services.desktop_app.processes.gpu_ml_worker.run`` does after a
queue ``get`` — validate the control message, attach to the
SharedMemory block, copy audio, verify SHA-256.

The Gate 0 corpus lives at ``tests/fixtures/v4_gate0/segment_*.json``.
Each fixture's ``_au12_series`` and other fields drive a synthetic
``assemble_segment`` call; the test then asserts that the reconstructed
handoff body matches the fixture (modulo schema-prunable optionals)
and that the audio bytes survive the SharedMemory round trip with
deterministic SHA-256.
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from typing import Any

import pytest

from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.ipc.control_messages import InferenceControlMessage
from services.desktop_app.ipc.shared_buffers import (
    PcmBlockMetadata,
    read_pcm_block,
)
from services.worker.pipeline.orchestrator import Orchestrator

GATE0_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "v4_gate0"


def _fixtures() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(GATE0_FIXTURE_DIR.glob("segment_*.json"))]


@pytest.mark.parametrize(
    "fixture",
    _fixtures(),
    ids=[p.stem for p in sorted(GATE0_FIXTURE_DIR.glob("segment_*.json"))],
)
def test_ipc_dispatch_round_trips_fixture(fixture: dict[str, Any]) -> None:
    """Synthesize a payload from each Gate 0 fixture, push through IPC, recover.

    The audio bytes are a synthetic deterministic pattern derived from
    the segment_id so every replay is byte-identical without needing
    the captured ``audio.wav``.
    """
    seed = fixture["segment_id"].encode("utf-8")  # 64 ASCII bytes
    repeats = (960_000 // len(seed)) + 1
    audio = (seed * repeats)[:960_000]
    assert len(audio) == 960_000  # 30 s × 16 kHz × s16le mono

    # Drive the orchestrator's dispatch path with a thread-safe Queue;
    # the producer/consumer ends are the same process for parity.
    ipc_queue: Queue[Any] = Queue()
    orch = Orchestrator(session_id=fixture["session_id"], ipc_queue=ipc_queue)
    try:
        # Inject the fixture's pre-computed orchestrator state so
        # assemble_segment emits a payload with matching identity.
        orch._segment_window_anchor_utc = None
        orch._active_arm = fixture["_active_arm"]
        orch._expected_greeting = fixture["_expected_greeting"]
        orch._stimulus_time = fixture["_stimulus_time"]
        orch._au12_series = list(fixture["_au12_series"])
        orch._bandit_decision_snapshot = dict(fixture["_bandit_decision_snapshot"])

        payload = orch.assemble_segment(audio, [])

        orch._dispatch_payload(payload)

        assert ipc_queue.qsize() == 1
        raw = ipc_queue.get_nowait()
        msg = InferenceControlMessage.model_validate(raw)

        # Handoff body validates against §6.1.
        validated = InferenceHandoffPayload.model_validate(msg.handoff)
        # The pre-update §6.4 snapshot carries the greeting from the
        # *previous* arm selection — fixture's _expected_greeting is the
        # current segment's, while the snapshot reflects the prior one.
        assert (
            validated.bandit_decision_snapshot.expected_greeting
            == fixture["_bandit_decision_snapshot"]["expected_greeting"]
        )
        assert validated.expected_greeting == fixture["_expected_greeting"]
        assert validated.au12_series[0].timestamp_s == fixture["_au12_series"][0]["timestamp_s"]

        # Audio survives the SharedMemory round trip byte-identically.
        recovered = read_pcm_block(
            PcmBlockMetadata(
                name=msg.audio.name,
                byte_length=msg.audio.byte_length,
                sha256=msg.audio.sha256,
            )
        )
        assert recovered == audio
        assert msg.audio.byte_length == 960_000
        assert len(msg.audio.sha256) == 64

        # forward_fields carries _experiment_code only (no _audio_data,
        # no _frame_data — the desktop path drops the base64 round trip).
        assert "_audio_data" not in msg.forward_fields
        assert "_frame_data" not in msg.forward_fields
    finally:
        orch.close_inflight_blocks()
