"""IPC primitives for the v4.0 desktop process graph (WS3 P2).

The Celery + Redis dispatch path is replaced by ``multiprocessing.Queue``
for control messages and ``multiprocessing.shared_memory.SharedMemory``
for the 30-second PCM windows. The base64-in-JSON ``_audio_data`` round
trip is dropped on the desktop path. ``sanitize_json_payload`` remains
in use because it does more than base64 (it prunes empty
``_physiological_context`` and absent bandit-snapshot optionals).

Modules:
    shared_buffers   Producer/consumer wrappers around SharedMemory blocks
                     named ``lsie_ipc_pcm_{uuid4}`` with SHA-256 integrity.
    control_messages Pydantic envelope carrying the validated handoff
                     payload + the SharedMemory metadata.
    cleanup          Dirty State Recovery sweep that unlinks orphan
                     blocks left behind by an ungraceful parent crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.queues import Queue


@dataclass
class IpcChannels:
    """The set of multiprocessing channels shared across the graph.

    Constructed by the parent before spawn and pickled into each child
    via the launcher. Phase 2 wired ``ml_inbox`` (orchestrator →
    gpu_ml_worker). Phase 3 adds ``drift_updates`` (capture_supervisor
    → module_c_orchestrator). Future phases add ``analytics_inbox``
    (gpu_ml_worker → analytics_state_worker) and ``cloud_outbox``
    (analytics_state_worker → cloud_sync_worker).
    """

    ml_inbox: Queue[object] = field(repr=False)
    drift_updates: Queue[object] = field(repr=False)
