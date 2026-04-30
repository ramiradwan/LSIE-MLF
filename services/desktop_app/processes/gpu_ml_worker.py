"""GPU ML worker process (v4.0 §4.D / WS3 P1 stub).

This is the **only** module in the desktop graph that may import
``torch``, ``mediapipe``, ``faster_whisper``, or ``ctranslate2``. The
parent process never imports this module — it is launched by string
through :func:`services.desktop_app.process_graph._launch`, so the ML
libraries are pulled into a dedicated child process and never into the
UI / API / orchestrator / state / cloud-sync surfaces.

WS3 P2 will wire ``run`` to consume IPC control messages and the
SharedMemory PCM blocks; WS3 P3 + WS2 P1/P2 settle the actual ML
runtime call (``WhisperModel`` + ``FaceMeshProcessor`` + cross-encoder).
For Phase 1 the imports are present at module top level so the
isolation contract is provable, and ``run`` is a stub.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

# The four ML library imports below are intentional at module top
# level: they prove the v4.0 §9 isolation contract. The canary test
# imports each non-ML process module in a clean subprocess and asserts
# these names are absent from ``sys.modules``; re-importing this
# module DOES bring them in — that is the whole point of routing ML
# inference into a dedicated child process.
import ctranslate2  # noqa: F401
import faster_whisper  # noqa: F401
import mediapipe  # noqa: F401
import torch  # noqa: F401

logger = logging.getLogger(__name__)


def run(shutdown_event: mpsync.Event) -> None:
    logger.info("gpu_ml_worker started")
    shutdown_event.wait()
    logger.info("gpu_ml_worker stopped")
