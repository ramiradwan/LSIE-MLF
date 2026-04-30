"""Desktop app entry point (v4.0 §9 / WS3 P1).

Run with ``python -m services.desktop_app``. The first executable line
inside ``__main__`` is :func:`multiprocessing.freeze_support`, required
on Windows so a frozen PyInstaller bundle does not re-launch itself
when ``mp.Process`` spawns a child. The start method is forced to
``spawn`` everywhere so child processes do not inherit the parent's
loaded modules — the ML import discipline relies on this.
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
from types import FrameType

from services.desktop_app.process_graph import ProcessGraph

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    multiprocessing.set_start_method("spawn", force=True)

    graph = ProcessGraph()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %s — initiating cooperative shutdown", signum)
        graph.stop_all()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    graph.start_all()
    try:
        graph.wait()
    finally:
        graph.stop_all()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
