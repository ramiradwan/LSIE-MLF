"""Desktop app entry point.

Run with ``python -m services.desktop_app``. The first executable line
inside ``__main__`` is :func:`multiprocessing.freeze_support`, required
on Windows so a frozen PyInstaller bundle does not re-launch itself
when ``mp.Process`` spawns a child. The start method is forced to
``spawn`` everywhere so child processes do not inherit the parent's
loaded modules — the ML import discipline relies on this.

Privacy guards are installed before any child spawns. The same guards
are also installed by each child process at the top of its ``run()`` —
crash-dialog state is per-process on Windows so the parent's call does
not propagate to children.
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from types import FrameType

from services.desktop_app.privacy.crash_dumps import install_crash_privacy_guards
from services.desktop_app.process_graph import ProcessGraph
from services.desktop_launcher import preflight

logger = logging.getLogger(__name__)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m services.desktop_app")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Validate preflight and parent-process startup hooks without launching child processes."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    multiprocessing.set_start_method("spawn", force=True)

    preflight.ensure_preflight()
    install_crash_privacy_guards()
    if args.smoke:
        logger.info("desktop app smoke ok")
        return 0

    graph = ProcessGraph()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %s — initiating cooperative shutdown", signum)
        graph.request_shutdown()

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
    sys.exit(main(sys.argv[1:]))
