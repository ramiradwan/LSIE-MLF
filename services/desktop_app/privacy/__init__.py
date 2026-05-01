"""Privacy / data-governance helpers for the v4.0 desktop graph (WS4 P3).

The two surfaces that live here:

* :mod:`zeroize` — wipe transient PCM SharedMemory before unlink so a
  leaked block cannot be scavenged from ``/dev/shm`` or a Windows
  page-file artefact.
* :mod:`crash_dumps` — disable the OS modal crash dialog and exclude
  our app binary from Windows Error Reporting LocalDumps so a crash
  mid-segment does not freeze raw PCM into a ``.dmp`` outside the
  Ephemeral Vault perimeter.

All Win32 / POSIX branching is delegated to
:mod:`services.desktop_app.os_adapter` per the Platform Abstraction
Rule. These wrappers exist so the call sites in the privacy/data path
stay OS-agnostic and the spec wording reads cleanly.
"""

from __future__ import annotations
