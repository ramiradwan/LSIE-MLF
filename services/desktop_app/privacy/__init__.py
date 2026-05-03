"""Privacy and data-governance helpers for the desktop graph.

The surfaces that live here:

* :mod:`zeroize` — wipe transient PCM SharedMemory before unlink so a
  leaked block cannot be scavenged from ``/dev/shm`` or a Windows
  page-file artefact.
* :mod:`crash_dumps` — disable the OS modal crash dialog and exclude
  our app binary from Windows Error Reporting LocalDumps so a crash
  mid-segment does not freeze raw PCM into a ``.dmp`` outside the
  Ephemeral Vault perimeter.
* :mod:`secrets` — persist long-lived credentials (cloud OAuth refresh
  token, Oura webhook secret slot) in the OS secret store so no secret
  literal ships inside the signed binary or the operator-readable
  install tree.

All Win32 / POSIX branching is delegated to
:mod:`services.desktop_app.os_adapter` per the Platform Abstraction
Rule. These wrappers exist so the call sites in the privacy/data path
stay OS-agnostic and the spec wording reads cleanly.
"""

from __future__ import annotations
