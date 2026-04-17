"""LSIE-MLF Operator Console — production-grade PySide6 dashboard (§4.E.1).

This package is the operator-facing counterpart to `scripts/debug_studio.py`.
The Debug Studio is a single-file developer tool for live pipeline
introspection (video, landmarks, AU12 normalization, diagnostic polling);
the Operator Console is split into clean modules so it can be extended
safely over time — session monitoring, experiment readback, physiology
views, and future operator actions (external-wearable OAuth, device
configuration, stimulus buttons).

See SPEC-AMEND-008 in `docs/SPEC_AMENDMENTS.md` for the amendment that
retired the original Streamlit dashboard in favor of this two-surface
PySide6 layout.

Entry point:
    python -m services.operator_console
"""

from __future__ import annotations
