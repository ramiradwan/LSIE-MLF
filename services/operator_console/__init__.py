"""LSIE-MLF Operator Console — production-grade PySide6 dashboard (§4.E.1).

This package is the operator-facing counterpart to
`scripts/debug_studio.py`. Debug Studio is a single-file developer tool
for live pipeline introspection (video, landmarks, AU12 normalization,
diagnostic polling); the Operator Console is a modular, multi-page
surface designed for routine operation of the live inference stack.

Pages (SPEC-AMEND-008):
    Overview      — active session, experiment, physiology, health, and
                    the attention queue in a single glance.
    Live Session  — encounter timeline with per-segment reward
                    explanation (§7B p90_intensity × semantic_gate) and
                    physiology freshness (§4.C.4).
    Experiments   — Thompson Sampling posteriors per arm with evaluation
                    variance and the latest update summary (§7B).
    Physiology    — operator and streamer RMSSD, heart rate, freshness,
                    and the §7C Co-Modulation Index (null-valid is a
                    legitimate outcome, not an error).
    Health        — subsystem status rollup that distinguishes degraded
                    vs recovering vs error states per §12 with
                    operator-action hints.
    Sessions      — recent-sessions history for drill-down.

Persistent stimulus rail:
    A single `ActionBar` mounts below the content area and is the
    console's only write path. Submission is a one-shot via the polling
    coordinator; idempotency is keyed by `client_action_id` and the
    authoritative `stimulus_time` stays orchestrator-owned per §4.C.

See SPEC-AMEND-008 in `docs/SPEC_AMENDMENTS.md` for the amendment that
retired the Streamlit dashboard in favor of this multi-page PySide6
layout.

Entry point:
    python -m services.operator_console
"""

from __future__ import annotations
