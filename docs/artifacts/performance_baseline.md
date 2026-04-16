# LSIE-MLF Performance Baseline Log

Append-only ledger of latency and throughput measurements captured after feature merges that touch the orchestrator, inference, analytics, transcription, face-mesh, or IPC paths. Maintained per Standing Post-Merge Chore #7 (`docs/POST_MERGE_PLAYBOOK.md`).

**Benchmark protocol.** Standard 30-second segment benchmark against a recorded capture fixture (no live device). Run the orchestrator with `AUTO_STIMULUS_DELAY_S=0`, inject a synthetic stimulus at t=2.0s, and record the assembly wall-clock log line for segments 1..10. Whisper INT8 timer is read from `services/worker/tasks/inference.py` log line at segment completion. AU12 per-frame p50 from `packages/ml_core/au12.py` timer. For physiology-aware runs, prime Redis `physio:events` with 20 synthetic Oura samples per subject_role at 300s cadence before starting the orchestrator.

**Regression threshold.** >20% above the previous row's p95 value triggers investigation before the post-merge cycle can close.

| Date | Commit SHA | Cycle / PR | Segment-assembly p50 (ms) | Segment-assembly p95 (ms) | ML inference p50 (ms) | ML inference p95 (ms) | AU12 per-frame p50 (ms) | Co-Modulation window compute (ms) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-16 | `60be7ec` | PR 91 — `feature/v31-physio-comodulation` | TBD | TBD | TBD | TBD | TBD | TBD | Physiology ingress + co-modulation added. Benchmark TBD pending live stack (no GPU in hardening env). Follow-up ADO work item required to populate the row. |
