# LSIE-MLF Performance Baseline Log

Fresh v4-native ledger of deterministic desktop fixture latency measurements captured after feature merges that touch Module C dispatch, IPC/shared-memory handoff, `gpu_ml_worker` analytics publication, `analytics_state_worker` SQLite persistence, or Operator Console read paths. Maintained per Standing Post-Merge Chore #7 (`docs/POST_MERGE_PLAYBOOK.md`).

**Benchmark protocol.** Run `scripts/run_fixture_benchmark.py` against the checked-in deterministic capture fixture. The benchmark constructs `DesktopSegment` payloads, dispatches them through `DesktopSegmentDispatcher`, transfers PCM through shared-memory `InferenceControlMessage` metadata, publishes the active `AnalyticsResultMessage` shape through the `gpu_ml_worker` publication seam, and persists results through `LocalAnalyticsProcessor` into a temporary SQLite database. Live Android, ADB, scrcpy, camera frames, provider APIs, retained worker paths, brokers, and cloud services are not part of this reproducible baseline.

**Regression threshold.** >20% above the previous `v4-fixture:@` row's p95 value for dispatch, ML publish, analytics state, or end-to-end latency triggers investigation before the post-merge cycle can close.

| Date | Commit SHA | Scenario | Segments | Dispatch p50 (ms) | Dispatch p95 (ms) | ML publish p50 (ms) | ML publish p95 (ms) | Analytics state p50 (ms) | Analytics state p95 (ms) | Visual AU12 tick p50 (ms) | End-to-end p95 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-06 | `de15579` | v4-fixture:@ | 3 | 1.252 | 1.671 | 35.587 | 36.003 | 7.745 | 8.431 | 0.043 | 45.354 | v4 desktop IPC/SQLite fixture benchmark; persisted=3; live ADB/scrcpy path not measured; warnings=0 |
