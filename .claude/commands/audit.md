Run the 21-item autonomous implementation audit checklist from §13 of the spec (v3.1).

Source of truth: `docs/content.json` (extracted from `docs/tech-spec-v3.1.pdf`). Resolve individual items with:

```bash
python scripts/spec_ref_check.py --resolve "13.<n>" --json
```

For each item below, verify the criterion and report PASS or FAIL with evidence:

1. Directory structure — repo dirs match §3.1 hierarchy.
2. Docker topology — six containers defined with correct images, networks, volumes, restart policies per §9 (redis, postgres, stream_scrcpy, orchestrator, worker, api; SPEC-AMEND-003).
3. IPC lifecycle — named pipe lifecycle per §4.A.2 (create, non-blocking open, write, read, shutdown, crash recovery).
4. Audio pipeline — FFmpeg resample 48→16 kHz with exact command from §4.C.2.
5. Drift correction — ADB poll every 30s, drift_offset formula, fallback strategy per §4.C.1.
6. AU12 implementation — landmark indices [61,291,33,133,362,263], epsilon guard, 5.0 clamp per §7A.
7. LLM determinism — temperature=0, top_p=1.0, seed=42, structured JSON outputs per §8.1.
8. Ephemeral Vault — AES-256-GCM, os.urandom key/nonce, shred -vfz -n 3, 24h deletion per §5.1.
9. Schema validation — InferenceHandoffPayload Pydantic model matches §6.1 JSON Schema.
10. Module contracts — all six modules implement inputs, outputs, deps, side effects, failures per §4.A–F.
11. Error handling — all four failure categories (§12.1–12.4) implemented across all modules.
12. Dependency versions — pinned versions match §10.2 matrix exactly.
13. Variable traceability — all variables from §11 matrix produced by correct module.
14. Data classification — transient/debug/permanent tiers enforced per §5.2.
15. Canonical terminology — grep for retired synonyms from §0.3; must find zero matches.
16. Reward pipeline — Thompson Sampling reward pipeline implements the §7B math recipe: P90 aggregation, binary semantic gate, fractional Beta-Bernoulli posterior updates with Beta(1,1) prior, stimulus window [t+0.5s, t+5.0s].
17. Physiological schema backward compatibility — `_physiological_context` optional on InferenceHandoffPayload; payloads without physiology remain valid; `PhysiologicalSampleEvent` validates against §6.2.
18. Physiological freshness gating — `is_stale` is True when `freshness_s` exceeds `PHYSIO_STALENESS_THRESHOLD_S`; stale samples do not block segment dispatch; Module E excludes stale samples from Co-Modulation Index computation.
19. Physiological data governance — no raw Oura webhook JSON reaches the Persistent Store; only normalized scalar derivatives (RMSSD, HR, freshness, Co-Modulation Index) persisted to `physiology_log` and `comodulation_log`.
20. Co-modulation determinism — returns null when fewer than 4 paired observations exist; uses `scipy.stats.pearsonr`; deterministic for identical input.
21. Reward pipeline invariance — §7B pipeline (`compute_reward`, `ThompsonSamplingEngine.update`) unchanged in v3.1; no physiological data enters reward computation; all v3.0 math recipe tests pass unmodified.

Run the canonical name grep (use `grep -E` — the alternation is extended-regex):

```bash
grep -rnE "Celery node|GPU worker|inference worker|task queue|\bFIFO\b|named pipe|POSIX pipe|audio pipe|kernel pipe|24-hour vault|data vault|transient storage|secure buffer|handoff schema|payload schema|inference payload|FastAPI server|web server|ASGI server|Celery worker|scrcpy container|capture service|stream ingester|relational database" services/ packages/ docker-compose.yml
```

Output a final summary table of all 21 items. Every FAIL must point to either a follow-up commit or a registered entry in `docs/SPEC_AMENDMENTS.md`.
