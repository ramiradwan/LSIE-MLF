Run the autonomous implementation audit checklist from §13 of the current spec.

Source of truth: `docs/content.json` (extracted from the single committed `docs/tech-spec-v*.pdf`). Do not use a hardcoded item count; derive the checklist from the current §13 entries on every run.

Resolve individual items with:

```bash
python scripts/spec_ref_check.py --resolve "13.<n>" --json
```

To enumerate the current audit items without assuming a count:

```bash
python - <<'PY'
import json
from pathlib import Path
items = json.loads(Path('docs/content.json').read_text())['audit_checklist']['items']
for item in items:
    print(f"13.{item['item_number']}: {item['audit_item']} — {item['verification_criterion']}")
PY
```

For each current §13 item, verify the criterion and report PASS or FAIL with evidence. Common evidence sources include:

- Repository structure and canonical terminology from §0.3 / §3.
- Data-flow, module-contract, schema, and error-handling checks from §2 / §4 / §6 / §12.
- Data-governance and dependency-pin checks from §5 / §10.
- Mathematical recipe and reward-invariance checks from §7.
- Extension-specific checks for physiology, acoustic analytics, attribution, and deterministic semantic scoring where the current §13 checklist includes them.

Run the canonical-name / retired-synonym grep (use `grep -E` — the alternation is extended-regex). Run it against source paths exactly as shown; do not filter comments or docstrings:

```bash
grep -rnE "Celery node|GPU worker|inference worker|task queue|\bFIFO\b|named pipe|POSIX pipe|audio pipe|kernel pipe|24-hour vault|data vault|transient storage|secure buffer|handoff schema|payload schema|inference payload|FastAPI server|web server|ASGI server|Celery worker|scrcpy container|capture service|stream ingester|relational database|Physiological Chunk Event|Physiological Sample Event|oura event|HRV event|wearable event|physio event|bandit snapshot|decision snapshot|selection snapshot|attribution event|event ledger row|encounter attribution record|conversion event|terminal event|outcome row|attribution link\b|event link\b|causal link row|attribution metric|score row|ledger score|free-form rationale|free-form rationales|free-form semantic rationale|free-form semantic rationales|x[_-]?max[- ]normalized reward|x[_-]?max as reward input|x[_-]?max reward input|\bpitch_f0\b|legacy acoustic scalar|scalar-only acoustic|\[0\.0, 5\.0\].*AU12|AU12.*\[0\.0, 5\.0\]|AU12 clamp.*5\.0|clamp.*AU12.*5\.0" services/ packages/ scripts/
```

Retired phrase → canonical replacement reminders:

- `Physiological Chunk Event`, `Physiological Sample Event`, `oura event`, `HRV event`, `wearable event`, `physio event` → `PhysiologicalChunkEvent`.
- `bandit snapshot`, `decision snapshot`, `selection snapshot` → `BanditDecisionSnapshot`.
- `attribution event`, `event ledger row`, `encounter attribution record` → `AttributionEvent`.
- `conversion event`, `terminal event`, `outcome row` → `OutcomeEvent`.
- `attribution link`, `event link`, `causal link row` → `EventOutcomeLink`.
- `attribution metric`, `score row`, `ledger score` → `AttributionScore`.
- `free-form rationale` / `free-form semantic rationale` → bounded semantic reason code (`bounded_reason_code` / `semantic_reason_code`).
- `x_max-normalized reward`, `x_max as reward input`, `x_max reward input` → §7B `gated_reward` from bounded post-stimulus AU12 plus binary semantic gate; `_x_max` is diagnostic/compatibility telemetry only.
- `pitch_f0` and legacy/scalar-only acoustic-output phrasing for `jitter`/`shimmer` → canonical §7D windowed fields (`f0_*`, `jitter_mean_*`, `jitter_delta`, `shimmer_mean_*`, `shimmer_delta`).
- `[0.0, 5.0]` AU12 clamp phrasing → bounded AU12 intensity interval `[0.0, 1.0]`.

Output a final summary table covering every current §13 item. Every FAIL must point to either a follow-up commit or a registered entry in `docs/SPEC_AMENDMENTS.md`.
