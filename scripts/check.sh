#!/usr/bin/env bash
# =============================================================================
# LSIE-MLF Local CI Check
#
# Mirrors .github/workflows/ci.yml for local pre-push validation.
# Run from the project root with the virtualenv activated.
# Usage: bash scripts/check.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; EXIT_CODE=1; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }

EXIT_CODE=0

echo "═══════════════════════════════════════"
echo " LSIE-MLF Local CI Check"
echo "═══════════════════════════════════════"
echo ""

# Each gate mirrors .github/workflows/ci.yml exactly. Any drift here means a
# green local run can still fail on GitHub. When updating CI, update this too.

# 1. Ruff lint
echo "── Ruff lint ──"
if ruff check packages/ services/ tests/; then
    pass "Ruff lint"
else
    fail "Ruff lint"
fi
echo ""

# 2. Ruff format
echo "── Ruff format ──"
if ruff format --check packages/ services/ tests/; then
    pass "Ruff format"
else
    fail "Ruff format"
fi
echo ""

# 3. Mypy — scope and flags MUST match ci.yml lint-and-typecheck job.
# The local venv must also have PySide6 installed (via requirements/cli.txt
# or `pip install PySide6`); without it, every QObject/QWidget subclass in
# services/operator_console resolves to Any and mypy fails the same way CI
# would. CI installs PySide6 explicitly in the lint-and-typecheck job.
echo "── Mypy type check ──"
if mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases; then
    pass "Mypy type check"
else
    fail "Mypy type check"
fi
echo ""

# 4. Pytest
echo "── Pytest ──"
if python -m pytest tests/ -x -q --tb=short; then
    pass "Pytest"
else
    fail "Pytest"
fi
echo ""

# 5. Canonical terminology audit (§0.3) — scope/pattern matches .claude/commands/audit.md item 15
echo "── Canonical terminology audit ──"
RETIRE_PARTS=(
    "Celery n"'ode'
    "GPU work"'er'
    "inference work"'er'
    "task que"'ue'
    "\bFI"'FO\b'
    "named pi"'pe'
    "POSIX pi"'pe'
    "audio pi"'pe'
    "kernel pi"'pe'
    "24-hour vau"'lt'
    "data vau"'lt'
    "transient stor"'age'
    "secure buff"'er'
    "handoff sche"'ma'
    "payload sche"'ma'
    "inference pay"'load'
    "FastAPI serv"'er'
    "web serv"'er'
    "ASGI serv"'er'
    "Celery work"'er'
    "scrcpy contain"'er'
    "capture serv"'ice'
    "stream ingest"'er'
    "relational data"'base'
    "Physiological Chunk Even"'t'
    "Physiological Sample Even"'t'
    "oura even"'t'
    "HRV even"'t'
    "wearable even"'t'
    "physio even"'t'
    "bandit snap"'shot'
    "decision snap"'shot'
    "selection snap"'shot'
    "attribution even"'t'
    "event ledger r"'ow'
    "encounter attribution rec"'ord'
    "conversion even"'t'
    "terminal even"'t'
    "outcome r"'ow'
    "attribution lin"'k\b'
    "event lin"'k\b'
    "causal link r"'ow'
    "attribution metr"'ic'
    "score r"'ow'
    "ledger sco"'re'
    "free-form ration"'ale'
    "free-form ration"'ales'
    "free-form semantic ration"'ale'
    "free-form semantic ration"'ales'
    "x[_-]?max[- ]normalized reward"
    "x[_-]?max as reward input"
    "x[_-]?max reward input"
    "\bpitch_f"'0\b'
    "legacy acoustic scal"'ar'
    "scalar-only acous"'tic'
    "\[0\.0, 5\.0\].*AU"'12'
    "AU"'12.*\[0\.0, 5\.0\]'
    "AU"'12 clamp.*5\.0'
    "clamp.*AU"'12.*5\.0'
)
RETIRED_TERMS=$(IFS='|'; printf '%s' "${RETIRE_PARTS[*]}")
MATCHES=$(grep -rnE "$RETIRED_TERMS" services/ packages/ scripts/ 2>/dev/null || true)
if [ -z "$MATCHES" ]; then
    pass "No retired synonyms found"
else
    fail "Retired synonyms found (§0.3 violation):"
    echo "$MATCHES"
fi
echo ""

# 6. Docker compose validation
echo "── Docker compose config ──"
if docker compose config --quiet 2>/dev/null; then
    pass "Docker compose config"
else
    warn "Docker compose config failed (Docker may not be running)"
fi
echo ""

# 7. Schema consistency check (Pydantic vs SQL files vs Python DDL string vs content.json)
echo "── Schema consistency check ──"
if python scripts/check_schema_consistency.py; then
    pass "Schema consistency check"
else
    fail "Schema consistency check"
fi
echo ""

# 8. Dependency pin check
echo "── Dependency pin check ──"
UNPINNED=0
for f in requirements/base.txt requirements/api.txt requirements/worker.txt requirements/cli.txt; do
    if grep -E "^[a-zA-Z]" "$f" 2>/dev/null | grep -v -E "==|>=|~=|\*" | grep -v "^-r" | grep -v "^#" | grep -q .; then
        fail "Unpinned dependency in $f"
        UNPINNED=1
    fi
done
if [ "$UNPINNED" -eq 0 ]; then
    pass "All dependencies pinned"
fi
echo ""

echo "═══════════════════════════════════════"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN} All checks passed${NC}"
else
    echo -e "${RED} Some checks failed${NC}"
fi
echo "═══════════════════════════════════════"

exit $EXIT_CODE