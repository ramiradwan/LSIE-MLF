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

# 3. Mypy — scope and flags MUST match ci.yml lint-and-typecheck job
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

# 5. Canonical terminology audit (§0.3) — file-type filter matches ci.yml
echo "── Canonical terminology audit ──"
RETIRED_TERMS="Celery node|GPU worker|inference worker|task queue|FIFO|named pipe|POSIX pipe|audio pipe|kernel pipe|24-hour vault|data vault|transient storage|secure buffer|handoff schema|payload schema|inference payload|FastAPI server|web server|ASGI server|Celery worker|scrcpy container|capture service|stream ingester|relational database"
MATCHES=$(grep -rnE "$RETIRED_TERMS" services/ packages/ docker-compose.yml \
    --include='*.py' --include='*.yml' --include='*.yaml' --include='*.sh' --include='*.txt' \
    2>/dev/null || true)
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