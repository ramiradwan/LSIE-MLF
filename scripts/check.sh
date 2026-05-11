#!/usr/bin/env bash
# =============================================================================
# LSIE-MLF Local CI Check
#
# Mirrors .github/workflows/ci.yml for local pre-push validation.
# Run from the project root. Uses `uv run` so the active shell does not
# need to have the venv pre-activated.
# Usage: bash scripts/check.sh
# =============================================================================

set -e

export PATH="$HOME/.local/bin:$USERPROFILE/.local/bin:$PATH"

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
if uv run ruff check packages/ services/ tests/ automation/; then
    pass "Ruff lint"
else
    fail "Ruff lint"
fi
echo ""

# 2. Ruff format
echo "── Ruff format ──"
if uv run ruff format --check packages/ services/ tests/ automation/; then
    pass "Ruff format"
else
    fail "Ruff format"
fi
echo ""

# 3. Mypy — scope and flags MUST match ci.yml lint-and-typecheck job.
echo "── Mypy type check ──"
if uv run mypy packages/ services/ tests/ automation/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases; then
    pass "Mypy type check"
else
    fail "Mypy type check"
fi
echo ""

# 4. Pytest
echo "── Pytest ──"
if uv run pytest tests/ -x -q --tb=short; then
    pass "Pytest"
else
    fail "Pytest"
fi
echo ""

# 5. Strict §13 audit harness — mirrors the first-class CI audit job.
echo "── §13 audit harness ──"
if uv run python scripts/run_audit.py --strict; then
    pass "§13 audit harness"
else
    fail "§13 audit harness"
fi
echo ""

# 6. Canonical terminology audit
echo "── Canonical terminology audit ──"
if grep -rnE \
    --exclude='check.sh' \
    --exclude='check.ps1' \
    --exclude='mechanical.py' \
    --exclude='*.pyc' \
    --exclude-dir='__pycache__' \
    'GPU worker|video pipe|free-form rationale|free-form rationales|free-form semantic rationale|free-form semantic rationales|x[_-]?max[- ]normalized reward|x[_-]?max as reward input|x[_-]?max reward input|\bpitch_f0\b|legacy acoustic scalar|scalar-only acoustic|\[0\.0, 5\.0\].*AU12|AU12.*\[0\.0, 5\.0\]|AU12 clamp.*5\.0|clamp.*AU12.*5\.0' \
    services/ packages/ scripts/; then
    fail "Canonical terminology audit"
else
    pass "Canonical terminology audit"
fi
echo ""

# 7. Schema consistency check (Pydantic vs extracted JSON Schema vs cloud PostgreSQL DDL)
echo "── Schema consistency check ──"
if uv run python scripts/check_schema_consistency.py; then
    pass "Schema consistency check"
else
    fail "Schema consistency check"
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
