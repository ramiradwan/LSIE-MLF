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

# 5. Strict §13 audit harness — mirrors the first-class CI audit job.
echo "── §13 audit harness ──"
if python scripts/run_audit.py --strict; then
    pass "§13 audit harness"
else
    fail "§13 audit harness"
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


echo "═══════════════════════════════════════"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN} All checks passed${NC}"
else
    echo -e "${RED} Some checks failed${NC}"
fi
echo "═══════════════════════════════════════"

exit $EXIT_CODE