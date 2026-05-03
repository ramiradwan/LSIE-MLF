# =============================================================================
# LSIE-MLF Local CI Check (PowerShell)
#
# Mirrors .github/workflows/ci.yml for local pre-push validation.
# Run from the project root. Uses `uv run` so the active shell does not
# need to have the venv pre-activated.
# Usage: .\scripts\check.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$exitCode = 0

function Pass($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "  $msg" -ForegroundColor Red; $script:exitCode = 1 }
function Warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host " LSIE-MLF Local CI Check" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Each gate mirrors .github/workflows/ci.yml exactly. Any drift here means a
# green local run can still fail on GitHub. When updating CI, update this too.

# 1. Ruff lint
Write-Host "-- Ruff lint --"
uv run ruff check packages/ services/ tests/
if ($LASTEXITCODE -eq 0) { Pass "Ruff lint" } else { Fail "Ruff lint" }
Write-Host ""

# 2. Ruff format
Write-Host "-- Ruff format --"
uv run ruff format --check packages/ services/ tests/
if ($LASTEXITCODE -eq 0) { Pass "Ruff format" } else { Fail "Ruff format" }
Write-Host ""

# 3. Mypy -- scope and flags MUST match ci.yml lint-and-typecheck job.
Write-Host "-- Mypy type check --"
uv run mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases
if ($LASTEXITCODE -eq 0) { Pass "Mypy type check" } else { Fail "Mypy type check" }
Write-Host ""

# 4. Pytest
Write-Host "-- Pytest --"
uv run pytest tests/ -x -q --tb=short
if ($LASTEXITCODE -eq 0) { Pass "Pytest" } else { Fail "Pytest" }
Write-Host ""

# 5. Strict §13 audit harness -- mirrors the first-class CI audit job.
Write-Host "-- §13 audit harness --"
uv run python scripts/run_audit.py --strict
if ($LASTEXITCODE -eq 0) { Pass "§13 audit harness" } else { Fail "§13 audit harness" }
Write-Host ""

# 6. Canonical terminology audit
Write-Host "-- Canonical terminology audit --"
$canonicalPattern = 'GPU worker|video pipe|free-form rationale|free-form rationales|free-form semantic rationale|free-form semantic rationales|x[_-]?max[- ]normalized reward|x[_-]?max as reward input|x[_-]?max reward input|\bpitch_f0\b|legacy acoustic scalar|scalar-only acoustic|\[0\.0, 5\.0\].*AU12|AU12.*\[0\.0, 5\.0\]|AU12 clamp.*5\.0|clamp.*AU12.*5\.0'
$canonicalHits = Select-String -Path services\**\*,packages\**\*,scripts\**\* -Pattern $canonicalPattern -AllMatches -ErrorAction SilentlyContinue
if ($canonicalHits) { Fail "Canonical terminology audit" } else { Pass "Canonical terminology audit" }
Write-Host ""

# 7. Schema consistency check (Pydantic vs SQL files vs Python DDL string vs content.json)
Write-Host "-- Schema consistency check --"
uv run python scripts/check_schema_consistency.py
if ($LASTEXITCODE -eq 0) { Pass "Schema consistency check" } else { Fail "Schema consistency check" }
Write-Host ""


Write-Host "=======================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host " All checks passed" -ForegroundColor Green
} else {
    Write-Host " Some checks failed" -ForegroundColor Red
}
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

exit $exitCode
