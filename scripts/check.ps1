# =============================================================================
# LSIE-MLF Local CI Check (PowerShell)
#
# Mirrors .github/workflows/ci.yml for local pre-push validation.
# Run from the project root with the virtualenv activated.
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
ruff check packages/ services/ tests/
if ($LASTEXITCODE -eq 0) { Pass "Ruff lint" } else { Fail "Ruff lint" }
Write-Host ""

# 2. Ruff format
Write-Host "-- Ruff format --"
ruff format --check packages/ services/ tests/
if ($LASTEXITCODE -eq 0) { Pass "Ruff format" } else { Fail "Ruff format" }
Write-Host ""

# 3. Mypy -- scope and flags MUST match ci.yml lint-and-typecheck job.
# The local venv must also have PySide6 installed (via requirements/cli.txt
# or `pip install PySide6`); without it, every QObject/QWidget subclass in
# services/operator_console resolves to Any and mypy fails the same way CI
# would. CI installs PySide6 explicitly in the lint-and-typecheck job.
Write-Host "-- Mypy type check --"
mypy packages/ services/ tests/ --python-version 3.11 --ignore-missing-imports --explicit-package-bases
if ($LASTEXITCODE -eq 0) { Pass "Mypy type check" } else { Fail "Mypy type check" }
Write-Host ""

# 4. Pytest
Write-Host "-- Pytest --"
python -m pytest tests/ -x -q --tb=short
if ($LASTEXITCODE -eq 0) { Pass "Pytest" } else { Fail "Pytest" }
Write-Host ""

# 5. Canonical terminology audit -- scope/pattern matches .claude/commands/audit.md item 15
# NOTE: $matches is a PowerShell automatic variable; use a different name.
Write-Host "-- Canonical terminology audit --"
$termParts = @(
    "Celery n" + "ode",
    "GPU work" + "er",
    "inference work" + "er",
    "task que" + "ue",
    "\bFI" + "FO\b",
    "named pi" + "pe",
    "POSIX pi" + "pe",
    "audio pi" + "pe",
    "kernel pi" + "pe",
    "24-hour vau" + "lt",
    "data vau" + "lt",
    "transient stor" + "age",
    "secure buff" + "er",
    "handoff sche" + "ma",
    "payload sche" + "ma",
    "inference pay" + "load",
    "FastAPI serv" + "er",
    "web serv" + "er",
    "ASGI serv" + "er",
    "Celery work" + "er",
    "scrcpy contain" + "er",
    "capture serv" + "ice",
    "stream ingest" + "er",
    "relational data" + "base",
    "Physiological Chunk Even" + "t",
    "Physiological Sample Even" + "t",
    "oura even" + "t",
    "HRV even" + "t",
    "wearable even" + "t",
    "physio even" + "t",
    "bandit snap" + "shot",
    "decision snap" + "shot",
    "selection snap" + "shot",
    "attribution even" + "t",
    "event ledger r" + "ow",
    "encounter attribution rec" + "ord",
    "conversion even" + "t",
    "terminal even" + "t",
    "outcome r" + "ow",
    "attribution lin" + "k\b",
    "event lin" + "k\b",
    "causal link r" + "ow",
    "attribution metr" + "ic",
    "score r" + "ow",
    "ledger sco" + "re",
    "free-form ration" + "ale",
    "free-form ration" + "ales",
    "free-form semantic ration" + "ale",
    "free-form semantic ration" + "ales",
    "x[_-]?max[- ]normalized reward",
    "x[_-]?max as reward input",
    "x[_-]?max reward input",
    "\bpitch_f" + "0\b",
    "legacy acoustic scal" + "ar",
    "scalar-only acous" + "tic",
    "\[0\.0, 5\.0\].*AU" + "12",
    "AU" + "12.*\[0\.0, 5\.0\]",
    "AU" + "12 clamp.*5\.0",
    "clamp.*AU" + "12.*5\.0"
)
$pattern = $termParts -join "|"
$roots = @("services", "packages", "scripts")
$scanFiles = @(Get-ChildItem -Path $roots -Recurse -File -ErrorAction SilentlyContinue)
# -CaseSensitive mirrors `grep -E` semantics used by check.sh and audit.md.
# The recursive file list intentionally matches audit.md's services/ packages/
# scripts/ scope without file-type or comment/docstring filtering.
$retiredHits = @()
if ($scanFiles.Count -gt 0) {
    $retiredHits = @($scanFiles | Select-String -Pattern $pattern -CaseSensitive -ErrorAction SilentlyContinue)
}
if ($retiredHits.Count -eq 0) { Pass "No retired synonyms found" } else { Fail "Retired synonyms found:"; $retiredHits | ForEach-Object { Write-Host "    $_" } }
Write-Host ""

# 6. Docker compose
Write-Host "-- Docker compose config --"
docker compose config --quiet 2>$null
if ($LASTEXITCODE -eq 0) { Pass "Docker compose config" } else { Warn "Docker compose config failed (Docker may not be running)" }
Write-Host ""

# 7. Schema consistency check (Pydantic vs SQL files vs Python DDL string vs content.json)
Write-Host "-- Schema consistency check --"
python scripts/check_schema_consistency.py
if ($LASTEXITCODE -eq 0) { Pass "Schema consistency check" } else { Fail "Schema consistency check" }
Write-Host ""

# 8. Dependency pin check
Write-Host "-- Dependency pin check --"
$unpinned = $false
foreach ($f in @("requirements\base.txt", "requirements\api.txt", "requirements\worker.txt", "requirements\cli.txt")) {
    $bad = Get-Content $f | Where-Object { $_ -match "^[a-zA-Z]" -and $_ -notmatch "==|>=|~=|\*" -and $_ -notmatch "^-r" -and $_ -notmatch "^#" }
    if ($bad) { Fail "Unpinned dependency in $f"; $unpinned = $true }
}
if (-not $unpinned) { Pass "All dependencies pinned" }
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