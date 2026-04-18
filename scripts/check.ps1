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

# 5. Canonical terminology audit -- file types MUST match ci.yml scan
# NOTE: $matches is a PowerShell automatic variable; use a different name.
Write-Host "-- Canonical terminology audit --"
$pattern = "Celery node|GPU worker|inference worker|task queue|FIFO|named pipe|POSIX pipe|audio pipe|kernel pipe|24-hour vault|data vault|transient storage|secure buffer|handoff schema|payload schema|inference payload|FastAPI server|web server|ASGI server|Celery worker|scrcpy container|capture service|stream ingester|relational database"
$paths = @(
    "services\**\*.py","services\**\*.yml","services\**\*.yaml","services\**\*.sh","services\**\*.txt",
    "packages\**\*.py","packages\**\*.yml","packages\**\*.yaml","packages\**\*.sh","packages\**\*.txt",
    "docker-compose.yml"
)
$retiredHits = Select-String -Path $paths -Pattern $pattern -ErrorAction SilentlyContinue
if ($null -eq $retiredHits) { Pass "No retired synonyms found" } else { Fail "Retired synonyms found:"; $retiredHits | ForEach-Object { Write-Host "    $_" } }
Write-Host ""

# 6. Docker compose
Write-Host "-- Docker compose config --"
docker compose config --quiet 2>$null
if ($LASTEXITCODE -eq 0) { Pass "Docker compose config" } else { Warn "Docker compose config failed (Docker may not be running)" }
Write-Host ""

# 7. Dependency pin check
Write-Host "-- Dependency pin check --"
$unpinned = $false
foreach ($f in @("requirements\base.txt", "requirements\api.txt", "requirements\worker.txt")) {
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