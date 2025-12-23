# PowerShell script for Windows users - equivalent to Makefile targets

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function Install-All {
    Write-Host "Installing all dependencies..." -ForegroundColor Cyan
    Install-Backend
    Install-Frontend
    Install-E2E
}

function Install-Backend {
    Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
    Push-Location backend
    python -m pip install -e ".[dev]"
    Pop-Location
}

function Install-Frontend {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location frontend
    npm ci
    Pop-Location
}

function Install-E2E {
    Write-Host "Installing E2E dependencies..." -ForegroundColor Cyan
    Push-Location e2e
    npm ci
    npx playwright install chromium --with-deps
    Pop-Location
}

function Test-All {
    Write-Host "Running all tests..." -ForegroundColor Cyan
    Test-Backend
    Test-Frontend
}

function Test-Backend {
    Write-Host "Running backend tests with coverage..." -ForegroundColor Cyan
    Push-Location backend
    pytest --cov=tunix_rt_backend --cov-branch --cov-report=term --cov-report=json:coverage.json -v
    python tools/coverage_gate.py
    Pop-Location
}

function Test-Frontend {
    Write-Host "Running frontend tests..." -ForegroundColor Cyan
    Push-Location frontend
    npm run test
    Pop-Location
}

function Test-E2E {
    Write-Host "Running E2E tests (mock mode)..." -ForegroundColor Cyan
    Push-Location e2e
    $env:REDIAI_MODE = "mock"
    npx playwright test
    Pop-Location
}

function Invoke-Lint {
    Write-Host "Running linters..." -ForegroundColor Cyan
    Push-Location backend
    ruff check .
    mypy tunix_rt_backend
    Pop-Location
}

function Invoke-Format {
    Write-Host "Formatting backend code..." -ForegroundColor Cyan
    Push-Location backend
    ruff format .
    Pop-Location
}

function Start-Docker {
    Write-Host "Starting Docker Compose services..." -ForegroundColor Cyan
    docker compose up -d
}

function Stop-Docker {
    Write-Host "Stopping Docker Compose services..." -ForegroundColor Cyan
    docker compose down
}

function Show-DockerLogs {
    Write-Host "Showing Docker Compose logs..." -ForegroundColor Cyan
    docker compose logs -f
}

function Build-Frontend {
    Write-Host "Building frontend for production..." -ForegroundColor Cyan
    Push-Location frontend
    npm run build
    Pop-Location
}

function Clear-Generated {
    Write-Host "Cleaning generated files..." -ForegroundColor Cyan
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -File -Filter "*.pyc" | Remove-Item -Force
    if (Test-Path "backend/*.egg-info") { Remove-Item "backend/*.egg-info" -Recurse -Force }
    if (Test-Path "backend/coverage.json") { Remove-Item "backend/coverage.json" }
    if (Test-Path "frontend/dist") { Remove-Item "frontend/dist" -Recurse -Force }
    if (Test-Path "e2e/playwright-report") { Remove-Item "e2e/playwright-report" -Recurse -Force }
}

function Show-Help {
    Write-Host "`nAvailable commands:" -ForegroundColor Green
    Write-Host "  install          - Install all dependencies" -ForegroundColor Cyan
    Write-Host "  install-backend  - Install backend dependencies" -ForegroundColor Cyan
    Write-Host "  install-frontend - Install frontend dependencies" -ForegroundColor Cyan
    Write-Host "  install-e2e      - Install E2E dependencies" -ForegroundColor Cyan
    Write-Host "  test             - Run all tests (excluding E2E)" -ForegroundColor Cyan
    Write-Host "  test-backend     - Run backend tests with coverage" -ForegroundColor Cyan
    Write-Host "  test-frontend    - Run frontend tests" -ForegroundColor Cyan
    Write-Host "  test-e2e         - Run E2E tests (mock mode)" -ForegroundColor Cyan
    Write-Host "  lint             - Run backend linting and type checking" -ForegroundColor Cyan
    Write-Host "  format           - Format backend code" -ForegroundColor Cyan
    Write-Host "  docker-up        - Start Docker Compose services" -ForegroundColor Cyan
    Write-Host "  docker-down      - Stop Docker Compose services" -ForegroundColor Cyan
    Write-Host "  docker-logs      - Show Docker Compose logs" -ForegroundColor Cyan
    Write-Host "  build-frontend   - Build frontend for production" -ForegroundColor Cyan
    Write-Host "  clean            - Clean generated files" -ForegroundColor Cyan
    Write-Host "`nUsage: .\dev.ps1 <command>" -ForegroundColor Yellow
}

# Main execution
switch ($Target.ToLower()) {
    "install" { Install-All }
    "install-backend" { Install-Backend }
    "install-frontend" { Install-Frontend }
    "install-e2e" { Install-E2E }
    "test" { Test-All }
    "test-backend" { Test-Backend }
    "test-frontend" { Test-Frontend }
    "test-e2e" { Test-E2E }
    "lint" { Invoke-Lint }
    "format" { Invoke-Format }
    "docker-up" { Start-Docker }
    "docker-down" { Stop-Docker }
    "docker-logs" { Show-DockerLogs }
    "build-frontend" { Build-Frontend }
    "clean" { Clear-Generated }
    "help" { Show-Help }
    default {
        Write-Host "Unknown command: $Target" -ForegroundColor Red
        Show-Help
    }
}
