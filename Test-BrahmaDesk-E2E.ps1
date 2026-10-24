[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipUnitTests,
    [switch]$KeepStartedBackend
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DesktopDir = Join-Path $ProjectRoot 'desktop'
$BackendTests = Join-Path $ProjectRoot 'backend\app\tests\test_settings_pipeline.py'
$ResultsDir = Join-Path $ProjectRoot 'test-results\e2e'
$BackendLog = Join-Path $ResultsDir 'backend.stdout.log'
$BackendErr = Join-Path $ResultsDir 'backend.stderr.log'
$BackendUrl = if ($env:BRAHMA_API_URL) { $env:BRAHMA_API_URL.TrimEnd('/') } else { 'http://127.0.0.1:8000' }
$PythonCommand = Get-Command python -ErrorAction Stop
$NodeCommand = Get-Command node -ErrorAction Stop
$NpmCommand = Get-Command npm -ErrorAction Stop
$StartedBackend = $null

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

function Invoke-CheckedCommand([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory, [string]$Label) {
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}

function Get-BackendHealth {
    try {
        return Invoke-RestMethod -Uri "$BackendUrl/api/health" -TimeoutSec 3 -ErrorAction Stop
    }
    catch { return $null }
}

function Wait-ForBackend([int]$TimeoutSeconds = 240) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $health = Get-BackendHealth
        if ($health -and $health.backend -eq 'ok' -and $health.llama_server -eq 'ok') { return $health }
        if ($StartedBackend -and $StartedBackend.HasExited) {
            throw "Backend process exited early. Check logs: $BackendLog and $BackendErr"
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for a healthy backend and llama.cpp server at $BackendUrl. Check logs: $BackendLog and $BackendErr"
}

try {
    Write-Host 'Brahma Desk — automated backend + desktop end-to-end test' -ForegroundColor Green
    Write-Host "Project: $ProjectRoot"
    Write-Host "Python: $($PythonCommand.Source)"
    Write-Host "Node: $(& $NodeCommand.Source --version)"
    Write-Host "Backend: $BackendUrl"

    if (-not $SkipUnitTests) {
        if (-not (Test-Path $BackendTests)) {
            throw "Backend settings tests not found: $BackendTests. Copy the included backend\app\tests\test_settings_pipeline.py into your project."
        }
        Invoke-CheckedCommand $PythonCommand.Source @('-m','pytest',$BackendTests,'-q') $ProjectRoot 'Backend unit/API validation tests'
    }

    if (-not (Test-Path (Join-Path $DesktopDir 'package.json'))) {
        throw "Desktop package.json not found at $DesktopDir. This script must be extracted to the BrahmaDesk Main project root."
    }

    if (-not (Test-Path (Join-Path $DesktopDir 'node_modules'))) {
        Invoke-CheckedCommand $NpmCommand.Source @('install') $DesktopDir 'Install desktop dependencies'
    }
    Push-Location $DesktopDir
    try {
        & $NodeCommand.Source -e "require.resolve('playwright')" 2>$null
        $PlaywrightMissing = ($LASTEXITCODE -ne 0)
    }
    finally { Pop-Location }
    if ($PlaywrightMissing) {
        Invoke-CheckedCommand $NpmCommand.Source @('install','--save-dev','playwright') $DesktopDir 'Install Playwright for Electron automation'
    }

    if (-not $SkipBuild) {
        Invoke-CheckedCommand $NpmCommand.Source @('run','build') $DesktopDir 'Desktop TypeScript + Vite production build'
    }

    $health = Get-BackendHealth
    if ($health -and ($health.backend -ne 'ok' -or $health.llama_server -ne 'ok')) {
        throw "A backend is already responding at $BackendUrl but is unhealthy: $(ConvertTo-Json $health -Compress). Fix it before running E2E; the test runner will not start a second backend on the same port."
    }
    if (-not $health) {
        if ($BackendUrl -ne 'http://127.0.0.1:8000') { throw "No backend is reachable at custom URL $BackendUrl. Start that backend first; auto-start supports the default http://127.0.0.1:8000 only." }
        Write-Host "Starting backend (it will manage llama.cpp): $BackendUrl" -ForegroundColor Yellow
        $StartedBackend = Start-Process -FilePath $PythonCommand.Source `
            -ArgumentList @('-m','uvicorn','app.main:app','--app-dir','backend','--host','127.0.0.1','--port','8000') `
            -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErr
        $health = Wait-ForBackend
    }
    else {
        Write-Host 'Reusing the already-running healthy backend/model server.' -ForegroundColor Green
    }
    Write-Host "Health OK: backend=$($health.backend); llama_server=$($health.llama_server); model=$($health.model)"

    $env:BRAHMA_API_URL = $BackendUrl
    Invoke-CheckedCommand $NodeCommand.Source @('.\e2e\run-electron-ui-tests.cjs') $DesktopDir 'Electron UI + live model end-to-end tests'

    Write-Host "`nALL BRAHMA DESK E2E STAGES PASSED." -ForegroundColor Green
    Write-Host "Artifacts: $ResultsDir"
}
catch {
    Write-Host "`nE2E FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Artifacts/logs, if created: $ResultsDir"
    exit 1
}
finally {
    Remove-Item Env:\BRAHMA_API_URL -ErrorAction SilentlyContinue
    if ($StartedBackend -and -not $KeepStartedBackend) {
        Write-Host 'Stopping the backend process started by this test run...' -ForegroundColor DarkYellow
        & taskkill.exe /PID $StartedBackend.Id /T /F 2>$null | Out-Null
    }
    elseif ($StartedBackend -and $KeepStartedBackend) {
        Write-Host "Keeping test-started backend running (PID $($StartedBackend.Id)) as requested."
    }
}
