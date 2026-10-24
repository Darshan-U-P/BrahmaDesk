
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$PyInstaller = Join-Path $PSScriptRoot ".venv\Scripts\pyinstaller.exe"

if (-not (Test-Path $Python)) {
    py -3.12 -m venv .venv
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $Python -m pip install -r .\requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install backend dependencies."
}

@'
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller extracts the application into its runtime directory.
    app_dir = Path(sys._MEIPASS)
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

import uvicorn

host = os.getenv("BRAHMA_BACKEND_HOST", "127.0.0.1")
port = int(os.getenv("BRAHMA_BACKEND_PORT", "8000"))

uvicorn.run(
    "app.main:app",
    host=host,
    port=port,
    log_level="info",
)
'@ | Set-Content ".\run_backend.py" -Encoding utf8

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name brahma-backend `
    --paths $PSScriptRoot `
    --collect-submodules app `
    --collect-all uvicorn `
    --collect-all fastapi `
    --collect-all starlette `
    --collect-all pydantic `
    --collect-all httpx `
    --hidden-import app.main `
    ".\run_backend.py"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$Exe = Join-Path $PSScriptRoot "dist\brahma-backend.exe"
if (-not (Test-Path $Exe)) {
    throw "Backend executable was not created."
}

Write-Host "Backend built: $Exe" -ForegroundColor Green
