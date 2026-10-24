$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    py -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller

@'
import os
import uvicorn

host = os.getenv("BRAHMA_BACKEND_HOST", "127.0.0.1")
port = int(os.getenv("BRAHMA_BACKEND_PORT", "8000"))
uvicorn.run("app.main:app", host=host, port=port, log_level="info")
'@ | Set-Content ".\run_backend.py" -Encoding utf8

& ".\.venv\Scripts\pyinstaller.exe" `
    --noconfirm `
    --clean `
    --onefile `
    --name brahma-backend `
    --collect-all uvicorn `
    --collect-all fastapi `
    --collect-all pydantic `
    --collect-all httpx `
    --hidden-import app.main `
    --paths . `
    ".\run_backend.py"

if (-not (Test-Path ".\dist\brahma-backend.exe")) {
    throw "Backend executable was not created."
}
Write-Host "Backend built: $PSScriptRoot\dist\brahma-backend.exe"
