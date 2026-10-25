param(
    [string]$RuntimeDir = $env:LLAMA_CPP_RUNTIME_DIR
)
$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$StageDir = Join-Path $DesktopRoot "build/llama-runtime"

if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = "D:\all project\llama-b11552-bin-win-cuda-12.4-x64"
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$Server = Join-Path $RuntimeDir "llama-server.exe"

if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "llama-server.exe was not found at '$Server'. Set LLAMA_CPP_RUNTIME_DIR to the folder containing the CUDA-enabled llama-server.exe and its DLLs."
}
$Version = & $Server --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "The selected llama-server.exe failed its --version check: $Version"
}

if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
Copy-Item -Path (Join-Path $RuntimeDir "*") -Destination $StageDir -Recurse -Force
if (-not (Test-Path -LiteralPath (Join-Path $StageDir "llama-server.exe"))) {
    throw "Runtime staging did not place llama-server.exe at the expected root."
}
Write-Host "Staged llama.cpp runtime from $RuntimeDir" -ForegroundColor Green
Write-Host "Runtime version: $Version"
