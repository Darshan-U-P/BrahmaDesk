$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$SourceModel = "D:\all project\Projects\AI projects\BrahmaDesk\BrahmaDesk-Phase0\Models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$StageDir = Join-Path $DesktopRoot "build/bundled-model"
$TargetModel = Join-Path $StageDir "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

if (-not (Test-Path -LiteralPath $SourceModel -PathType Leaf)) {
    throw "Bundled GGUF model not found: '$SourceModel'. Update desktop/scripts/stage-bundled-model.ps1 to the actual model location."
}
$header = [System.IO.File]::OpenRead($SourceModel)
try {
    $magic = New-Object byte[] 4
    $read = $header.Read($magic, 0, 4)
} finally {
    $header.Dispose()
}
if ($read -ne 4 -or [System.Text.Encoding]::ASCII.GetString($magic) -ne "GGUF") {
    throw "The selected model does not have a valid GGUF header."
}
if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
Copy-Item -LiteralPath $SourceModel -Destination $TargetModel
$hash = Get-FileHash -LiteralPath $TargetModel -Algorithm SHA256
@{
    filename = [System.IO.Path]::GetFileName($TargetModel)
    size_bytes = (Get-Item -LiteralPath $TargetModel).Length
    sha256 = $hash.Hash.ToLowerInvariant()
    source = $SourceModel
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StageDir "manifest.json") -Encoding utf8
Write-Host "Staged bundled model: $TargetModel"
Write-Host ("Model size: {0:N2} GB" -f ((Get-Item -LiteralPath $TargetModel).Length / 1GB))
Write-Host "SHA-256: $($hash.Hash)"
