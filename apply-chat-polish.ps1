param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot
)
$ErrorActionPreference = 'Stop'
$TargetRoot = (Resolve-Path -LiteralPath $TargetRoot).Path
$PatchRoot = $PSScriptRoot
$TargetApp = Join-Path $TargetRoot 'desktop\src\App.tsx'
$TargetCss = Join-Path $TargetRoot 'desktop\src\styles.css'
$PatchApp = Join-Path $PatchRoot 'desktop\src\App.tsx'
$PatchCss = Join-Path $PatchRoot 'desktop\src\styles.css'
foreach ($path in @($TargetApp, $TargetCss, $PatchApp, $PatchCss)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $TargetRoot "rollback-backups\chat-polish-$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item -LiteralPath $TargetApp -Destination (Join-Path $backup 'App.tsx')
Copy-Item -LiteralPath $TargetCss -Destination (Join-Path $backup 'styles.css')
Copy-Item -LiteralPath $PatchApp -Destination $TargetApp -Force
Copy-Item -LiteralPath $PatchCss -Destination $TargetCss -Force
Write-Host 'Chat polish patch applied.' -ForegroundColor Green
Write-Host "Original files backed up to: $backup"
Write-Host 'No backend files or database files were changed.' -ForegroundColor Cyan
