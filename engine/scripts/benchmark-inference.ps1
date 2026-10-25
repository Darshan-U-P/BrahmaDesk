param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$BrahmaExe = ".\engine\build\Release\brahma-engine.exe",
    [string]$Prompt = "Explain photosynthesis in one simple sentence.",
    [ValidateRange(1, 20)]
    [int]$Runs = 5,
    [ValidateRange(1, 4096)]
    [int]$MaxNewTokens = 8,
    [string]$OutputCsv = ".\brahma-benchmark.csv"
)

$ErrorActionPreference = "Stop"
foreach ($path in @($ModelPath, $BrahmaExe)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

$exePath = (Resolve-Path -LiteralPath $BrahmaExe).Path
$modelFullPath = (Resolve-Path -LiteralPath $ModelPath).Path
$outputFullPath = [System.IO.Path]::GetFullPath($OutputCsv)
$outputDirectory = Split-Path -Parent $outputFullPath
if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

function Get-Metric {
    param([string]$Text, [string]$Pattern, [string]$MetricName)
    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) {
        throw ("Could not parse '" + $MetricName + "' from Brahma diagnostics. Output was: " + $Text)
    }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$records = @()
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("brahma-benchmark-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot | Out-Null

try {
    Write-Host "Brahma Engine repeatable benchmark"
    Write-Host "Model: $modelFullPath"
    Write-Host "Runs: $Runs; generated-token cap: $MaxNewTokens"
    Write-Host "Prompt: $Prompt"
    Write-Host "First run may be affected by cold file cache and CPU boost state."
    Write-Host ""

    for ($run = 1; $run -le $Runs; $run++) {
        $stdoutPath = Join-Path $tempRoot "run-$run.stdout.txt"
        $stderrPath = Join-Path $tempRoot "run-$run.stderr.txt"
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        & $exePath $modelFullPath --generate $Prompt $MaxNewTokens 1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $diagnostics = Get-Content -LiteralPath $stderrPath -Raw
        if ($exitCode -ne 0) {
            throw "Brahma run $run failed with exit code $exitCode. $diagnostics"
        }

        $construction = Get-Metric $diagnostics 'Model construction:\s*([0-9.eE+-]+)\s*s' "model construction"
        $prefill = Get-Metric $diagnostics 'Prompt prefill:\s*([0-9.eE+-]+)\s*s' "prompt prefill"
        $decode = Get-Metric $diagnostics 'Decode loop:\s*([0-9.eE+-]+)\s*s' "decode loop"
        $projection = Get-Metric $diagnostics 'Projections:\s*([0-9.eE+-]+)\s*s' "projection time"
        $normalization = Get-Metric $diagnostics 'RMSNorm:\s*([0-9.eE+-]+)\s*s' "RMSNorm time"
        $attention = Get-Metric $diagnostics 'Attention core:\s*([0-9.eE+-]+)\s*s' "attention time"
        $promptTokens = [int](Get-Metric $diagnostics 'Prompt tokens:\s*(\d+)' "prompt token count")
        $generatedTokens = [int](Get-Metric $diagnostics 'Generated tokens:\s*(\d+)' "generated token count")
        $backendMatch = [regex]::Match($diagnostics, 'Parallel backend:\s*(\S+)')
        $backend = if ($backendMatch.Success) { $backendMatch.Groups[1].Value } else { "unknown" }
        $prefillRate = if ($prefill -gt 0) { $promptTokens / $prefill } else { 0.0 }
        $decodeRate = if ($decode -gt 0) { $generatedTokens / $decode } else { 0.0 }

        $record = [PSCustomObject]@{
            Run = $run
            Backend = $backend
            PromptTokens = $promptTokens
            GeneratedTokens = $generatedTokens
            WallSeconds = [math]::Round($watch.Elapsed.TotalSeconds, 6)
            ConstructionSeconds = $construction
            PrefillSeconds = $prefill
            PrefillTokensPerSecond = [math]::Round($prefillRate, 6)
            DecodeSeconds = $decode
            DecodeTokensPerSecond = [math]::Round($decodeRate, 6)
            ProjectionSeconds = $projection
            RMSNormSeconds = $normalization
            AttentionCoreSeconds = $attention
        }
        $records += $record
        Write-Host ("Run {0}/{1}: backend={2}, wall={3:N3}s, prefill={4:N3} tok/s, decode={5:N3} tok/s" -f $run, $Runs, $backend, $watch.Elapsed.TotalSeconds, $prefillRate, $decodeRate)
    }

    $records | Export-Csv -LiteralPath $outputFullPath -NoTypeInformation -Encoding UTF8
    Write-Host ""
    Write-Host "Per-run results saved to: $outputFullPath"
    Write-Host "Median summary (medians calculated across all runs):"
    function Get-Median {
        param([double[]]$Values)
        $sorted = @($Values | Sort-Object)
        $middle = [int][math]::Floor($sorted.Count / 2)
        if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$middle] }
        return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
    }
    $summary = [PSCustomObject]@{
        Runs = $Runs
        Backend = (($records | Select-Object -ExpandProperty Backend -Unique) -join ",")
        MedianWallSeconds = [math]::Round((Get-Median -Values @($records | ForEach-Object { [double]$_.WallSeconds })), 6)
        MedianConstructionSeconds = [math]::Round((Get-Median -Values @($records | ForEach-Object { [double]$_.ConstructionSeconds })), 6)
        MedianPrefillTokensPerSecond = [math]::Round((Get-Median -Values @($records | ForEach-Object { [double]$_.PrefillTokensPerSecond })), 6)
        MedianDecodeTokensPerSecond = [math]::Round((Get-Median -Values @($records | ForEach-Object { [double]$_.DecodeTokensPerSecond })), 6)
        MedianProjectionSeconds = [math]::Round((Get-Median -Values @($records | ForEach-Object { [double]$_.ProjectionSeconds })), 6)
    }
    $summary | Format-List
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
