param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$BrahmaExe = ".\engine\build\Release\brahma-engine.exe",
    [string]$LlamaTokenizer = "D:\all project\llama-b11552-bin-win-cuda-12.4-x64\llama-tokenize.exe"
)

$ErrorActionPreference = "Continue"
foreach ($path in @($ModelPath, $BrahmaExe, $LlamaTokenizer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

$cases = @(
    "Hello, world! How are you?",
    "The quick brown fox jumps over 123 lazy dogs.",
    "C++ templates, std::vector<int>, and x == 42;",
    "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "नमस्ते, आप कैसे हैं?",
    ("Line one." + [Environment]::NewLine + "Line two."),
    "Email: test@example.com; URL: https://example.com/a?b=1",
    "I'm testing contractions: we're, they've, she'll, and don't.",
    "Trailing whitespace test   "
)

$passed = 0
$failed = 0
foreach ($text in $cases) {
    Write-Host ""
    Write-Host "===== $text ====="

    $brahmaOutput = & $BrahmaExe $ModelPath --tokenize $text 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Brahma tokenizer failed for input: $text. Output: $brahmaOutput"
        $failed++
        continue
    }
    if ($brahmaOutput -notmatch 'Tokenizer IDs \(\d+ tokens\):\s*([0-9 ]*)') {
        Write-Error "Could not parse Brahma tokenizer output: $brahmaOutput"
        $failed++
        continue
    }
    $brahmaIds = @()
    if ($Matches[1].Trim().Length -gt 0) {
        $brahmaIds = @($Matches[1].Trim() -split '\s+' | ForEach-Object { [uint32]$_ })
    }

    $llamaOutput = & $LlamaTokenizer -m $ModelPath -p $text --ids --no-bos 2>$null | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Error "llama.cpp tokenizer failed for input: $text. Output: $llamaOutput"
        $failed++
        continue
    }
    $idLine = ([regex]::Split($llamaOutput, '\r?\n') | Where-Object { $_ -match '^\s*\[\s*\d' } | Select-Object -First 1)
    if (-not $idLine) {
        Write-Error "Could not parse llama.cpp token IDs: $llamaOutput"
        $failed++
        continue
    }
    $llamaIds = @([regex]::Matches($idLine, '\d+') | ForEach-Object { [uint32]$_.Value })

    $same = ($brahmaIds.Count -eq $llamaIds.Count)
    if ($same) {
        for ($i = 0; $i -lt $brahmaIds.Count; $i++) {
            if ($brahmaIds[$i] -ne $llamaIds[$i]) { $same = $false; break }
        }
    }

    if ($same) {
        Write-Host "PASS ($($brahmaIds.Count) tokens)" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        Write-Host "  Brahma:   [$($brahmaIds -join ', ')]"
        Write-Host "  llama.cpp:[$($llamaIds -join ', ')]"
        $failed++
    }
}

Write-Host ""
Write-Host "Tokenizer parity: $passed passed, $failed failed, $($cases.Count) total."
if ($failed -gt 0) { exit 1 }
exit 0
