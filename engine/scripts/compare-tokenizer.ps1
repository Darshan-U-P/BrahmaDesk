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

function From-CodePoints([int[]]$CodePoints) {
    return -join ($CodePoints | ForEach-Object { [char]$_ })
}

$cases = @(
    "Hello, world! How are you?",
    "The quick brown fox jumps over 123 lazy dogs.",
    "C++ templates, std::vector<int>, and x == 42;",
    (From-CodePoints @(0x0CA8,0x0CAE,0x0CB8,0x0CCD,0x0C95,0x0CBE,0x0CB0,0x002C,0x0020,0x0CA8,0x0CC0,0x0CB5,0x0CC1,0x0020,0x0CB9,0x0CC7,0x0C97,0x0CBF,0x0CA6,0x0CCD,0x0CA6,0x0CC0,0x0CB0,0x0CBF,0x003F)),
    (From-CodePoints @(0x0928,0x092E,0x0938,0x094D,0x0924,0x0947,0x002C,0x0020,0x0906,0x092A,0x0020,0x0915,0x0948,0x0938,0x0947,0x0020,0x0939,0x0948,0x0902,0x003F)),
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

    # Pass UTF-8 bytes as ASCII hex to avoid Windows PowerShell native-argument code-page conversion.
    $utf8Hex = [BitConverter]::ToString([System.Text.Encoding]::UTF8.GetBytes($text)).Replace("-", "")
    $brahmaOutput = & $BrahmaExe $ModelPath --tokenize-utf8-hex $utf8Hex 2>&1 | Out-String
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
