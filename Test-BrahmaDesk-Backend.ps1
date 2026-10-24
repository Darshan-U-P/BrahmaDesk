[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$IncludeChat,
    [switch]$IncludeMemoryTests
)

$ErrorActionPreference = "Stop"
$script:Results = [System.Collections.Generic.List[object]]::new()
$script:Passed = 0
$script:Failed = 0
$script:Skipped = 0

function Add-Result {
    param([string]$Name, [string]$Status, [string]$Detail = "")
    $script:Results.Add([pscustomobject]@{
        Test = $Name
        Status = $Status
        Detail = $Detail
    })
    switch ($Status) {
        "PASS" { $script:Passed++ ; Write-Host "[PASS] $Name $Detail" -ForegroundColor Green }
        "FAIL" { $script:Failed++ ; Write-Host "[FAIL] $Name $Detail" -ForegroundColor Red }
        default { $script:Skipped++ ; Write-Host "[SKIP] $Name $Detail" -ForegroundColor Yellow }
    }
}

function Invoke-Api {
    param(
        [Parameter(Mandatory)][string]$Method,
        [Parameter(Mandatory)][string]$Path,
        [object]$Body = $null,
        [int[]]$ExpectedStatus = @(200),
        [switch]$Raw
    )
    $uri = "$($BaseUrl.TrimEnd('/'))$Path"
    $args = @{
        Uri = $uri
        Method = $Method
        TimeoutSec = 90
        UseBasicParsing = $true
        ErrorAction = "Stop"
    }
    if ($null -ne $Body) {
        $args.ContentType = "application/json"
        $args.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
    }

    try {
        $response = Invoke-WebRequest @args
        $status = [int]$response.StatusCode
        if ($ExpectedStatus -notcontains $status) {
            throw "Unexpected HTTP $status; expected $($ExpectedStatus -join ',')"
        }
        if ($Raw) { return $response.Content }
        if ([string]::IsNullOrWhiteSpace($response.Content)) { return $null }
        try { return ($response.Content | ConvertFrom-Json) }
        catch { return $response.Content }
    }
    catch {
        $statusCode = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
        }
        if ($null -ne $statusCode -and $ExpectedStatus -contains $statusCode) {
            return $null
        }
        throw "$Method $Path failed: $($_.Exception.Message)"
    }
}

Write-Host "`nBrahma Desk Backend API Test" -ForegroundColor Cyan
Write-Host "Base URL: $BaseUrl`n"

# Connectivity + OpenAPI
try {
    $openapi = Invoke-Api -Method GET -Path "/openapi.json"
    Add-Result "OpenAPI document" "PASS" "title=$($openapi.info.title), version=$($openapi.info.version)"
} catch {
    Add-Result "OpenAPI document" "FAIL" $_.Exception.Message
    Write-Host "`nCannot continue without a reachable API. Is Uvicorn running?" -ForegroundColor Red
    exit 1
}

# Automatically smoke-test safe GET routes with no path parameters.
$paths = @($openapi.paths.PSObject.Properties)
$getRoutes = foreach ($p in $paths) {
    if ($p.Value.get -and $p.Name -notmatch '\{') { $p.Name }
}
foreach ($route in ($getRoutes | Sort-Object -Unique)) {
    if ($route -eq "/openapi.json") { continue }
    try {
        $null = Invoke-Api -Method GET -Path $route
        Add-Result "GET $route" "PASS"
    } catch {
        Add-Result "GET $route" "FAIL" $_.Exception.Message
    }
}

# Health endpoint, if present.
if ($openapi.paths.PSObject.Properties.Name -contains "/api/health") {
    try {
        $health = Invoke-Api -Method GET -Path "/api/health"
        Add-Result "Health endpoint" "PASS" (($health | ConvertTo-Json -Compress -Depth 5))
    } catch {
        Add-Result "Health endpoint" "FAIL" $_.Exception.Message
    }
}

# Conversation CRUD lifecycle. Creates a clearly labelled test conversation and deletes it at the end.
$conversationId = $null
if ($openapi.paths.PSObject.Properties.Name -contains "/api/conversations") {
    try {
        $created = Invoke-Api -Method POST -Path "/api/conversations" -Body @{
            title = "API TEST - temporary - $(Get-Date -Format 'yyyyMMdd-HHmmss')"
        }
        $conversationId = $created.id
        if (-not $conversationId) { throw "Create response did not include an id." }
        Add-Result "Create conversation" "PASS" "id=$conversationId"

        $list = Invoke-Api -Method GET -Path "/api/conversations"
        $items = @($list.items)
        if ($items.id -contains $conversationId) {
            Add-Result "Conversation appears in list" "PASS"
        } else {
            Add-Result "Conversation appears in list" "FAIL" "Created id not found in items."
        }

        $detail = Invoke-Api -Method GET -Path "/api/conversations/$conversationId"
        if ($detail.id -eq $conversationId) {
            Add-Result "Get conversation by id" "PASS"
        } else {
            Add-Result "Get conversation by id" "FAIL" "Response id mismatch."
        }

        if ($openapi.paths.PSObject.Properties.Name -contains "/api/conversations/{conversation_id}") {
            $renamedTitle = "API TEST renamed $(Get-Date -Format 'HHmmss')"
            $renamed = Invoke-Api -Method PATCH -Path "/api/conversations/$conversationId" -Body @{ title = $renamedTitle }
            if ($renamed.title -eq $renamedTitle) {
                Add-Result "Rename conversation" "PASS"
            } else {
                Add-Result "Rename conversation" "FAIL" "Rename response did not contain expected title."
            }
        }
    } catch {
        Add-Result "Conversation CRUD lifecycle" "FAIL" $_.Exception.Message
    } finally {
        if ($conversationId -and ($openapi.paths.PSObject.Properties.Name -contains "/api/conversations/{conversation_id}")) {
            try {
                $null = Invoke-Api -Method DELETE -Path "/api/conversations/$conversationId"
                Add-Result "Delete temporary conversation" "PASS"
            } catch {
                Add-Result "Delete temporary conversation" "FAIL" $_.Exception.Message
            }
        }
    }
} else {
    Add-Result "Conversation CRUD lifecycle" "SKIP" "Conversation endpoints not present in this backend."
}

# Chat stream is optional because it invokes the local model and may take time.
if ($IncludeChat) {
    if ($openapi.paths.PSObject.Properties.Name -contains "/api/chat/stream") {
        try {
            $chatBody = @{
                messages = @(@{ role = "user"; content = "Reply with exactly: BRAHMA_API_TEST_OK" })
                temperature = 0.1
                max_tokens = 32
            }
            $stream = Invoke-Api -Method POST -Path "/api/chat/stream" -Body $chatBody -Raw
            if ($stream -match 'data: \[DONE\]' -and $stream -match 'delta') {
                Add-Result "Chat streaming" "PASS" "Received stream deltas and [DONE]."
            } elseif ($stream -match '"error"') {
                Add-Result "Chat streaming" "FAIL" "Stream returned an error event: $stream"
            } else {
                Add-Result "Chat streaming" "FAIL" "Unexpected stream format."
            }
        } catch {
            Add-Result "Chat streaming" "FAIL" $_.Exception.Message
        }
    } else {
        Add-Result "Chat streaming" "SKIP" "Chat stream route not present."
    }
} else {
    Add-Result "Chat streaming" "SKIP" "Use -IncludeChat to send a prompt to the local model."
}

# Memory endpoints vary by project revision; only test if advertised in OpenAPI.
if ($IncludeMemoryTests) {
    $memoryCollection = "/api/memories"
    if ($openapi.paths.PSObject.Properties.Name -contains $memoryCollection) {
        try {
            $memories = Invoke-Api -Method GET -Path $memoryCollection
            Add-Result "List memories" "PASS" "Returned $(@($memories.items).Count) item(s)."
        } catch {
            Add-Result "List memories" "FAIL" $_.Exception.Message
        }
        Add-Result "Memory write/delete" "SKIP" "Not run automatically to avoid modifying persistent user memory. Test with a dedicated disposable profile/database."
    } else {
        Add-Result "Memory API" "SKIP" "Memory routes are not present in the OpenAPI document."
    }
}

Write-Host "`n---------------- SUMMARY ----------------" -ForegroundColor Cyan
$script:Results | Format-Table -AutoSize -Wrap
Write-Host ("Passed: {0} | Failed: {1} | Skipped: {2}" -f $script:Passed, $script:Failed, $script:Skipped)
if ($script:Failed -gt 0) { exit 1 } else { exit 0 }
