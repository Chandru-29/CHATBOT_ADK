# run_server.ps1 - Production Launch Script (Windows Compatible)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Starting SQL Chatbot Production Server...   " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Load environment variables from .env if present
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match "^\s*[^#\s]+" } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        if ($name -and $value) {
            [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
        }
    }
}

$Workers = 2
$PoolSize = if ($env:MCP_POOL_SIZE) { [int]$env:MCP_POOL_SIZE } else { 5 }
$TotalSessions = $PoolSize * $Workers

Write-Host "  Config Summary:" -ForegroundColor Yellow
Write-Host "    - Workers       : $Workers"
Write-Host "    - MCP Pool Size : $PoolSize sessions/worker ($TotalSessions total)"
Write-Host "    - LLM Engine    : gemini-2.5-flash ($($env:GEMINI_MAX_CONCURRENT) concurrent / $($env:GEMINI_RPM_LIMIT) RPM)"
Write-Host "==============================================" -ForegroundColor Cyan

uvicorn main:app `
    --workers $Workers `
    --http httptools `
    --host 0.0.0.0 `
    --port 8000 `
    --timeout-keep-alive 30