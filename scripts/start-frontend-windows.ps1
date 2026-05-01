$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $RootDir "excel-agent-dashboard"
$Port = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "3000" }

Set-Location -LiteralPath $FrontendDir

if (-not (Test-Path -LiteralPath "node_modules")) {
    Write-Host "Installing frontend dependencies"
    if (Get-Command pnpm -ErrorAction SilentlyContinue) {
        pnpm install
    } elseif (Test-Path -LiteralPath "package-lock.json") {
        npm ci
    } else {
        npm install
    }
}

Write-Host "Frontend directory: $FrontendDir"
Write-Host "Frontend URL:       http://127.0.0.1:$Port"

if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    pnpm dev -- -p $Port
} else {
    npm run dev -- -p $Port
}
