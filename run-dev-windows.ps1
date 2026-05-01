$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendScript = Join-Path $RootDir "scripts\start-backend-windows.ps1"
$FrontendScript = Join-Path $RootDir "scripts\start-frontend-windows.ps1"

if (-not (Test-Path -LiteralPath $BackendScript)) {
    throw "Backend script not found: $BackendScript"
}

if (-not (Test-Path -LiteralPath $FrontendScript)) {
    throw "Frontend script not found: $FrontendScript"
}

Start-Process powershell.exe `
    -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$BackendScript`"" `
    -WindowStyle Normal

Start-Process powershell.exe `
    -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$FrontendScript`"" `
    -WindowStyle Normal

$BackendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { "8000" }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "3000" }

Write-Host "Opened Data Pilot in two PowerShell windows:"
Write-Host "  Backend:  http://127.0.0.1:$BackendPort"
Write-Host "  Frontend: http://127.0.0.1:$FrontendPort"
