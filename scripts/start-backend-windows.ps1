$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $RootDir "excel-agent-backend"
$Port = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { "8000" }
$HostName = if ($env:BACKEND_HOST) { $env:BACKEND_HOST } else { "127.0.0.1" }

Set-Location -LiteralPath $BackendDir

if (Test-Path -LiteralPath ".venv") {
    $VenvDir = ".venv"
} elseif (Test-Path -LiteralPath "venv") {
    $VenvDir = "venv"
} else {
    $VenvDir = ".venv"
    Write-Host "Creating backend virtual environment at $BackendDir\$VenvDir"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv $VenvDir
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv $VenvDir
    } else {
        python -m venv $VenvDir
    }
}

& ".\$VenvDir\Scripts\Activate.ps1"

python -c "import fastapi, uvicorn" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing backend dependencies into $VenvDir"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv pip install -r requirements.txt
    } else {
        python -m pip install -r requirements.txt
    }
}

Write-Host "Backend venv: $BackendDir\$VenvDir"
Write-Host "Backend URL:  http://$HostName`:$Port"
python -m uvicorn main:app --reload --host $HostName --port $Port
