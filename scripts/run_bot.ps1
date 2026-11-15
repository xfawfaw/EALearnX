<#
.SYNOPSIS
    Creates/activates .venv, installs requirements, and starts the EA-LearningX Python AI & Telegram service.
#>
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Fill in GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, and TELEGRAM_ALLOWED_USER_IDS first." -ForegroundColor Yellow
    exit 1
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment (.venv)..."
    python -m venv .venv
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
}

Write-Host "Starting EA-LearningX AI & Telegram service..." -ForegroundColor Green
& $VenvPython -m python.src.main
