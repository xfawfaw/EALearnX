<#
.SYNOPSIS
    Starts the EA-LearningX Python service silently in the background (logs to logs/bot.log).
#>
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe   = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StopScript  = Join-Path $PSScriptRoot "stop_bot.ps1"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment python.exe not found at $PythonExe. Run .\scripts\run_bot.ps1 once first."
}

if (Test-Path $StopScript) {
    & $StopScript
}

Start-Process -FilePath $PythonExe -ArgumentList "-m", "python.src.main" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
Write-Host "Started EA-LearningX silently in background (logs: $ProjectRoot\logs\bot.log)." -ForegroundColor Green
