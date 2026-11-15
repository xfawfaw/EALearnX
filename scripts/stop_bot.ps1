<#
.SYNOPSIS
    Stops any running EA-LearningX Python service on this Windows PC.
#>
$ErrorActionPreference = "SilentlyContinue"

$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and $_.CommandLine -match "python\.src\.main"
}

if (-not $procs) {
    Write-Host "No running EA-LearningX Python service found." -ForegroundColor Yellow
    exit 0
}

foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force
    Write-Host "Stopped EA-LearningX Python service (PID $($p.ProcessId))." -ForegroundColor Green
}
