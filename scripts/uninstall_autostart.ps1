<#
.SYNOPSIS
    Removes the EA-LearningX and MetaTrader 5 EXNESS Windows Startup shortcuts.
#>
$ErrorActionPreference = "SilentlyContinue"

$StartupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$Links = @(
    (Join-Path $StartupDir "EA-LearningX-MT5.lnk"),
    (Join-Path $StartupDir "EA-LearningX-Bot.lnk")
)

foreach ($lnk in $Links) {
    if (Test-Path $lnk) {
        Remove-Item $lnk -Force
        Write-Host "Removed $lnk" -ForegroundColor Green
    } else {
        Write-Host "Not found: $lnk" -ForegroundColor Yellow
    }
}
