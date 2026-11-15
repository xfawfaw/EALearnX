<#
.SYNOPSIS
    Installs Windows Startup shortcuts so MetaTrader 5 EXNESS and EA-LearningX
    auto-start silently on login in Idle Standby (PAUSE) mode.
#>
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StartupDir  = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$Mt5Exe      = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
$StartScript = Join-Path $ProjectRoot "scripts\start_background.ps1"

if (-not (Test-Path $Mt5Exe)) {
    throw "MetaTrader 5 EXNESS not found at $Mt5Exe"
}
if (-not (Test-Path $StartScript)) {
    throw "start_background.ps1 not found at $StartScript"
}

$WshShell = New-Object -ComObject WScript.Shell

$Mt5LinkPath = Join-Path $StartupDir "EA-LearningX-MT5.lnk"
$Mt5Shortcut = $WshShell.CreateShortcut($Mt5LinkPath)
$Mt5Shortcut.TargetPath       = $Mt5Exe
$Mt5Shortcut.WorkingDirectory = Split-Path -Parent $Mt5Exe
$Mt5Shortcut.Description      = "Auto-start MetaTrader 5 EXNESS for EA-LearningX"
$Mt5Shortcut.Save()

$BotLinkPath = Join-Path $StartupDir "EA-LearningX-Bot.lnk"
$BotShortcut = $WshShell.CreateShortcut($BotLinkPath)
$BotShortcut.TargetPath       = "powershell.exe"
$BotShortcut.Arguments        = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`""
$BotShortcut.WorkingDirectory = $ProjectRoot
$BotShortcut.WindowStyle      = 7
$BotShortcut.Description      = "Auto-start EA-LearningX Telegram & AI Standby Listener"
$BotShortcut.Save()

Write-Host "Installed Windows Startup shortcuts:" -ForegroundColor Green
Write-Host "  1. $Mt5LinkPath"
Write-Host "  2. $BotLinkPath"
