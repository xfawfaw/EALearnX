<#
.SYNOPSIS
    Deploys JsonLite.mqh and EA_LearningX_Gold.mq5 into your Exness MT5 Data Folder
    and compiles EA_LearningX_Gold.ex5 using Exness MetaEditor64.exe.
#>
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MetaEditor  = "C:\Program Files\MetaTrader 5 EXNESS\MetaEditor64.exe"

if (-not (Test-Path $MetaEditor)) {
    throw "MetaEditor64.exe not found at: $MetaEditor"
}

# Locate Exness MT5 Terminal Data Folder via origin.txt
$TerminalRoot = Join-Path $env:APPDATA "MetaQuotes\Terminal"
$ExnessDataDir = $null

Get-ChildItem $TerminalRoot -Directory | ForEach-Object {
    $OriginFile = Join-Path $_.FullName "origin.txt"
    if (Test-Path $OriginFile) {
        $Origin = (Get-Content $OriginFile -Raw).Trim()
        if ($Origin -like "*MetaTrader 5 EXNESS*") {
            $ExnessDataDir = $_.FullName
        }
    }
}

if (-not $ExnessDataDir) {
    throw "Could not find Exness MT5 data folder under $TerminalRoot"
}

$TargetIncludeDir = Join-Path $ExnessDataDir "MQL5\Include"
$TargetExpertsDir = Join-Path $ExnessDataDir "MQL5\Experts"

New-Item -ItemType Directory -Force -Path $TargetIncludeDir | Out-Null
New-Item -ItemType Directory -Force -Path $TargetExpertsDir | Out-Null

$SrcInclude = Join-Path $ProjectRoot "mql5\Include\JsonLite.mqh"
$SrcExpert  = Join-Path $ProjectRoot "mql5\Experts\EA_LearningX_Gold.mq5"
$DstInclude = Join-Path $TargetIncludeDir "JsonLite.mqh"
$DstExpert  = Join-Path $TargetExpertsDir "EA_LearningX_Gold.mq5"

Copy-Item $SrcInclude $DstInclude -Force
Copy-Item $SrcExpert  $DstExpert  -Force

$LogFile = Join-Path $ProjectRoot "mql5\compile.log"
& $MetaEditor "/compile:$DstExpert" "/log:$LogFile" | Out-Null
Start-Sleep -Seconds 2

if (Test-Path $LogFile) {
    Get-Content $LogFile
}

$Ex5Path = Join-Path $TargetExpertsDir "EA_LearningX_Gold.ex5"
if (-not (Test-Path $Ex5Path)) {
    throw "Compilation failed: $Ex5Path was not generated."
}

Write-Host "`nSUCCESS: Compiled EA ready at $Ex5Path" -ForegroundColor Green
