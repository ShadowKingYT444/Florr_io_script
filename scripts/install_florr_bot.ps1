param(
    [switch]$DesktopShortcut = $true,
    [switch]$StartMenuShortcut = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\FlorrBot"
$InstalledExe = Join-Path $InstallDir "FlorrBot.exe"

$ReleaseOneFile = Join-Path $ScriptDir "FlorrBot.exe"
$OneFile = Join-Path $RepoRoot "dist\florr-bot.exe"
$OneDir = Join-Path $RepoRoot "dist\florr-bot"
$OneDirExe = Join-Path $OneDir "florr-bot.exe"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if (Test-Path $ReleaseOneFile) {
    Copy-Item -LiteralPath $ReleaseOneFile -Destination $InstalledExe -Force
} elseif (Test-Path $OneFile) {
    Copy-Item -LiteralPath $OneFile -Destination $InstalledExe -Force
} elseif (Test-Path $OneDirExe) {
    if (Test-Path $InstallDir) {
        Get-ChildItem -LiteralPath $InstallDir -Force | Remove-Item -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $OneDir "*") -Destination $InstallDir -Recurse -Force
    Rename-Item -LiteralPath (Join-Path $InstallDir "florr-bot.exe") -NewName "FlorrBot.exe" -Force
} else {
    throw "No built app found. Run scripts\build_release.ps1 first."
}

function New-Shortcut {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Target
    )
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($Path)
    $Shortcut.TargetPath = $Target
    $Shortcut.WorkingDirectory = Split-Path -Parent $Target
    $Shortcut.Description = "Florr.io Bot"
    $Shortcut.Save()
}

if ($DesktopShortcut) {
    New-Shortcut -Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "Florr Bot.lnk") -Target $InstalledExe
}

if ($StartMenuShortcut) {
    $Programs = [Environment]::GetFolderPath("Programs")
    $ShortcutDir = Join-Path $Programs "FlorrBot"
    New-Item -ItemType Directory -Force -Path $ShortcutDir | Out-Null
    New-Shortcut -Path (Join-Path $ShortcutDir "Florr Bot.lnk") -Target $InstalledExe
}

Write-Host "Installed Florr Bot to: $InstalledExe"
Write-Host "You can launch it from the Start Menu, Desktop shortcut, or by running the EXE above."
