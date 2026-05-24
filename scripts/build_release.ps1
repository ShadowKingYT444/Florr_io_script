$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "release" | Out-Null

Write-Host "Building one-file GUI executable..."
.\scripts\build_exe.ps1 -OneFile

$OneFile = "dist\florr-bot.exe"
if (-not (Test-Path $OneFile)) {
    throw "One-file build did not produce $OneFile"
}

Copy-Item -LiteralPath $OneFile -Destination "release\FlorrBot.exe" -Force

Write-Host "Building portable app folder..."
.\scripts\build_exe.ps1

$PortableZip = "release\FlorrBot-Portable.zip"
if (Test-Path $PortableZip) {
    Remove-Item -LiteralPath $PortableZip -Force
}
Compress-Archive -Path "dist\florr-bot\*" -DestinationPath $PortableZip -Force

Copy-Item -LiteralPath "scripts\install_florr_bot.ps1" -Destination "release\install_florr_bot.ps1" -Force
Copy-Item -LiteralPath "scripts\install_florr_bot.bat" -Destination "release\install_florr_bot.bat" -Force

Write-Host "Release outputs:"
Write-Host "  release\FlorrBot.exe"
Write-Host "  release\FlorrBot-Portable.zip"
Write-Host "  release\install_florr_bot.ps1"
Write-Host "  release\install_florr_bot.bat"
