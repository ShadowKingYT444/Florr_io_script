param(
    [switch]$OneFile,
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

$modeArgs = @("--onedir")
if ($OneFile) {
    $modeArgs = @("--onefile")
}

$windowArgs = @("--windowed")
if ($Console) {
    $windowArgs = @("--console")
}

.\.venv\Scripts\python.exe -m PyInstaller `
    --name florr-bot `
    @modeArgs `
    @windowArgs `
    --clean `
    --noconfirm `
    --paths src `
    --add-data "config;config" `
    --add-data "assets\references;assets\references" `
    "src\florr_bot\gui.py"
