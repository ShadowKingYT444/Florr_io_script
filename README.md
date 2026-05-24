# florr.io Automation

Desktop automation bot for the florr.io grind flow described in `TECH_STACK.md`.

The packaged app opens a small Windows control panel with calibration, Start/Stop, dry-run/live mode, and editable hotkeys.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Build/open the GUI:

```powershell
.\scripts\build_exe.ps1
.\dist\florr-bot\florr-bot.exe
```

In the GUI:

- Click `Calibrate Screen` with florr.io open in your browser.
- Start with `Dry Run` selected and confirm the log says the minimap was detected.
- Switch to `Live` and click `Start`.
- Use `Stop`, `F12`, or the mouse failsafe corner to stop input.

CLI dry run is still available for debugging:

```powershell
python -m florr_bot --config config/default.yaml --dry-run
```

Live run after calibration:

```powershell
python -m florr_bot --config config/default.yaml --calibration config/calibration.yaml --live
```

Emergency stop defaults to `F12`.

Current default game hotkeys:

- `9`: faster/speed travel
- `5`: magnet/loot collection
- `4`: damage/grind mode

Combat holds the left mouse button when grinding, and releases it for magnet collection, popup solving, and forced death wait.

## GitHub Build Artifact

The repository includes `.github/workflows/build-windows.yml`. On GitHub, run the `Build Windows App` workflow and download the `florr-bot-windows` artifact for a ready-to-run Windows app folder.

## Build EXE

```powershell
.\scripts\build_exe.ps1
```

The executable is created under `dist\florr-bot\florr-bot.exe`.

## Calibration

The starting thresholds in `config/default.yaml` are conservative placeholders. Use saved debug frames and the guide video to tune:

- minimap region
- first portal target
- second-map grind target
- brown dimension background color
- mob masks
- drop-density thresholds
- ball-task red/grey masks

Video frame sampling:

```powershell
python scripts/sample_video_frames.py "C:\Users\terry\Downloads\Florr_io_guide.mp4" --start 0 --end 180 --every 5
```
