# florr.io Automation Tech Stack

## Goal

Build a Windows desktop automation bot for florr.io that can:

- Activate faster/speed travel with `9`.
- Navigate from the first map to the portal shown in the supplied minimap references.
- Verify the dimension transition by detecting the brown background.
- Navigate the second map toward the top-right grinding zone.
- Fight mobs by holding attack in damage mode `4`, kiting, and switching to magnet mode `5` for loot.
- Detect large clusters of drops and collect them.
- Detect the popup/ball task, drag the ball through the endzone path, then resume grinding.
- Intentionally die every 3 hours and wait 30 minutes before resuming.

This stack assumes visible desktop automation rather than game memory reads, packet inspection, browser injection, or anti-detection work.

## Chosen Language

Python 3.12 on Windows.

Python is the best fit because the bot needs real-time screen capture, computer vision, pathfinding, input simulation, data logging, and fast calibration scripts. Its ecosystem covers all of that without needing a compiled app early on.

## Runtime Layout

Use a local virtual environment inside this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Recommended package manager for the first version: plain `pip` plus `requirements.txt`.

Reason: this project will need quick iteration while tuning image thresholds from screenshots and recordings. A simple requirements file keeps setup friction low.

## Core Dependencies

```txt
mss==10.1.0
opencv-python-headless==4.12.0.88
numpy==2.4.4
pyautogui==0.9.54
pydirectinput==1.0.4
pynput==1.8.1
pygetwindow==0.0.9
pydantic==2.12.5
pydantic-settings==2.12.0
PyYAML==6.0.3
rich==14.2.0
pytest==9.0.2
```

Optional later dependency if template/color detection is not reliable enough:

```txt
ultralytics==8.3.0
onnxruntime==1.23.2
```

The MVP should start without YOLO. Use deterministic OpenCV masks, contours, and template matching first. Add a trained detector only if mobs, drops, or popup states cannot be detected reliably with classical vision.

## Screen Capture

Primary library: `mss`.

Why:

- Fast enough for real-time loops.
- Can capture specific monitor regions.
- Lower overhead than PyAutoGUI screenshots.
- Works well with OpenCV because captured frames can be converted directly into NumPy arrays.

Capture strategy:

- Full game frame at low frequency for global state checks.
- Minimap crop at higher frequency during navigation.
- Center/world crop at high frequency during combat.
- Popup/task crop whenever the UI state detector sees a modal-like overlay.

Target rates:

- Navigation/minimap loop: 10 to 20 FPS.
- Combat loop: 20 to 30 FPS.
- Logging/debug snapshots: 1 to 2 FPS or event-triggered.

## Computer Vision

Primary library: OpenCV.

Supporting library: NumPy.

### Map and Portal Navigation

Inputs:

- Supplied first-map minimap screenshot.
- Supplied portal crop.
- Supplied second-map minimap screenshot.
- Supplied second-map target crop.

Approach:

1. Crop minimap region from the live game frame.
2. Normalize resolution against stored reference dimensions.
3. Create color masks:
   - White walkable paths.
   - Dark walls/background.
   - Player marker.
   - Portal/target marker.
4. Clean masks with morphological open/close operations.
5. Simplify broad walkable paths when needed.
6. Convert walkable pixels into a grid graph.
7. Run A* from current player marker to target marker.
8. Convert the route into movement vectors.

Pathfinding implementation:

- Use a small custom A* implementation with Python `heapq`.
- Avoid a heavy graph dependency for the core loop.
- Recalculate every few frames so the bot can correct drift.

Movement execution:

- Convert the next path waypoint into a direction vector.
- Use keyboard movement or mouse-relative movement depending on how florr.io responds most reliably.
- Keep the player near the route while using short movement pulses instead of long blind holds.

### Dimension Transition Verification

Use a brown-background detector:

1. Sample several world-background regions away from UI overlays.
2. Convert to HSV.
3. Check that the average hue/saturation/value falls inside a calibrated brown range.
4. Require the condition to hold for several consecutive frames before switching state.

This prevents a one-frame flash or UI element from falsely triggering the dimension state.

### Mob Detection and Combat

MVP approach:

- Use OpenCV color masks and contour detection for mobs if they have stable colors.
- Add template matching for common mob shapes.
- Track detected mobs over time with simple nearest-neighbor matching.

Fallback approach:

- Train a tiny YOLO model only if classical detection fails.
- Export to ONNX and run with `onnxruntime` for local inference.

Combat model:

- Player position is assumed to be screen center.
- For every detected mob, compute distance and bearing from center.
- If a mob is inside threat radius, move away while clicking continuously.
- If a mob is inside attack radius but outside threat radius, keep cursor/attack pressure on it.
- If mobs are sparse, prioritize nearest visible mob.
- If mobs are dense, kite away from the center of the mob cluster.

Clicking:

- Use controlled click/hold cadence, not maximum-speed spam.
- Track attack state so the bot can interrupt for movement, magnet collection, or popup handling.

### Drop Detection and Magnet Switching

Detect drops using blob/contour density in the world crop:

1. Mask likely drop colors and small bright objects.
2. Count connected components.
3. Calculate a drop-density score near the player.
4. If score exceeds threshold, press `5` for magnet mode.
5. Stay in magnet mode until density falls below a lower threshold for several frames.
6. Press `4` to return to damage mode.

Use hysteresis so the bot does not flicker between `4` and `5`.

### Popup and Ball Task

Use the supplied ball-task screenshot as the first reference.

Detection:

- Detect green testing-area background.
- Detect red ball using HSV red mask.
- Detect grey path/endzone using saturation/value mask.
- Confirm task state only if both red ball and grey path are present.

Solve:

1. Find the red ball centroid.
2. Extract the grey path contour.
3. Find the path centerline or generate waypoints along the contour's medial axis.
4. Press mouse down on the ball.
5. Drag through waypoints with smooth movement.
6. Release inside the endzone.
7. Wait for popup/task state to disappear before resuming grind.

Library details:

- OpenCV for masks and contours.
- Deterministic shortest-path routing through the grey path mask for curved paths.
- PyAutoGUI for controlled drag movements.

## Input Simulation

Primary:

- `pydirectinput` for keyboard presses and movement keys.
- `pyautogui` for mouse clicks, cursor movement, and dragging.

Supporting:

- `pygetwindow` to locate/focus the browser window.
- `pynput` for an emergency stop hotkey.

Emergency controls:

- Global pause/resume hotkey.
- Global kill switch.
- Optional dry-run mode where the bot draws overlays and logs intended actions without sending input.

## Bot Architecture

Use a state machine with a fixed tick loop.

Main states:

- `BOOT`
- `FOCUS_GAME`
- `ACTIVATE_SPEED`
- `NAVIGATE_TO_PORTAL`
- `WAIT_FOR_DIMENSION_CHANGE`
- `NAVIGATE_TO_GRIND_ZONE`
- `GRIND_COMBAT`
- `COLLECT_DROPS`
- `SOLVE_BALL_TASK`
- `FORCED_DEATH`
- `DEATH_WAIT`
- `RECOVER_AND_RESUME`
- `PAUSED`
- `ERROR`

Each tick should:

1. Capture relevant screen regions.
2. Update detectors.
3. Update world/bot state.
4. Choose one action.
5. Execute the action.
6. Log state, confidence, and debug information.

Use `time.monotonic()` for timers:

- Forced death interval: 3 hours.
- Death wait interval: 30 minutes.
- Consecutive-frame confirmation windows for dimension transition, popup state, and drop-density state.

## Project Structure

```txt
florr.io/
  TECH_STACK.md
  README.md
  requirements.txt
  config/
    default.yaml
    calibration.yaml
  assets/
    references/
      first_map.png
      first_portal_crop.png
      second_map.png
      second_target_crop.png
      ball_task.png
    debug/
  src/
    florr_bot/
      __init__.py
      main.py
      config.py
      capture.py
      input_control.py
      state_machine.py
      navigation.py
      pathfinding.py
      combat.py
      drops.py
      ball_task.py
      detectors.py
      overlay.py
      logging_utils.py
  scripts/
    calibrate_minimap.py
    calibrate_colors.py
    sample_video_frames.py
    run_debug_overlay.py
  tests/
    test_pathfinding.py
    test_color_masks.py
    fixtures/
```

## Calibration Files

Use YAML for thresholds and screen regions:

```yaml
screen:
  monitor_index: 1
  game_window_title_contains: "florr"
  minimap_roi: [0, 0, 300, 300]
  world_roi: [0, 0, 1920, 1080]

hotkeys:
  speed: "9"
  battle: "4"
  magnet: "5"

timers:
  forced_death_seconds: 10800
  death_wait_seconds: 1800

vision:
  brown_background_hsv_min: [5, 40, 20]
  brown_background_hsv_max: [35, 255, 180]
  drop_density_enter: 35
  drop_density_exit: 12
```

All numbers above are starting placeholders. They must be calibrated from live screenshots and the guide video.

## Video Usage

The attached guide video should be used to tune:

- Actual grinding-zone appearance.
- Mob movement patterns.
- Drop appearance and density.
- Timing of magnet collection.
- Any popup animation or visual state before the ball task appears.

Current local note: `ffmpeg` is not installed and OpenCV is not currently installed in the active Python environment, so video frame extraction should happen after the virtual environment and dependencies are created.

Planned helper:

```powershell
python scripts/sample_video_frames.py "C:\Users\terry\Downloads\Florr_io_guide.mp4" --start 0 --end 180 --every 5
```

## Testing Strategy

Use three testing layers:

1. Offline vision tests against saved screenshots and sampled video frames.
2. Dry-run overlay mode that shows detections, path, mob targets, drop density, and selected action.
3. Live controlled run with input enabled and an emergency stop.

Minimum tests before live grinding:

- A* returns a valid route on first-map and second-map minimap masks.
- Brown dimension detector does not trigger on the first map.
- Brown dimension detector does trigger after portal transition.
- Drop-density hysteresis switches to magnet and back.
- Ball-task detector finds the red ball and grey path.
- Emergency stop interrupts all input loops.

## Implementation Order

1. Create project skeleton, requirements, and config files.
2. Add screen capture and debug snapshot saving.
3. Add minimap ROI calibration and reference image storage.
4. Build color-mask detectors for map, target, portal, and brown dimension state.
5. Build A* pathfinding over the minimap mask.
6. Implement input control with dry-run and emergency stop.
7. Wire state machine through portal navigation and dimension verification.
8. Add second-map navigation to the grinding zone.
9. Add combat detection, kiting, and continuous attack loop.
10. Add drop-density detection and `5`/`4` switching.
11. Add popup/ball-task detector and drag solver.
12. Add forced death timer and 30-minute wait cycle.
13. Run live calibration and tune thresholds.

## Final Stack Decision

Start with a deterministic Python/OpenCV desktop bot:

- `mss` for fast screen capture.
- `opencv-python-headless` and `numpy` for vision.
- Custom A* pathfinding with `heapq`.
- `pydirectinput`, `pyautogui`, `pygetwindow`, and `pynput` for desktop control.
- Pydantic/YAML for configuration.
- Rich logs plus debug overlays for calibration.
- Pytest with saved image fixtures for repeatable detector tests.

Only add YOLO/ONNX after proving that classical OpenCV detection is not reliable enough for mobs, drops, or popup states.
