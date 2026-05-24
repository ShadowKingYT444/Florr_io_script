from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from florr_bot.capture import ScreenCapture
from florr_bot.config import load_config
from florr_bot.input_control import InputController
from florr_bot.logging_utils import configure_logging
from florr_bot.screen_analysis import detect_minimap_roi, draw_roi
from florr_bot.state_machine import FlorrBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="florr.io desktop automation bot")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--calibration", default=None, help="Optional calibration YAML overlay")
    parser.add_argument("--calibrate-screen", action="store_true", help="Focus browser, detect minimap, save calibration, and exit")
    parser.add_argument("--calibration-out", default="calibration.yaml", help="Output path for --calibrate-screen")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Capture and log without sending input")
    mode.add_argument("--live", action="store_true", help="Send keyboard and mouse input")
    parser.add_argument("--check-config", action="store_true", help="Load config and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser


def resolve_resource_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    bundled = bundle_root / candidate
    if bundled.exists():
        return bundled
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_resource_path(args.config)
    calibration_path = resolve_resource_path(args.calibration) if args.calibration else None
    config = load_config(config_path, calibration_path)
    if args.dry_run:
        config.runtime.dry_run = True
    if args.live:
        config.runtime.dry_run = False

    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
        try:
            config_is_bundled = config_path.resolve().is_relative_to(bundle_root)
        except ValueError:
            config_is_bundled = False
        if config_is_bundled:
            config.runtime.debug_dir = Path(sys.executable).parent / "assets" / "debug"

    logger = configure_logging(config.runtime.debug_dir, verbose=args.verbose)
    if args.check_config:
        logger.info("Config OK: %s", config_path)
        return 0

    capture = ScreenCapture(config.screen)
    inputs = InputController(
        movement=config.movement,
        dry_run=config.runtime.dry_run,
        logger=logger,
        emergency_stop_key=config.runtime.emergency_stop_key,
        pause_key=config.runtime.pause_key,
    )
    if args.calibrate_screen:
        try:
            return calibrate_screen(config, capture, inputs, logger, Path(args.calibration_out))
        finally:
            capture.close()
            inputs.close()

    bot = FlorrBot(config, capture, inputs, logger)
    bot.run()
    return 0


def calibrate_screen(config, capture: ScreenCapture, inputs: InputController, logger, output: Path) -> int:
    window = inputs.focus_window(
        config.screen.game_window_title_contains,
        maximize=config.screen.maximize_window,
    )
    if window is not None:
        capture.set_world_roi(capture.absolute_to_monitor_roi(*window))

    world = capture.grab_world()
    detection = detect_minimap_roi(world.bgr, config.screen.minimap_search_roi)
    config.runtime.debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(config.runtime.debug_dir / "calibration_world.png"), world.bgr)

    data = {
        "screen": {
            "world_roi": list(config.screen.world_roi),
        }
    }
    if detection is not None:
        wx, wy, _, _ = config.screen.world_roi
        minimap_roi = (wx + detection.roi[0], wy + detection.roi[1], detection.roi[2], detection.roi[3])
        data["screen"]["minimap_roi"] = list(minimap_roi)
        data["screen"]["auto_detect_minimap"] = False
        cv2.imwrite(str(config.runtime.debug_dir / "calibration_minimap_detected.png"), draw_roi(world.bgr, detection))
        logger.info(
            "Detected minimap ROI %s score=%.3f. Wrote debug screenshots to %s",
            minimap_roi,
            detection.score,
            config.runtime.debug_dir,
        )
    else:
        logger.warning("Could not detect minimap. Wrote current screen to %s", config.runtime.debug_dir)

    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    logger.info("Wrote calibration overlay: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
