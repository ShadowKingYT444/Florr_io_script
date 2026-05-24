from pathlib import Path

import cv2
import numpy as np

from florr_bot.config import load_config
from florr_bot.detectors import BackgroundDetector, PopupDetector


def test_default_config_loads() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.hotkeys.speed == "9"
    assert config.hotkeys.battle == "4"
    assert config.hotkeys.magnet == "5"
    assert config.timers.forced_death_seconds == 10800


def test_brown_background_detector_scores_brown_region() -> None:
    config = load_config(Path("config/default.yaml"))
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[:, :] = (42, 72, 120)
    detector = BackgroundDetector(config.vision.masks, sample_rois=[])
    assert detector.brown_score(frame) > 0.8


def test_popup_detector_finds_synthetic_ball_task() -> None:
    config = load_config(Path("config/default.yaml"))
    frame = np.zeros((220, 220, 3), dtype=np.uint8)
    frame[:, :] = (35, 90, 35)
    cv2.line(frame, (80, 150), (150, 60), (130, 130, 130), 18)
    cv2.circle(frame, (80, 150), 10, (0, 0, 255), -1)
    detector = PopupDetector(config.vision.masks, min_green_fraction=0.05)
    assert detector.looks_like_ball_task(frame)
