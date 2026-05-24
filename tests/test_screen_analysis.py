import cv2
import numpy as np

from florr_bot.screen_analysis import detect_marker, detect_minimap_roi


def test_detect_minimap_roi_finds_square_black_white_map() -> None:
    frame = np.full((500, 900, 3), (30, 120, 30), dtype=np.uint8)
    x, y, size = 680, 30, 150
    frame[y : y + size, x : x + size] = (10, 10, 10)
    cv2.line(frame, (x + 20, y + 20), (x + 130, y + 20), (245, 245, 245), 14)
    cv2.line(frame, (x + 130, y + 20), (x + 130, y + 130), (245, 245, 245), 14)
    cv2.line(frame, (x + 40, y + 120), (x + 130, y + 130), (245, 245, 245), 14)

    detection = detect_minimap_roi(frame)

    assert detection is not None
    rx, ry, rw, rh = detection.roi
    assert abs((rx + rw // 2) - (x + size // 2)) < 35
    assert abs((ry + rh // 2) - (y + size // 2)) < 35


def test_detect_marker_finds_yellow_player_dot() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (72, 34), 5, (0, 220, 255), -1)

    marker = detect_marker(image, (15, 60, 80), (38, 255, 255), prefer_center=False)

    assert marker is not None
    assert marker.point == (72, 34)
