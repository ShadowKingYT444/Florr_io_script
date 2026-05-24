import cv2
import numpy as np

import florr_bot.ball_task as ball_task
from florr_bot.ball_task import BallTaskDetection, BallTaskDetector, BallTaskResult


GREEN_BGR = (42, 132, 62)
GREY_BGR = (148, 148, 148)
RED_BGR = (20, 20, 238)


def make_straight_task() -> np.ndarray:
    image = np.full((180, 260, 3), GREEN_BGR, dtype=np.uint8)
    cv2.line(image, (42, 90), (212, 90), GREY_BGR, 28, cv2.LINE_AA)
    cv2.circle(image, (222, 90), 25, GREY_BGR, -1, cv2.LINE_AA)
    cv2.circle(image, (42, 90), 13, RED_BGR, -1, cv2.LINE_AA)
    return image


def make_curved_task() -> np.ndarray:
    image = np.full((230, 240, 3), GREEN_BGR, dtype=np.uint8)
    cv2.line(image, (42, 56), (168, 56), GREY_BGR, 26, cv2.LINE_AA)
    cv2.line(image, (168, 56), (168, 174), GREY_BGR, 26, cv2.LINE_AA)
    cv2.circle(image, (168, 184), 22, GREY_BGR, -1, cv2.LINE_AA)
    cv2.circle(image, (42, 56), 12, RED_BGR, -1, cv2.LINE_AA)
    return image


def test_detects_red_ball_and_grey_path() -> None:
    detector = BallTaskDetector(min_path_area=150)

    detection = detector.detect(make_straight_task())

    assert isinstance(detection, BallTaskDetection)
    assert detection.present
    assert detection.ball is not None
    assert detection.path is not None
    assert detection.ball.center == (42, 90)
    assert 10 <= detection.ball.radius <= 15
    assert detection.path.area > 3_000
    assert detection.red_mask.dtype == np.uint8
    assert detection.grey_mask.shape == (180, 260)


def test_generates_ordered_waypoints_to_straight_endzone() -> None:
    detector = BallTaskDetector(min_path_area=150, waypoint_spacing=24)

    result = detector.generate_drag_waypoints(make_straight_task())

    assert isinstance(result, BallTaskResult)
    assert result.success
    assert result.detection.ball is not None
    assert result.waypoints[0] == result.detection.ball.center
    assert result.target is not None
    assert result.target[0] >= 210
    assert 60 <= result.target[1] <= 120
    assert result.path_length > 150
    assert 4 <= len(result.waypoints) <= detector.max_waypoints


def test_generates_waypoints_around_a_bend() -> None:
    detector = BallTaskDetector(min_path_area=150, waypoint_spacing=22)

    result = detector.solve(make_curved_task())

    assert result.success
    assert result.target is not None
    assert result.target[1] >= 170
    assert any(x >= 145 and y <= 75 for x, y in result.waypoints)
    assert any(x >= 145 and y >= 145 for x, y in result.waypoints)


def test_returns_failure_when_grey_path_is_missing() -> None:
    detector = BallTaskDetector(min_path_area=150)
    image = np.full((120, 160, 3), GREEN_BGR, dtype=np.uint8)
    cv2.circle(image, (40, 60), 12, RED_BGR, -1, cv2.LINE_AA)

    result = detector.generate_drag_waypoints(image)

    assert not result.success
    assert result.reason == "grey path not detected"
    assert result.waypoints == [(40, 60)]


def test_mask_route_fallback_still_reaches_endzone(monkeypatch) -> None:
    monkeypatch.setattr(ball_task, "_skimage_skeletonize", None)
    detector = BallTaskDetector(min_path_area=150, waypoint_spacing=30)

    result = detector.generate_drag_waypoints(make_straight_task())

    assert result.success
    assert result.used_fallback
    assert result.target is not None
    assert result.target[0] >= 210
