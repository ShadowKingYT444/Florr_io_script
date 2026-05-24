from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import Hsv, Roi


@dataclass(frozen=True)
class MinimapDetection:
    roi: Roi
    score: float
    white_fraction: float
    dark_fraction: float


@dataclass(frozen=True)
class MarkerDetection:
    point: tuple[int, int]
    area: float
    confidence: float


def detect_minimap_roi(frame_bgr: np.ndarray, search_roi: Roi | None = None) -> MinimapDetection | None:
    """Find the square black/white minimap on the current game/browser frame."""

    height, width = frame_bgr.shape[:2]
    if search_roi is None:
        sx, sy, sw, sh = width // 2, 0, width // 2, max(height // 2, 1)
    else:
        sx, sy, sw, sh = _clamp_roi(search_roi, width, height)

    crop = frame_bgr[sy : sy + sh, sx : sx + sw]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 175)).astype(np.uint8)
    dark = (hsv[:, :, 2] < 65).astype(np.uint8)
    white_i = _integral(white)
    dark_i = _integral(dark)

    min_size = max(80, min(width, height) // 16)
    max_size = min(340, max(90, min(sw, sh)))
    best: MinimapDetection | None = None

    for size in range(min_size, max_size + 1, 12):
        step = max(6, size // 8)
        for y in range(0, max(1, sh - size + 1), step):
            for x in range(0, max(1, sw - size + 1), step):
                area = float(size * size)
                white_fraction = _window_sum(white_i, x, y, size, size) / area
                if white_fraction < 0.08 or white_fraction > 0.62:
                    continue
                dark_fraction = _window_sum(dark_i, x, y, size, size) / area
                if dark_fraction < 0.08 or dark_fraction > 0.80:
                    continue

                right_bias = (sx + x + size) / max(width, 1)
                top_bias = 1.0 - ((sy + y) / max(height, 1))
                contrast_score = 4.0 * white_fraction * dark_fraction
                score = contrast_score + 0.12 * right_bias + 0.05 * top_bias
                candidate = MinimapDetection(
                    roi=(sx + x, sy + y, size, size),
                    score=float(score),
                    white_fraction=float(white_fraction),
                    dark_fraction=float(dark_fraction),
                )
                if best is None or candidate.score > best.score:
                    best = candidate

    if best is None or best.score < 0.12:
        return None
    return best


def detect_marker(
    minimap_bgr: np.ndarray,
    hsv_min: Hsv,
    hsv_max: Hsv,
    *,
    min_area: float = 3.0,
    max_area: float = 450.0,
    prefer_center: bool = True,
) -> MarkerDetection | None:
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_min, dtype=np.uint8), np.array(hsv_max, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[MarkerDetection] = []
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = int(round(moments["m10"] / moments["m00"]))
        y = int(round(moments["m01"] / moments["m00"]))
        distance = float(np.hypot(x - center[0], y - center[1]))
        confidence = area / (1.0 + distance if prefer_center else 1.0)
        candidates.append(MarkerDetection(point=(x, y), area=area, confidence=confidence))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.confidence)


def draw_roi(frame_bgr: np.ndarray, detection: MinimapDetection | None) -> np.ndarray:
    out = frame_bgr.copy()
    if detection is None:
        return out
    x, y, w, h = detection.roi
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        out,
        f"minimap {detection.score:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def _window_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2 = x + w
    y2 = y + h
    return float(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def _clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    x, y, w, h = roi
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h
