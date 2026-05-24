from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Hsv, MaskConfig, Roi


@dataclass(frozen=True)
class Blob:
    centroid: tuple[int, int]
    area: float
    bbox: tuple[int, int, int, int]


def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def hsv_mask(hsv: np.ndarray, hsv_min: Hsv, hsv_max: Hsv) -> np.ndarray:
    return cv2.inRange(hsv, np.array(hsv_min, dtype=np.uint8), np.array(hsv_max, dtype=np.uint8))


def dual_hsv_mask(hsv: np.ndarray, min_a: Hsv, max_a: Hsv, min_b: Hsv, max_b: Hsv) -> np.ndarray:
    return cv2.bitwise_or(hsv_mask(hsv, min_a, max_a), hsv_mask(hsv, min_b, max_b))


def clean_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_blobs(mask: np.ndarray, min_area: float = 1.0, max_area: float | None = None) -> list[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[Blob] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        blobs.append(Blob(centroid=(cx, cy), area=area, bbox=(x, y, w, h)))
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs


def mask_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def crop_roi(frame: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, w, h = roi
    return frame[max(0, y) : max(0, y) + h, max(0, x) : max(0, x) + w]


class BackgroundDetector:
    def __init__(self, masks: MaskConfig, sample_rois: list[Roi], threshold: float = 0.42):
        self.masks = masks
        self.sample_rois = sample_rois
        self.threshold = threshold

    def brown_score(self, world_bgr: np.ndarray) -> float:
        regions = [crop_roi(world_bgr, roi) for roi in self.sample_rois] if self.sample_rois else [world_bgr]
        scores: list[float] = []
        for region in regions:
            if region.size == 0:
                continue
            hsv = to_hsv(region)
            mask = hsv_mask(hsv, self.masks.brown_background_hsv_min, self.masks.brown_background_hsv_max)
            scores.append(mask_fraction(mask))
        return float(np.mean(scores)) if scores else 0.0

    def is_brown_dimension(self, world_bgr: np.ndarray) -> bool:
        return self.brown_score(world_bgr) >= self.threshold


class PopupDetector:
    def __init__(self, masks: MaskConfig, min_green_fraction: float = 0.12):
        self.masks = masks
        self.min_green_fraction = min_green_fraction

    def looks_like_ball_task(self, world_bgr: np.ndarray) -> bool:
        hsv = to_hsv(world_bgr)
        green = hsv_mask(hsv, self.masks.task_green_hsv_min, self.masks.task_green_hsv_max)
        red = dual_hsv_mask(
            hsv,
            self.masks.red_ball_hsv_min_1,
            self.masks.red_ball_hsv_max_1,
            self.masks.red_ball_hsv_min_2,
            self.masks.red_ball_hsv_max_2,
        )
        grey = hsv_mask(hsv, self.masks.grey_path_hsv_min, self.masks.grey_path_hsv_max)
        return (
            mask_fraction(green) >= self.min_green_fraction
            and len(find_blobs(red, min_area=20)) > 0
            and len(find_blobs(grey, min_area=120)) > 0
        )
