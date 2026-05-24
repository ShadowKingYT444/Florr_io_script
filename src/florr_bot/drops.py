from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DropsConfig, MaskConfig
from .detectors import clean_mask, find_blobs, hsv_mask, to_hsv


@dataclass(frozen=True)
class DropState:
    density: int
    should_collect: bool


class DropDetector:
    def __init__(self, masks: MaskConfig, drops: DropsConfig):
        self.masks = masks
        self.drops = drops
        self.collecting = False

    def detect(self, world_bgr: np.ndarray) -> DropState:
        hsv = to_hsv(world_bgr)
        mask = hsv_mask(hsv, self.masks.drop_hsv_min, self.masks.drop_hsv_max)
        mask = clean_mask(mask, kernel_size=3, iterations=1)
        blobs = find_blobs(mask, self.drops.min_blob_area_px, self.drops.max_blob_area_px)
        density = len(blobs)
        if self.collecting:
            self.collecting = density > self.drops.density_exit
        else:
            self.collecting = density >= self.drops.density_enter
        return DropState(density=density, should_collect=self.collecting)
