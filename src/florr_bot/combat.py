from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CombatConfig, MaskConfig
from .detectors import Blob, clean_mask, find_blobs, hsv_mask, to_hsv
from .input_control import MovementIntent


@dataclass(frozen=True)
class CombatDecision:
    should_attack: bool
    movement: MovementIntent
    target: tuple[int, int] | None
    mob_count: int
    nearest_distance: float | None


class MobDetector:
    def __init__(self, masks: MaskConfig, combat: CombatConfig):
        self.masks = masks
        self.combat = combat

    def detect(self, world_bgr: np.ndarray) -> list[Blob]:
        hsv = to_hsv(world_bgr)
        # A deliberately broad first-pass heuristic: highly saturated moving objects,
        # excluding very dark map background and tiny loot specks.
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = np.where((saturation > 70) & (value > 45), 255, 0).astype(np.uint8)
        mask = clean_mask(mask, kernel_size=3, iterations=1)
        return find_blobs(mask, self.combat.mob_min_area_px, self.combat.mob_max_area_px)


class CombatPlanner:
    def __init__(self, combat: CombatConfig):
        self.combat = combat

    def decide(self, world_shape: tuple[int, int, int], mobs: list[Blob]) -> CombatDecision:
        height, width = world_shape[:2]
        center = (width / 2.0, height / 2.0)
        if not mobs or not self.combat.enabled:
            return CombatDecision(False, MovementIntent(0.0, 0.0), None, len(mobs), None)

        def dist(blob: Blob) -> float:
            return math.hypot(blob.centroid[0] - center[0], blob.centroid[1] - center[1])

        nearest = min(mobs, key=dist)
        nearest_distance = dist(nearest)
        attack = nearest_distance <= self.combat.attack_radius_px

        if nearest_distance <= self.combat.threat_radius_px:
            dx = center[0] - nearest.centroid[0]
            dy = center[1] - nearest.centroid[1]
        elif len(mobs) >= 3:
            avg_x = sum(blob.centroid[0] for blob in mobs) / len(mobs)
            avg_y = sum(blob.centroid[1] for blob in mobs) / len(mobs)
            cluster_distance = math.hypot(avg_x - center[0], avg_y - center[1])
            if cluster_distance < self.combat.kite_radius_px:
                dx = center[0] - avg_x
                dy = center[1] - avg_y
            else:
                dx = dy = 0.0
        else:
            dx = dy = 0.0

        mag = math.hypot(dx, dy)
        movement = MovementIntent(dx / mag, dy / mag) if mag > 0 else MovementIntent(0.0, 0.0)
        return CombatDecision(attack, movement, nearest.centroid, len(mobs), nearest_distance)
