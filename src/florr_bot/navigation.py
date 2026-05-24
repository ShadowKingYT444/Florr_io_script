"""OpenCV/NumPy minimap navigation helpers for florr.io automation.

Public point values use image coordinates, ``(x, y)``. Masks and grids remain
indexed with NumPy's row-major convention, ``mask[y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, TypeAlias

import numpy as np

from .pathfinding import (
    GridPathfinder,
    GridPoint,
    nearest_walkable_point,
    simplify_path_with_line_of_sight,
)

try:  # pragma: no cover - exercised only when OpenCV is unavailable.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


HSVTriplet = tuple[int, int, int]


@dataclass(frozen=True)
class HSVRange:
    """Inclusive OpenCV HSV range.

    Hue uses OpenCV's 0..179 scale. Hue ranges may wrap around zero by setting
    ``lower[0] > upper[0]``.
    """

    lower: HSVTriplet
    upper: HSVTriplet


HSVRangeLike: TypeAlias = HSVRange | tuple[Sequence[int | float], Sequence[int | float]]


@dataclass(frozen=True)
class NavigationMasks:
    hsv: np.ndarray
    walkable: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class TargetCentroid:
    point: GridPoint
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TargetCandidate:
    point: GridPoint
    kind: str
    area: int = 0


@dataclass(frozen=True)
class PathPlan:
    start: GridPoint
    goal: GridPoint
    grid_start: GridPoint | None
    grid_goal: GridPoint | None
    grid_path: tuple[GridPoint, ...]
    pixel_path: tuple[GridPoint, ...]


@dataclass(frozen=True)
class MovementIntent:
    x: float
    y: float
    distance: float

    @property
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class NavigationPlan:
    target: TargetCandidate | None
    path: PathPlan | None
    movement: MovementIntent


def bgr_to_hsv(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to OpenCV HSV."""

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for BGR to HSV conversion")

    image = np.asarray(bgr_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BGR image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def ensure_hsv(image: np.ndarray, input_space: str = "bgr") -> np.ndarray:
    """Return an HSV image from a BGR or HSV minimap image."""

    normalized_space = input_space.lower()
    if normalized_space == "hsv":
        hsv = np.asarray(image)
        if hsv.ndim != 3 or hsv.shape[2] != 3:
            raise ValueError("HSV image must have shape (height, width, 3)")
        return hsv.astype(np.uint8, copy=False)
    if normalized_space == "bgr":
        return bgr_to_hsv(image)
    raise ValueError("input_space must be either 'bgr' or 'hsv'")


def mask_from_hsv_ranges(hsv_image: np.ndarray, ranges: Iterable[HSVRangeLike] | None) -> np.ndarray:
    """Create a boolean mask from one or more inclusive HSV ranges."""

    hsv = ensure_hsv(hsv_image, input_space="hsv")
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for hsv_range in _normalize_hsv_ranges(ranges):
        lower = np.array(hsv_range.lower, dtype=np.int16)
        upper = np.array(hsv_range.upper, dtype=np.int16)

        hue = hsv[:, :, 0].astype(np.int16)
        saturation = hsv[:, :, 1].astype(np.int16)
        value = hsv[:, :, 2].astype(np.int16)

        if lower[1] > upper[1] or lower[2] > upper[2]:
            raise ValueError("saturation and value range lower bounds must be <= upper bounds")

        if lower[0] <= upper[0]:
            hue_mask = (hue >= lower[0]) & (hue <= upper[0])
        else:
            hue_mask = (hue >= lower[0]) | (hue <= upper[0])

        mask |= (
            hue_mask
            & (saturation >= lower[1])
            & (saturation <= upper[1])
            & (value >= lower[2])
            & (value <= upper[2])
        )
    return mask


def create_navigation_masks(
    image: np.ndarray,
    walkable_ranges: Iterable[HSVRangeLike] | None,
    target_ranges: Iterable[HSVRangeLike] | None,
    input_space: str = "bgr",
) -> NavigationMasks:
    """Build walkable and target masks from a minimap image."""

    hsv = ensure_hsv(image, input_space=input_space)
    return NavigationMasks(
        hsv=hsv,
        walkable=mask_from_hsv_ranges(hsv, walkable_ranges),
        target=mask_from_hsv_ranges(hsv, target_ranges),
    )


def find_target_centroids(target_mask: np.ndarray, min_area: int = 1) -> list[TargetCentroid]:
    """Return connected target centroids sorted by image position."""

    mask = _as_bool_mask(target_mask)
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if cv2 is None:  # pragma: no cover
        return _find_target_centroids_numpy(mask, min_area=min_area)

    components = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count, _labels, stats, centroids = components
    found: list[TargetCentroid] = []
    height, width = mask.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(math.floor(float(centroids[label][0]) + 0.5))
        y = int(math.floor(float(centroids[label][1]) + 0.5))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        found.append(TargetCentroid(point=(x, y), area=area, bbox=bbox))
    return sorted(found, key=lambda centroid: (centroid.point[1], centroid.point[0], centroid.area))


def find_nearest_target_or_hint(
    target_mask: np.ndarray,
    origin: Sequence[int | float],
    hints: Iterable[Sequence[int | float]] | None = None,
    min_area: int = 1,
    prefer_centroids: bool = True,
) -> TargetCandidate | None:
    """Find the nearest target centroid, falling back to the nearest hint."""

    origin_point = _as_point(origin)
    centroids = [
        TargetCandidate(point=centroid.point, kind="centroid", area=centroid.area)
        for centroid in find_target_centroids(target_mask, min_area=min_area)
    ]
    hint_candidates = [
        TargetCandidate(point=_as_point(hint), kind="hint", area=0)
        for hint in (hints or [])
    ]

    if prefer_centroids and centroids:
        return _nearest_candidate(origin_point, centroids)
    return _nearest_candidate(origin_point, [*centroids, *hint_candidates])


def mask_to_grid(
    mask: np.ndarray,
    cell_size: int = 4,
    min_fraction: float = 0.5,
) -> np.ndarray:
    """Downsample a pixel mask into a boolean navigation grid."""

    bool_mask = _as_bool_mask(mask)
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_fraction must be between 0 and 1")

    height, width = bool_mask.shape
    grid_height = math.ceil(height / cell_size)
    grid_width = math.ceil(width / cell_size)
    grid = np.zeros((grid_height, grid_width), dtype=bool)

    for gy in range(grid_height):
        y0 = gy * cell_size
        y1 = min(y0 + cell_size, height)
        for gx in range(grid_width):
            x0 = gx * cell_size
            x1 = min(x0 + cell_size, width)
            grid[gy, gx] = float(np.mean(bool_mask[y0:y1, x0:x1])) >= min_fraction

    return grid


def pixel_to_grid_point(
    point: Sequence[int | float],
    cell_size: int,
    grid_shape: tuple[int, int] | None = None,
) -> GridPoint:
    """Convert a pixel point to a grid cell, optionally clamped to a grid shape."""

    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    x, y = _as_point(point)
    gx = math.floor(x / cell_size)
    gy = math.floor(y / cell_size)
    if grid_shape is not None:
        rows, cols = grid_shape
        gx = min(max(gx, 0), cols - 1)
        gy = min(max(gy, 0), rows - 1)
    return int(gx), int(gy)


def grid_to_pixel_point(
    point: Sequence[int | float],
    cell_size: int,
    mask_shape: tuple[int, int] | None = None,
) -> GridPoint:
    """Convert a grid cell to the center pixel for that cell."""

    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    gx, gy = _as_point(point)
    x = gx * cell_size + cell_size // 2
    y = gy * cell_size + cell_size // 2
    if mask_shape is not None:
        height, width = mask_shape
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
    return int(x), int(y)


def plan_path_from_mask(
    walkable_mask: np.ndarray,
    start: Sequence[int | float],
    goal: Sequence[int | float],
    cell_size: int = 4,
    min_walkable_fraction: float = 0.5,
    allow_diagonal: bool = True,
    simplify: bool = True,
) -> PathPlan:
    """Plan a path between pixel points using a walkable minimap mask."""

    walkable = _as_bool_mask(walkable_mask)
    grid = mask_to_grid(walkable, cell_size=cell_size, min_fraction=min_walkable_fraction)
    start_point = _as_point(start)
    goal_point = _as_point(goal)
    start_cell = nearest_walkable_point(
        grid,
        pixel_to_grid_point(start_point, cell_size, grid_shape=grid.shape),
    )
    goal_cell = nearest_walkable_point(
        grid,
        pixel_to_grid_point(goal_point, cell_size, grid_shape=grid.shape),
    )

    if start_cell is None or goal_cell is None:
        return PathPlan(
            start=start_point,
            goal=goal_point,
            grid_start=start_cell,
            grid_goal=goal_cell,
            grid_path=(),
            pixel_path=(),
        )

    grid_path = GridPathfinder(grid, allow_diagonal=allow_diagonal).find_path(start_cell, goal_cell)
    if grid_path and simplify:
        grid_path = simplify_path_with_line_of_sight(grid_path, grid)

    pixel_path = tuple(
        grid_to_pixel_point(point, cell_size=cell_size, mask_shape=walkable.shape)
        for point in grid_path
    )
    return PathPlan(
        start=start_point,
        goal=goal_point,
        grid_start=start_cell,
        grid_goal=goal_cell,
        grid_path=tuple(grid_path),
        pixel_path=pixel_path,
    )


def next_waypoint(
    path: Iterable[Sequence[int | float]],
    current: Sequence[int | float],
    deadzone: float = 2.0,
) -> GridPoint | None:
    """Return the first waypoint far enough away to steer toward."""

    current_point = _as_point(current)
    for waypoint in path:
        point = _as_point(waypoint)
        if math.hypot(point[0] - current_point[0], point[1] - current_point[1]) > deadzone:
            return point
    return None


def movement_intent_from_waypoint(
    current: Sequence[int | float],
    waypoint: Sequence[int | float] | None,
    deadzone: float = 1.0,
) -> MovementIntent:
    """Convert a waypoint into normalized ``x``/``y`` movement intent."""

    if waypoint is None:
        return MovementIntent(x=0.0, y=0.0, distance=0.0)

    current_point = _as_point(current)
    waypoint_point = _as_point(waypoint)
    dx = float(waypoint_point[0] - current_point[0])
    dy = float(waypoint_point[1] - current_point[1])
    distance = math.hypot(dx, dy)
    if distance <= deadzone or distance == 0.0:
        return MovementIntent(x=0.0, y=0.0, distance=distance)
    return MovementIntent(x=dx / distance, y=dy / distance, distance=distance)


def movement_intent_from_path(
    current: Sequence[int | float],
    path: Iterable[Sequence[int | float]],
    deadzone: float = 2.0,
) -> MovementIntent:
    """Convert the next useful waypoint in a path into movement intent."""

    waypoint = next_waypoint(path, current=current, deadzone=deadzone)
    return movement_intent_from_waypoint(current, waypoint, deadzone=deadzone)


class MinimapNavigator:
    """Small orchestration helper for mask creation, targeting, and path plans."""

    def __init__(
        self,
        walkable_ranges: Iterable[HSVRangeLike] | None,
        target_ranges: Iterable[HSVRangeLike] | None,
        *,
        cell_size: int = 4,
        min_walkable_fraction: float = 0.5,
        min_target_area: int = 1,
        allow_diagonal: bool = True,
        simplify: bool = True,
    ) -> None:
        self.walkable_ranges = tuple(_normalize_hsv_ranges(walkable_ranges))
        self.target_ranges = tuple(_normalize_hsv_ranges(target_ranges))
        self.cell_size = cell_size
        self.min_walkable_fraction = min_walkable_fraction
        self.min_target_area = min_target_area
        self.allow_diagonal = allow_diagonal
        self.simplify = simplify

    def masks(self, image: np.ndarray, input_space: str = "bgr") -> NavigationMasks:
        return create_navigation_masks(
            image,
            walkable_ranges=self.walkable_ranges,
            target_ranges=self.target_ranges,
            input_space=input_space,
        )

    def target(
        self,
        image: np.ndarray,
        origin: Sequence[int | float],
        *,
        hints: Iterable[Sequence[int | float]] | None = None,
        input_space: str = "bgr",
    ) -> TargetCandidate | None:
        masks = self.masks(image, input_space=input_space)
        return find_nearest_target_or_hint(
            masks.target,
            origin=origin,
            hints=hints,
            min_area=self.min_target_area,
        )

    def plan(
        self,
        image: np.ndarray,
        origin: Sequence[int | float],
        *,
        hints: Iterable[Sequence[int | float]] | None = None,
        input_space: str = "bgr",
        deadzone: float = 2.0,
    ) -> NavigationPlan:
        masks = self.masks(image, input_space=input_space)
        target = find_nearest_target_or_hint(
            masks.target,
            origin=origin,
            hints=hints,
            min_area=self.min_target_area,
        )
        if target is None:
            return NavigationPlan(target=None, path=None, movement=MovementIntent(0.0, 0.0, 0.0))

        path = plan_path_from_mask(
            masks.walkable,
            start=origin,
            goal=target.point,
            cell_size=self.cell_size,
            min_walkable_fraction=self.min_walkable_fraction,
            allow_diagonal=self.allow_diagonal,
            simplify=self.simplify,
        )
        movement = movement_intent_from_path(origin, path.pixel_path, deadzone=deadzone)
        return NavigationPlan(target=target, path=path, movement=movement)


def _normalize_hsv_ranges(ranges: Iterable[HSVRangeLike] | None) -> list[HSVRange]:
    if ranges is None:
        return []
    normalized: list[HSVRange] = []
    for item in ranges:
        if isinstance(item, HSVRange):
            lower = _as_hsv_triplet(item.lower)
            upper = _as_hsv_triplet(item.upper)
        else:
            lower_raw, upper_raw = item
            lower = _as_hsv_triplet(lower_raw)
            upper = _as_hsv_triplet(upper_raw)
        normalized.append(HSVRange(lower=lower, upper=upper))
    return normalized


def _as_hsv_triplet(value: Sequence[int | float]) -> HSVTriplet:
    if len(value) != 3:
        raise ValueError("HSV bounds must have exactly three values")
    return int(value[0]), int(value[1]), int(value[2])


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    bool_mask = np.asarray(mask, dtype=bool)
    if bool_mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return bool_mask


def _as_point(point: Sequence[int | float]) -> GridPoint:
    if len(point) != 2:
        raise ValueError("points must contain exactly two coordinates")
    return int(point[0]), int(point[1])


def _nearest_candidate(origin: GridPoint, candidates: Sequence[TargetCandidate]) -> TargetCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            (candidate.point[0] - origin[0]) ** 2 + (candidate.point[1] - origin[1]) ** 2,
            candidate.point[1],
            candidate.point[0],
            candidate.kind,
        ),
    )


def _find_target_centroids_numpy(mask: np.ndarray, min_area: int) -> list[TargetCentroid]:
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    centroids: list[TargetCentroid] = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_x), int(start_y))]
        visited[start_y, start_x] = True
        xs: list[int] = []
        ys: list[int] = []

        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))

        area = len(xs)
        if area < min_area:
            continue
        centroid_x = int(math.floor(float(sum(xs)) / area + 0.5))
        centroid_y = int(math.floor(float(sum(ys)) / area + 0.5))
        min_x = min(xs)
        min_y = min(ys)
        bbox = (min_x, min_y, max(xs) - min_x + 1, max(ys) - min_y + 1)
        centroids.append(TargetCentroid(point=(centroid_x, centroid_y), area=area, bbox=bbox))

    return sorted(centroids, key=lambda centroid: (centroid.point[1], centroid.point[0], centroid.area))


__all__ = [
    "HSVRange",
    "MovementIntent",
    "MinimapNavigator",
    "NavigationMasks",
    "NavigationPlan",
    "PathPlan",
    "TargetCandidate",
    "TargetCentroid",
    "bgr_to_hsv",
    "create_navigation_masks",
    "ensure_hsv",
    "find_nearest_target_or_hint",
    "find_target_centroids",
    "grid_to_pixel_point",
    "mask_from_hsv_ranges",
    "mask_to_grid",
    "movement_intent_from_path",
    "movement_intent_from_waypoint",
    "next_waypoint",
    "pixel_to_grid_point",
    "plan_path_from_mask",
]
