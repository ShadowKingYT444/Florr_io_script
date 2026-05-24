"""Vision solver for the florr.io red-ball popup task.

The solver is deliberately deterministic: it uses HSV color masks, connected
components, and shortest paths over the grey path mask. Coordinates are
reported in OpenCV image order as ``(x, y)`` pixel tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, pi, sqrt
from typing import Iterable

import cv2
import numpy as np

# Kept as a module variable so tests and future calibration can explicitly
# enable a skeletonizer, but the default build avoids scikit-image/SciPy to keep
# the executable smaller and lighter at runtime.
_skimage_skeletonize = None


Point = tuple[int, int]
Box = tuple[int, int, int, int]

__all__ = [
    "Point",
    "Box",
    "BallDetection",
    "PathDetection",
    "BallTaskDetection",
    "BallTaskResult",
    "BallTaskDetector",
]


@dataclass(frozen=True)
class BallDetection:
    """Detected red ball geometry."""

    center: Point
    radius: float
    area: float
    bbox: Box
    confidence: float


@dataclass(frozen=True)
class PathDetection:
    """Detected grey drag-path/endzone component."""

    area: int
    bbox: Box
    contour_area: float
    confidence: float
    mask: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class BallTaskDetection:
    """Combined detection result for the popup task."""

    present: bool
    ball: BallDetection | None
    path: PathDetection | None
    confidence: float
    image_shape: tuple[int, int]
    red_mask: np.ndarray = field(repr=False, compare=False)
    grey_mask: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class BallTaskResult:
    """Ordered drag waypoints and metadata for solving the popup task."""

    success: bool
    waypoints: list[Point]
    detection: BallTaskDetection
    target: Point | None
    reason: str = ""
    used_fallback: bool = False
    path_length: float = 0.0


class BallTaskDetector:
    """Detect and solve the red-ball-on-grey-path popup.

    The input frame is expected to be an OpenCV BGR or BGRA ``uint8`` image.
    Masks are tuned to be conservative and calibration-friendly: red is found
    by hue wraparound, while the path/endzone is a low-saturation, mid-value
    grey component near the red ball.
    """

    def __init__(
        self,
        *,
        min_ball_area: int = 40,
        min_path_area: int = 250,
        red_min_saturation: int = 80,
        red_min_value: int = 70,
        grey_max_saturation: int = 55,
        grey_min_value: int = 45,
        grey_max_value: int = 245,
        waypoint_spacing: int = 18,
        max_waypoints: int = 32,
    ) -> None:
        self.min_ball_area = int(min_ball_area)
        self.min_path_area = int(min_path_area)
        self.red_min_saturation = int(red_min_saturation)
        self.red_min_value = int(red_min_value)
        self.grey_max_saturation = int(grey_max_saturation)
        self.grey_min_value = int(grey_min_value)
        self.grey_max_value = int(grey_max_value)
        self.waypoint_spacing = int(waypoint_spacing)
        self.max_waypoints = int(max_waypoints)

    def detect(self, image: np.ndarray) -> BallTaskDetection:
        """Return red-ball and grey-path detections for ``image``."""

        bgr = self._as_bgr_uint8(image)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        red_mask = self._red_mask(hsv)
        ball = self._detect_ball(red_mask)

        raw_grey_mask = self._grey_mask(bgr, hsv, red_mask)
        path = self._detect_path(raw_grey_mask, ball)
        grey_mask = path.mask if path is not None else np.zeros(raw_grey_mask.shape, dtype=np.uint8)

        present = ball is not None and path is not None
        if present:
            confidence = min(ball.confidence, path.confidence)
        else:
            confidence = 0.0

        return BallTaskDetection(
            present=present,
            ball=ball,
            path=path,
            confidence=confidence,
            image_shape=red_mask.shape,
            red_mask=red_mask,
            grey_mask=grey_mask,
        )

    def solve(self, image: np.ndarray) -> BallTaskResult:
        """Alias for :meth:`generate_drag_waypoints`."""

        return self.generate_drag_waypoints(image)

    def generate_waypoints(self, image: np.ndarray) -> BallTaskResult:
        """Short alias for :meth:`generate_drag_waypoints`."""

        return self.generate_drag_waypoints(image)

    def generate_drag_waypoints(self, image: np.ndarray) -> BallTaskResult:
        """Generate ordered drag waypoints from the ball to the path endzone."""

        detection = self.detect(image)
        if detection.ball is None:
            return BallTaskResult(
                success=False,
                waypoints=[],
                detection=detection,
                target=None,
                reason="red ball not detected",
            )
        if detection.path is None:
            return BallTaskResult(
                success=False,
                waypoints=[detection.ball.center],
                detection=detection,
                target=None,
                reason="grey path not detected",
            )

        route, used_fallback = self._ordered_route(detection.path.mask, detection.ball.center)
        if len(route) < 2:
            return BallTaskResult(
                success=False,
                waypoints=[detection.ball.center],
                detection=detection,
                target=None,
                reason="no reachable grey route found",
                used_fallback=used_fallback,
            )

        waypoints = self._make_drag_waypoints(detection.ball.center, route)
        path_length = self._polyline_length(waypoints)
        return BallTaskResult(
            success=True,
            waypoints=waypoints,
            detection=detection,
            target=waypoints[-1],
            used_fallback=used_fallback,
            path_length=path_length,
        )

    def _red_mask(self, hsv: np.ndarray) -> np.ndarray:
        sat = self.red_min_saturation
        val = self.red_min_value
        low_red = cv2.inRange(hsv, (0, sat, val), (10, 255, 255))
        high_red = cv2.inRange(hsv, (170, sat, val), (179, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)
        mask = self._clean_mask(mask, open_size=3, close_size=5)
        return mask

    def _grey_mask(self, bgr: np.ndarray, hsv: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        max_channel = bgr.max(axis=2)
        min_channel = bgr.min(axis=2)
        channel_spread = max_channel.astype(np.int16) - min_channel.astype(np.int16)

        grey = (
            (saturation <= self.grey_max_saturation)
            & (value >= self.grey_min_value)
            & (value <= self.grey_max_value)
            & (channel_spread <= 42)
        )
        mask = np.where(grey, 255, 0).astype(np.uint8)

        red_clear = cv2.dilate(red_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        mask[red_clear > 0] = 0
        mask = self._clean_mask(mask, open_size=3, close_size=7)
        return mask

    def _detect_ball(self, red_mask: np.ndarray) -> BallDetection | None:
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) >= self.min_ball_area]
        if not candidates:
            return None

        contour = max(candidates, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
        else:
            center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))

        (_circle_x, _circle_y), radius = cv2.minEnclosingCircle(contour)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = min(1.0, (4.0 * pi * area) / (perimeter * perimeter))
        fill = min(1.0, area / max(pi * radius * radius, 1.0))
        confidence = float(np.clip(0.55 * circularity + 0.45 * fill, 0.0, 1.0))

        return BallDetection(
            center=center,
            radius=float(radius),
            area=area,
            bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
            confidence=confidence,
        )

    def _detect_path(self, grey_mask: np.ndarray, ball: BallDetection | None) -> PathDetection | None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grey_mask, connectivity=8)
        if count <= 1:
            return None

        component_ids = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.min_path_area
        ]
        if not component_ids:
            return None

        if ball is None:
            chosen = max(component_ids, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        else:
            chosen = min(
                component_ids,
                key=lambda label: (
                    self._component_distance(labels, label, ball.center),
                    -int(stats[label, cv2.CC_STAT_AREA]),
                ),
            )

        mask = np.where(labels == chosen, 255, 0).astype(np.uint8)
        x = int(stats[chosen, cv2.CC_STAT_LEFT])
        y = int(stats[chosen, cv2.CC_STAT_TOP])
        w = int(stats[chosen, cv2.CC_STAT_WIDTH])
        h = int(stats[chosen, cv2.CC_STAT_HEIGHT])
        area = int(stats[chosen, cv2.CC_STAT_AREA])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = float(max((cv2.contourArea(contour) for contour in contours), default=0.0))
        frame_area = float(mask.shape[0] * mask.shape[1])
        area_score = min(1.0, area / max(self.min_path_area * 4.0, 1.0))
        size_score = min(1.0, contour_area / max(frame_area * 0.01, 1.0))
        confidence = float(np.clip(0.65 * area_score + 0.35 * size_score, 0.0, 1.0))

        return PathDetection(
            area=area,
            bbox=(x, y, w, h),
            contour_area=contour_area,
            confidence=confidence,
            mask=mask,
        )

    def _ordered_route(self, path_mask: np.ndarray, start: Point) -> tuple[list[Point], bool]:
        bool_mask = path_mask > 0
        if not bool_mask.any():
            return [], True

        if _skimage_skeletonize is not None:
            skeleton = _skimage_skeletonize(bool_mask)
            skeleton = self._prune_to_reachable_component(skeleton, start)
            if int(skeleton.sum()) >= 2:
                route = self._longest_shortest_path(skeleton, start)
                if len(route) >= 2:
                    return route, False

        route = self._longest_shortest_path(bool_mask, start)
        return route, True

    def _longest_shortest_path(self, graph_mask: np.ndarray, start: Point) -> list[Point]:
        if not graph_mask.any():
            return []

        x_min, y_min, x_max, y_max = self._mask_bounds(graph_mask, padding=2)
        cropped = graph_mask[y_min : y_max + 1, x_min : x_max + 1]
        local_start = (start[0] - x_min, start[1] - y_min)
        nearest = self._nearest_true_point(cropped, local_start)
        if nearest is None:
            return []

        distances, previous_y, previous_x = self._dijkstra(cropped, nearest)
        finite = np.isfinite(distances)
        if int(finite.sum()) < 2:
            return []

        max_distance = float(np.max(distances[finite]))
        farthest_candidates = np.argwhere(np.isclose(distances, max_distance))
        farthest_local = self._choose_endpoint(farthest_candidates, local_start)
        local_route = self._reconstruct_route(previous_y, previous_x, farthest_local)
        return [(int(x + x_min), int(y + y_min)) for x, y in local_route]

    def _dijkstra(
        self, graph_mask: np.ndarray, start: Point
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = graph_mask.shape
        distances = np.full((height, width), np.inf, dtype=np.float64)
        previous_y = np.full((height, width), -1, dtype=np.int32)
        previous_x = np.full((height, width), -1, dtype=np.int32)
        start_x, start_y = start
        distances[start_y, start_x] = 0.0

        heap: list[tuple[float, int, int]] = [(0.0, start_y, start_x)]
        steps = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, sqrt(2.0)),
            (-1, 1, sqrt(2.0)),
            (1, -1, sqrt(2.0)),
            (1, 1, sqrt(2.0)),
        )

        while heap:
            distance, y, x = heappop(heap)
            if distance > distances[y, x]:
                continue
            for dx, dy, cost in steps:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height or not graph_mask[ny, nx]:
                    continue
                next_distance = distance + cost
                if next_distance < distances[ny, nx]:
                    distances[ny, nx] = next_distance
                    previous_y[ny, nx] = y
                    previous_x[ny, nx] = x
                    heappush(heap, (next_distance, ny, nx))

        return distances, previous_y, previous_x

    def _reconstruct_route(
        self, previous_y: np.ndarray, previous_x: np.ndarray, end: Point
    ) -> list[Point]:
        x, y = end
        route: list[Point] = [(int(x), int(y))]
        while previous_y[y, x] >= 0 and previous_x[y, x] >= 0:
            next_y = int(previous_y[y, x])
            next_x = int(previous_x[y, x])
            x, y = next_x, next_y
            route.append((int(x), int(y)))
        route.reverse()
        return route

    def _make_drag_waypoints(self, ball_center: Point, route: list[Point]) -> list[Point]:
        simplified = self._sample_route(route, spacing=max(self.waypoint_spacing, 1))
        waypoints = [ball_center]
        waypoints.extend(point for point in simplified if self._distance(point, ball_center) > 1.0)

        if waypoints[-1] != route[-1]:
            waypoints.append(route[-1])

        waypoints = self._dedupe_points(waypoints)
        if len(waypoints) > self.max_waypoints:
            waypoints = self._limit_waypoints(waypoints, self.max_waypoints)
        return waypoints

    def _sample_route(self, route: list[Point], spacing: int) -> list[Point]:
        if len(route) <= 2:
            return route[:]

        sampled = [route[0]]
        distance_since_last = 0.0
        previous = route[0]
        for point in route[1:]:
            distance_since_last += self._distance(previous, point)
            if distance_since_last >= spacing:
                sampled.append(point)
                distance_since_last = 0.0
            previous = point

        if sampled[-1] != route[-1]:
            sampled.append(route[-1])
        return sampled

    def _limit_waypoints(self, waypoints: list[Point], limit: int) -> list[Point]:
        if limit < 2 or len(waypoints) <= limit:
            return waypoints
        indices = np.linspace(0, len(waypoints) - 1, limit)
        limited = [waypoints[int(round(index))] for index in indices]
        return self._dedupe_points(limited)

    def _prune_to_reachable_component(self, mask: np.ndarray, start: Point) -> np.ndarray:
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 2:
            return mask
        start_label = self._nearest_label(labels, start)
        if start_label <= 0:
            return mask
        return labels == start_label

    @staticmethod
    def _as_bgr_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("expected a BGR or BGRA image with shape (height, width, 3|4)")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _clean_mask(mask: np.ndarray, *, open_size: int, close_size: int) -> np.ndarray:
        opened = mask
        if open_size > 1:
            opened = cv2.morphologyEx(
                opened,
                cv2.MORPH_OPEN,
                np.ones((open_size, open_size), dtype=np.uint8),
            )
        if close_size > 1:
            opened = cv2.morphologyEx(
                opened,
                cv2.MORPH_CLOSE,
                np.ones((close_size, close_size), dtype=np.uint8),
            )
        return opened

    @staticmethod
    def _mask_bounds(mask: np.ndarray, *, padding: int = 0) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0, 0, 0, 0
        height, width = mask.shape
        x_min = max(int(xs.min()) - padding, 0)
        y_min = max(int(ys.min()) - padding, 0)
        x_max = min(int(xs.max()) + padding, width - 1)
        y_max = min(int(ys.max()) + padding, height - 1)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _nearest_true_point(mask: np.ndarray, point: Point) -> Point | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        px, py = point
        distances = (xs - px) ** 2 + (ys - py) ** 2
        index = int(np.argmin(distances))
        return int(xs[index]), int(ys[index])

    @staticmethod
    def _choose_endpoint(candidates: np.ndarray, start: Point) -> Point:
        if len(candidates) == 1:
            y, x = candidates[0]
            return int(x), int(y)
        start_x, start_y = start
        distances = (candidates[:, 1] - start_x) ** 2 + (candidates[:, 0] - start_y) ** 2
        index = int(np.argmax(distances))
        y, x = candidates[index]
        return int(x), int(y)

    @staticmethod
    def _component_distance(labels: np.ndarray, label: int, point: Point) -> float:
        ys, xs = np.nonzero(labels == label)
        if len(xs) == 0:
            return float("inf")
        px, py = point
        distances = (xs - px) ** 2 + (ys - py) ** 2
        return float(sqrt(float(np.min(distances))))

    @staticmethod
    def _nearest_label(labels: np.ndarray, point: Point) -> int:
        mask = labels > 0
        nearest = BallTaskDetector._nearest_true_point(mask, point)
        if nearest is None:
            return 0
        x, y = nearest
        return int(labels[y, x])

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    @classmethod
    def _polyline_length(cls, points: Iterable[Point]) -> float:
        total = 0.0
        previous: Point | None = None
        for point in points:
            if previous is not None:
                total += cls._distance(previous, point)
            previous = point
        return total

    @staticmethod
    def _dedupe_points(points: list[Point]) -> list[Point]:
        deduped: list[Point] = []
        for point in points:
            if not deduped or deduped[-1] != point:
                deduped.append(point)
        return deduped
