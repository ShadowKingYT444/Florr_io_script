"""Deterministic grid pathfinding utilities for minimap navigation.

Points are expressed as ``(x, y)`` tuples to match image coordinates. Boolean
grids and masks are still indexed as ``grid[y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Iterator, Sequence

import numpy as np

GridPoint = tuple[int, int]


SQRT2 = math.sqrt(2.0)


def _as_bool_grid(walkable: np.ndarray) -> np.ndarray:
    grid = np.asarray(walkable, dtype=bool)
    if grid.ndim != 2:
        raise ValueError("walkable grid must be a 2D boolean array")
    return grid


def _as_point(point: Sequence[int | float]) -> GridPoint:
    if len(point) != 2:
        raise ValueError("points must contain exactly two coordinates")
    return int(point[0]), int(point[1])


@dataclass(frozen=True)
class GridPathfinder:
    """A* pathfinder over a 2D boolean walkable grid.

    ``True`` cells are walkable and ``False`` cells are blocked. Paths include
    both the start and goal cells. An unreachable or invalid request returns an
    empty list.
    """

    walkable: np.ndarray
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "walkable", _as_bool_grid(self.walkable))

    @property
    def width(self) -> int:
        return int(self.walkable.shape[1])

    @property
    def height(self) -> int:
        return int(self.walkable.shape[0])

    def in_bounds(self, point: Sequence[int | float]) -> bool:
        x, y = _as_point(point)
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, point: Sequence[int | float]) -> bool:
        x, y = _as_point(point)
        return self.in_bounds((x, y)) and bool(self.walkable[y, x])

    def neighbors(self, point: Sequence[int | float]) -> Iterator[tuple[GridPoint, float]]:
        x, y = _as_point(point)
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if self.allow_diagonal:
            directions.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)])

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if not self.is_walkable(neighbor):
                continue

            if dx and dy and self.prevent_corner_cutting:
                if not self.is_walkable((x + dx, y)) or not self.is_walkable((x, y + dy)):
                    continue

            yield neighbor, SQRT2 if dx and dy else 1.0

    def heuristic(self, point: Sequence[int | float], goal: Sequence[int | float]) -> float:
        x0, y0 = _as_point(point)
        x1, y1 = _as_point(goal)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if self.allow_diagonal:
            return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)
        return float(dx + dy)

    def find_path(self, start: Sequence[int | float], goal: Sequence[int | float]) -> list[GridPoint]:
        start_point = _as_point(start)
        goal_point = _as_point(goal)
        if not self.is_walkable(start_point) or not self.is_walkable(goal_point):
            return []
        if start_point == goal_point:
            return [start_point]

        frontier: list[tuple[float, float, int, GridPoint]] = []
        counter = 0
        heapq.heappush(frontier, (self.heuristic(start_point, goal_point), 0.0, counter, start_point))

        came_from: dict[GridPoint, GridPoint | None] = {start_point: None}
        cost_so_far: dict[GridPoint, float] = {start_point: 0.0}

        while frontier:
            _, _, _, current = heapq.heappop(frontier)
            if current == goal_point:
                return _reconstruct_path(came_from, current)

            current_cost = cost_so_far[current]
            for neighbor, step_cost in self.neighbors(current):
                new_cost = current_cost + step_cost
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue

                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                counter += 1
                priority = new_cost + self.heuristic(neighbor, goal_point)
                heapq.heappush(frontier, (priority, self.heuristic(neighbor, goal_point), counter, neighbor))

        return []


def _reconstruct_path(came_from: dict[GridPoint, GridPoint | None], end: GridPoint) -> list[GridPoint]:
    path = [end]
    current = end
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def simplify_path(path: Iterable[Sequence[int | float]]) -> list[GridPoint]:
    """Remove redundant collinear interior points from a path."""

    points = [_as_point(point) for point in path]
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for previous, current, following in zip(points, points[1:], points[2:]):
        ax = current[0] - previous[0]
        ay = current[1] - previous[1]
        bx = following[0] - current[0]
        by = following[1] - current[1]
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        if cross != 0 or dot < 0:
            simplified.append(current)
    simplified.append(points[-1])
    return simplified


def line_of_sight(walkable: np.ndarray, start: Sequence[int | float], end: Sequence[int | float]) -> bool:
    """Return whether a straight grid segment crosses only walkable cells."""

    grid = _as_bool_grid(walkable)
    for x, y in _supercover_cells(_as_point(start), _as_point(end)):
        if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]:
            return False
        if not bool(grid[y, x]):
            return False
    return True


def simplify_path_with_line_of_sight(
    path: Iterable[Sequence[int | float]],
    walkable: np.ndarray,
) -> list[GridPoint]:
    """Greedily shortcut a path while keeping every segment on walkable cells."""

    points = simplify_path(path)
    if len(points) <= 2:
        return points

    result = [points[0]]
    anchor_index = 0
    while anchor_index < len(points) - 1:
        next_index = len(points) - 1
        while next_index > anchor_index + 1:
            if line_of_sight(walkable, points[anchor_index], points[next_index]):
                break
            next_index -= 1

        result.append(points[next_index])
        anchor_index = next_index

    return result


def nearest_walkable_point(
    walkable: np.ndarray,
    point: Sequence[int | float],
    max_distance: float | None = None,
) -> GridPoint | None:
    """Find the closest walkable cell to ``point`` using deterministic tie breaks."""

    grid = _as_bool_grid(walkable)
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return None

    x, y = _as_point(point)
    dx = xs.astype(np.int64) - int(x)
    dy = ys.astype(np.int64) - int(y)
    distances = dx * dx + dy * dy
    order = np.lexsort((xs, ys, distances))
    best_index = int(order[0])
    if max_distance is not None and distances[best_index] > max_distance * max_distance:
        return None
    return int(xs[best_index]), int(ys[best_index])


def _supercover_cells(start: GridPoint, end: GridPoint) -> Iterator[GridPoint]:
    """Yield grid cells touched by a line from ``start`` to ``end``."""

    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    x = x0
    y = y0
    ix = 0
    iy = 0
    yield x, y

    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            x += sign_x
            y += sign_y
            ix += 1
            iy += 1
            if sign_y:
                yield x - sign_x, y
            if sign_x:
                yield x, y - sign_y
        elif decision < 0:
            x += sign_x
            ix += 1
        else:
            y += sign_y
            iy += 1
        yield x, y


__all__ = [
    "GridPathfinder",
    "GridPoint",
    "line_of_sight",
    "nearest_walkable_point",
    "simplify_path",
    "simplify_path_with_line_of_sight",
]
