import math

import numpy as np

from florr_bot.navigation import (
    HSVRange,
    MinimapNavigator,
    find_nearest_target_or_hint,
    find_target_centroids,
    mask_from_hsv_ranges,
    mask_to_grid,
    movement_intent_from_path,
    plan_path_from_mask,
)
from florr_bot.pathfinding import (
    GridPathfinder,
    line_of_sight,
    simplify_path,
    simplify_path_with_line_of_sight,
)


def test_grid_pathfinder_routes_around_blocked_cells() -> None:
    grid = np.ones((5, 5), dtype=bool)
    grid[2, :] = False
    grid[2, 3] = True

    path = GridPathfinder(grid, allow_diagonal=False).find_path((0, 0), (4, 4))

    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert (3, 2) in path
    assert all(grid[y, x] for x, y in path)


def test_grid_pathfinder_returns_empty_path_when_unreachable() -> None:
    grid = np.ones((3, 3), dtype=bool)
    grid[1, :] = False

    assert GridPathfinder(grid, allow_diagonal=False).find_path((0, 0), (2, 2)) == []


def test_simplify_path_removes_collinear_points() -> None:
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

    assert simplify_path(path) == [(0, 0), (2, 0), (2, 2)]


def test_line_of_sight_simplification_respects_blocked_cells() -> None:
    grid = np.ones((5, 5), dtype=bool)
    grid[1, 1] = False
    path = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 3)]

    assert not line_of_sight(grid, (0, 0), (2, 2))
    assert simplify_path_with_line_of_sight(path, grid) == [(0, 0), (0, 1), (2, 3), (3, 3)]


def test_mask_from_hsv_ranges_supports_wrapped_hue_ranges() -> None:
    hsv = np.zeros((1, 4, 3), dtype=np.uint8)
    hsv[0, 0] = (179, 200, 200)
    hsv[0, 1] = (1, 200, 200)
    hsv[0, 2] = (90, 200, 200)
    hsv[0, 3] = (1, 20, 200)

    mask = mask_from_hsv_ranges(hsv, [HSVRange((170, 100, 100), (10, 255, 255))])

    assert mask.tolist() == [[True, True, False, False]]


def test_find_nearest_target_centroid_prefers_mask_then_hint() -> None:
    target_mask = np.zeros((10, 10), dtype=bool)
    target_mask[1:3, 1:3] = True
    target_mask[7:9, 7:9] = True

    centroids = find_target_centroids(target_mask)
    nearest = find_nearest_target_or_hint(target_mask, origin=(9, 9), hints=[(0, 9)])
    fallback = find_nearest_target_or_hint(np.zeros((10, 10), dtype=bool), origin=(9, 9), hints=[(0, 9), (8, 8)])

    assert [centroid.point for centroid in centroids] == [(2, 2), (8, 8)]
    assert nearest is not None
    assert nearest.kind == "centroid"
    assert nearest.point == (8, 8)
    assert fallback is not None
    assert fallback.kind == "hint"
    assert fallback.point == (8, 8)


def test_mask_to_grid_uses_walkable_fraction_per_cell() -> None:
    mask = np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
        ],
        dtype=bool,
    )

    grid = mask_to_grid(mask, cell_size=2, min_fraction=0.5)

    assert grid.tolist() == [[True, False], [True, False]]


def test_plan_path_from_mask_returns_pixel_waypoints() -> None:
    walkable = np.ones((6, 6), dtype=bool)
    walkable[2, :] = False
    walkable[2, 4] = True

    plan = plan_path_from_mask(
        walkable,
        start=(0, 0),
        goal=(5, 5),
        cell_size=1,
        allow_diagonal=False,
        simplify=False,
    )

    assert plan.grid_start == (0, 0)
    assert plan.grid_goal == (5, 5)
    assert plan.pixel_path[0] == (0, 0)
    assert plan.pixel_path[-1] == (5, 5)
    assert (4, 2) in plan.grid_path


def test_movement_intent_from_path_is_normalized() -> None:
    intent = movement_intent_from_path((0, 0), [(0, 0), (3, 4)], deadzone=0.5)

    assert intent.as_tuple() == (0.6, 0.8)
    assert intent.distance == 5.0
    assert math.isclose(intent.magnitude, 1.0)


def test_minimap_navigator_builds_masks_targets_paths_and_intent() -> None:
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:, :] = (60, 30, 180)
    hsv[8, 8] = (0, 220, 220)

    navigator = MinimapNavigator(
        walkable_ranges=[HSVRange((0, 0, 0), (179, 255, 255))],
        target_ranges=[HSVRange((170, 100, 100), (10, 255, 255))],
        cell_size=1,
    )

    plan = navigator.plan(hsv, origin=(1, 1), input_space="hsv", deadzone=0.5)

    assert plan.target is not None
    assert plan.target.point == (8, 8)
    assert plan.path is not None
    assert plan.path.pixel_path[0] == (1, 1)
    assert plan.path.pixel_path[-1] == (8, 8)
    assert math.isclose(plan.movement.x, 1 / math.sqrt(2))
    assert math.isclose(plan.movement.y, 1 / math.sqrt(2))
