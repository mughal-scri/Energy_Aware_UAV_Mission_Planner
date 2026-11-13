"""Reactive patrol and chase behavior for the simulated interceptor."""

from __future__ import annotations

from uavplanner.config import Config
from uavplanner.models import Cell, Interceptor
from uavplanner.world import chebyshev


def step_interceptor(
    interceptor: Interceptor,
    drone_cells: list[Cell],
    cfg: Config | None = None,
) -> list[int]:
    """Advance one patrol/chase tick and return captured drone-cell indices.

    The returned indices refer to ``drone_cells`` in input order. This makes
    capture actionable for callers while keeping the interceptor independent
    of mutable ``Drone`` instances.
    """
    if cfg is None:
        cfg = Config()
    _validate(interceptor, drone_cells, cfg)

    nearest_index = _nearest_drone(interceptor.pos, drone_cells)
    in_range = (
        nearest_index is not None
        and chebyshev(interceptor.pos, drone_cells[nearest_index]) <= cfg.interceptor_detection
    )
    if in_range:
        interceptor.mode = "CHASE"
        interceptor.target_last_seen = drone_cells[nearest_index]
        interceptor.lost_ticks = 0
        interceptor.pos = _step_toward(interceptor.pos, drone_cells[nearest_index], cfg)
    elif interceptor.mode == "CHASE":
        interceptor.lost_ticks += 1
        if interceptor.lost_ticks >= cfg.interceptor_lost_ticks:
            interceptor.mode = "PATROL"
            interceptor.target_last_seen = None
            interceptor.lost_ticks = 0
            _patrol(interceptor, cfg)
        elif interceptor.target_last_seen is not None:
            interceptor.pos = _step_toward(interceptor.pos, interceptor.target_last_seen, cfg)
    else:
        interceptor.mode = "PATROL"
        interceptor.target_last_seen = None
        interceptor.lost_ticks = 0
        _patrol(interceptor, cfg)

    interceptor.last_seen = interceptor.pos
    return [
        index
        for index, cell in enumerate(drone_cells)
        if chebyshev(interceptor.pos, cell) <= cfg.interceptor_capture
    ]


def _patrol(interceptor: Interceptor, cfg: Config) -> None:
    if not interceptor.waypoints:
        return
    interceptor.wp_index %= len(interceptor.waypoints)
    waypoint = interceptor.waypoints[interceptor.wp_index]
    if interceptor.pos == waypoint:
        interceptor.wp_index = (interceptor.wp_index + 1) % len(interceptor.waypoints)
        waypoint = interceptor.waypoints[interceptor.wp_index]
    interceptor.pos = _step_toward(interceptor.pos, waypoint, cfg)


def _step_toward(start: Cell, goal: Cell, cfg: Config) -> Cell:
    position = start
    for _ in range(cfg.interceptor_speed):
        dr = (goal[0] > position[0]) - (goal[0] < position[0])
        dc = (goal[1] > position[1]) - (goal[1] < position[1])
        if dr == 0 and dc == 0:
            break
        row = min(max(position[0] + dr, 0), cfg.grid_n - 1)
        col = min(max(position[1] + dc, 0), cfg.grid_n - 1)
        position = row, col
    return position


def _nearest_drone(position: Cell, drone_cells: list[Cell]) -> int | None:
    if not drone_cells:
        return None
    return min(range(len(drone_cells)), key=lambda index: (chebyshev(position, drone_cells[index]), index))


def _validate(interceptor: Interceptor, drone_cells: list[Cell], cfg: Config) -> None:
    if (
        cfg.grid_n < 1
        or cfg.interceptor_detection < 0
        or cfg.interceptor_capture < 0
        or cfg.interceptor_speed < 1
    ):
        raise ValueError("grid and interceptor ranges must be valid")
    if cfg.interceptor_lost_ticks < 1:
        raise ValueError("interceptor_lost_ticks must be positive")
    if interceptor.mode not in {"PATROL", "CHASE"}:
        raise ValueError("interceptor mode must be PATROL or CHASE")
    cells = [interceptor.pos, *interceptor.waypoints, *drone_cells]
    if any(not (0 <= row < cfg.grid_n and 0 <= col < cfg.grid_n) for row, col in cells):
        raise ValueError("interceptor, waypoint, and drone cells must be inside the grid")
