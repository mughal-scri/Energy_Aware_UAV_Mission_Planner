"""Bounded minimax and Alpha-Beta evasion search on the grid."""

from __future__ import annotations

import math

import numpy as np

from uavplanner.config import Config
from uavplanner.models import Cell
from uavplanner.world import chebyshev, neighbors


def choose_move(
    drone_pos: Cell,
    interceptor_pos: Cell,
    waypoint: Cell,
    blocked: np.ndarray,
    depth: int | None = None,
    window_r: int | None = None,
    use_alpha_beta: bool = True,
    cfg: Config | None = None,
) -> tuple[Cell, float, int]:
    """Choose the drone's next cell with alternating minimax search.

    The drone maximizes waypoint progress and distance from capture; the
    interceptor minimizes that value. ``depth`` counts individual moves by
    either player. The third result is the number of evaluated search nodes.
    """
    if cfg is None:
        cfg = Config()
    if depth is None:
        depth = cfg.minimax_depth
    if window_r is None:
        window_r = cfg.minimax_window_r
    _validate(drone_pos, interceptor_pos, waypoint, blocked, depth, window_r, cfg)
    grid = np.asarray(blocked, dtype=bool)
    terminal = cfg.minimax_terminal_value
    if chebyshev(drone_pos, interceptor_pos) <= cfg.interceptor_capture:
        return drone_pos, -terminal, 1
    if drone_pos == waypoint:
        return drone_pos, terminal, 1
    if depth == 0:
        return drone_pos, _evaluate(drone_pos, interceptor_pos, waypoint, cfg), 1

    root = drone_pos
    root_moves = _moves(drone_pos, grid, root, window_r)
    if not root_moves:
        return drone_pos, _evaluate(drone_pos, interceptor_pos, waypoint, cfg), 1

    nodes = 1
    best_move = root_moves[0]
    best_value = -math.inf
    alpha, beta = -math.inf, math.inf
    for move in root_moves:
        value, visited = _search(
            move,
            interceptor_pos,
            waypoint,
            grid,
            root,
            depth - 1,
            maximizing=False,
            window_r=window_r,
            alpha=alpha,
            beta=beta,
            prune=use_alpha_beta,
            cfg=cfg,
        )
        nodes += visited
        if value > best_value:
            best_value, best_move = value, move
        if use_alpha_beta:
            alpha = max(alpha, best_value)
    return best_move, float(best_value), nodes


def _search(
    drone: Cell,
    interceptor: Cell,
    waypoint: Cell,
    blocked: np.ndarray,
    root: Cell,
    depth: int,
    maximizing: bool,
    window_r: int,
    alpha: float,
    beta: float,
    prune: bool,
    cfg: Config,
) -> tuple[float, int]:
    nodes = 1
    if chebyshev(drone, interceptor) <= cfg.interceptor_capture:
        return -cfg.minimax_terminal_value, nodes
    if drone == waypoint:
        return cfg.minimax_terminal_value, nodes
    if depth == 0:
        return _evaluate(drone, interceptor, waypoint, cfg), nodes

    current = drone if maximizing else interceptor
    moves = _moves(current, blocked, root, window_r)
    if not moves:
        return _evaluate(drone, interceptor, waypoint, cfg), nodes

    best = -math.inf if maximizing else math.inf
    for move in moves:
        if maximizing:
            child_drone, child_interceptor = move, interceptor
        else:
            child_drone, child_interceptor = drone, move
        value, child_nodes = _search(
            child_drone,
            child_interceptor,
            waypoint,
            blocked,
            root,
            depth - 1,
            maximizing=not maximizing,
            window_r=window_r,
            alpha=alpha,
            beta=beta,
            prune=prune,
            cfg=cfg,
        )
        nodes += child_nodes
        if maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if prune and alpha >= beta:
            break
    return float(best), nodes


def _moves(cell: Cell, blocked: np.ndarray, root: Cell, window_r: int) -> list[Cell]:
    size = int(blocked.shape[0])
    return [
        neighbour
        for neighbour, _distance in neighbors(cell, size)
        if not blocked[neighbour]
        and abs(neighbour[0] - root[0]) <= window_r
        and abs(neighbour[1] - root[1]) <= window_r
    ]


def _evaluate(drone: Cell, interceptor: Cell, waypoint: Cell, cfg: Config) -> float:
    goal_distance = chebyshev(drone, waypoint)
    threat_distance = min(chebyshev(drone, interceptor), cfg.minimax_cap)
    return cfg.minimax_w_goal * (-goal_distance) + cfg.minimax_w_threat * threat_distance


def _validate(
    drone: Cell,
    interceptor: Cell,
    waypoint: Cell,
    blocked: np.ndarray,
    depth: int,
    window_r: int,
    cfg: Config,
) -> None:
    mask = np.asarray(blocked)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("blocked must be a square 2-D mask")
    if depth < 0 or window_r < 0:
        raise ValueError("depth and window_r must be non-negative")
    if cfg.interceptor_capture < 0 or cfg.minimax_cap < 0:
        raise ValueError("capture radius and evaluation cap must be non-negative")
    n = int(mask.shape[0])
    if any(not (0 <= row < n and 0 <= col < n) for row, col in (drone, interceptor, waypoint)):
        raise ValueError("drone, interceptor, and waypoint must be inside the grid")
