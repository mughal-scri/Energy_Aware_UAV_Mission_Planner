"""Test Search Algorithms on the world with config"""

import math

import numpy as np
import pytest

from uavplanner.config import Config
from uavplanner.energy import e_min
from uavplanner.informed import astar, cost_matrix, euclid, greedy
from uavplanner.models import EnergyCoef, World
from uavplanner.uninformed import bfs, dfs_reachable, ucs
from uavplanner.world import chebyshev, make_targets, make_world, neighbors


def test_bfs_steps_equal_chebyshev_on_open_grid() -> None:
    n = 12
    blocked = np.zeros((n, n), dtype=bool)
    start, goal = (1, 2), (8, 6)
    result = bfs(blocked, start, goal)
    assert result.path[0] == start
    assert result.path[-1] == goal
    assert len(result.path) - 1 == chebyshev(start, goal)


def test_enclosed_cell_not_reachable_by_dfs() -> None:
    cfg = Config(seed=11, demo_rejections=True)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    n = world.terrain.shape[0]
    enclosed = [
        t.pos
        for t in targets
        if not world.no_fly[t.pos]
        and all(world.no_fly[nb] for nb, _ in neighbors(t.pos, n))
    ]
    assert enclosed
    reach = dfs_reachable(world.no_fly, world.base)
    assert not reach[enclosed[0]]
    assert reach[world.base]


def test_ucs_and_greedy_find_a_path() -> None:
    cfg = Config(seed=42)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    goal = next(t.pos for t in targets if not world.no_fly[t.pos])
    coef = EnergyCoef()
    ucs_res = ucs(world, world.no_fly, world.base, goal, coef, 1.0)
    greedy_res = greedy(world, world.no_fly, world.base, goal, coef, 1.0)
    astar_res = astar(world, world.no_fly, world.base, goal, coef, 1.0, cfg.wind_limit)
    assert ucs_res.path[0] == world.base and ucs_res.path[-1] == goal
    assert greedy_res.path[0] == world.base and greedy_res.path[-1] == goal
    assert astar_res.path[0] == world.base and astar_res.path[-1] == goal
    assert greedy_res.cost >= ucs_res.cost - 1e-9
    assert greedy_res.cost >= astar_res.cost - 1e-9
    
def test_astar_matches_ucs_and_expands_no_more_on_thirty_pairs() -> None:
    cfg = Config(seed=19)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    free = [tuple(cell) for cell in np.argwhere(~world.no_fly)]
    pairs = rng.choice(len(free), size=(30, 2), replace=True)
    coef = EnergyCoef()
    for start_index, goal_index in pairs:
        start = free[int(start_index)]
        goal = free[int(goal_index)]
        ucs_res = ucs(world, world.no_fly, start, goal, coef, 1.0)
        astar_res = astar(world, world.no_fly, start, goal, coef, 1.0, cfg.wind_limit)
        assert astar_res.cost == pytest.approx(ucs_res.cost, abs=1e-9)
        assert astar_res.expanded <= ucs_res.expanded


def test_astar_heuristic_is_not_above_ucs_remaining_cost() -> None:
    cfg = Config(seed=23)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    free = [tuple(cell) for cell in np.argwhere(~world.no_fly)]
    pairs = rng.choice(len(free), size=(30, 2), replace=True)
    coef = EnergyCoef()
    lower_per_cell = e_min(coef, cfg.wind_limit)
    for start_index, goal_index in pairs:
        start = free[int(start_index)]
        goal = free[int(goal_index)]
        remaining_cost = ucs(world, world.no_fly, start, goal, coef, 1.0).cost
        assert lower_per_cell * euclid(start, goal) <= remaining_cost + 1e-9

def test_cost_matrix_is_directed_and_marks_unreachable_legs() -> None:
    terrain = np.zeros((3, 3), dtype=float)
    terrain[0, 1] = 10.0
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[2, 2] = True
    world = World(terrain=terrain, no_fly=blocked, wind=(0.0, 0.0), base=(0, 0))
    stops = [(0, 0), (0, 1), (2, 2)]
    matrix, paths = cost_matrix(
        world, blocked, EnergyCoef(), stops, payload=0.0, max_wind=8.0
    )
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] > matrix[1, 0]
    assert math.isinf(matrix[0, 2]) and math.isinf(matrix[2, 0])
    assert paths[(0, 1)][0] == stops[0] and paths[(0, 1)][-1] == stops[1]
    assert paths[(0, 2)] == [] and paths[(2, 0)] == []