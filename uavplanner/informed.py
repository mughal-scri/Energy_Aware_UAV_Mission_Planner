"""Informed Search: Greedy best-first, energy-aware A*, and leg costs."""

from __future__ import annotations

import heapq
import math

import numpy as np 

from uavplanner.config import Config
from uavplanner.energy import e_min, headwind, move_energy, path_energy
from uavplanner.models import Cell, EnergyCoef, SearchResult, World
from uavplanner.world import neighbors, reconstruct_path

def euclid(a: Cell, b: Cell) -> float: 
    """Straight distance in cells"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def greedy(
    world: World, 
    blocked: np.ndarray,
    start: Cell,
    goal: Cell,
    coef: EnergyCoef,
    payload: float,
    floor: float | None = None,
    diag_step: float | None = None,
    ) -> SearchResult: 
    """greedy best-first search orderd by Euclidean distance to the goal."""
    if floor is None: 
        floor = Config.energy_floor
    if diag_step is None: 
        diag_step = Config.diag_step
    if start == goal: 
        return SearchResult(path=[], cost=0.0, expanded=1, explored=[start])
    n = int(blocked.shape[0])
    if blocked[start] or blocked[goal]: 
        return SearchResult(path=[], cost=math.inf, expanded=0, explored=[])
    
    heap: list[tuple[float, int, Cell]] = [(euclid(start, goal), 0, start)]
    parent: dict[Cell, Cell | None] = {start: None}
    explored: list[Cell] = []
    tie = 1
    found = False
    while heap: 
        _h, _t, cell = heapq.heappop(heap)
        explored.append(cell)
        if cell == goal: 
            found = True
            break 
        for nb, _length in neighbors(cell, n, diag_step=diag_step):
            if blocked[nb] or nb in parent: 
                continue
            parent[nb] = cell
            heapq.heappush(heap, (euclid(nb, goal), tie, nb))
            tie += 1
    if not found: 
        return SearchResult(path=[], cost=math.inf, expanded=len(explored), explored=explored)
    path = reconstruct_path(parent, goal)
    cost = path_energy(world, path, coef, payload, floor=floor, diag_step=diag_step)
    return SearchResult(path=path, cost=cost, expanded=len(explored), explored=explored)

def astar(
    world: World,
    blocked: np.ndarray,
    start: Cell,
    goal: Cell,
    coef: EnergyCoef,
    payload: float,
    max_wind: float,
    floor: float | None = None,
    diag_step: float | None = None,
) -> SearchResult:
    """Find a minimum-energy path using an admissible energy lower bound.

    ``max_wind`` must bound the magnitude of any headwind that can occur in
    the world. The heuristic is the least possible energy per cell times the
    straight-line distance to the goal.
    """
    if floor is None:
        floor = Config.energy_floor
    if diag_step is None:
        diag_step = Config.diag_step
    if start == goal:
        return SearchResult(path=[], cost=0.0, expanded=1, explored=[start])
    n = int(blocked.shape[0])
    if blocked[start] or blocked[goal]:
        return SearchResult(path=[], cost=math.inf, expanded=0, explored=[])

    lower_per_cell = e_min(coef, max_wind, floor=floor)

    def heuristic(cell: Cell) -> float:
        return lower_per_cell * euclid(cell, goal)

    heap: list[tuple[float, int, float, Cell]] = [(heuristic(start), 0, 0.0, start)]
    best_g: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell | None] = {start: None}
    explored: list[Cell] = []
    tie = 1

    while heap:
        _f, _tie, g, cell = heapq.heappop(heap)
        if g > best_g.get(cell, math.inf):
            continue
        explored.append(cell)
        if cell == goal:
            path = reconstruct_path(parent, goal)
            return SearchResult(path=path, cost=g, expanded=len(explored), explored=explored)
        for nb, length in neighbors(cell, n, diag_step=diag_step):
            if blocked[nb]:
                continue
            climb = float(world.terrain[nb] - world.terrain[cell])
            hw = headwind(world.wind, (nb[0] - cell[0], nb[1] - cell[1]))
            step_cost = move_energy(coef, length, climb, hw, payload, floor=floor)
            new_g = g + step_cost
            if new_g >= best_g.get(nb, math.inf):
                continue
            best_g[nb] = new_g
            parent[nb] = cell
            heapq.heappush(heap, (new_g + heuristic(nb), tie, new_g, nb))
            tie += 1
    return SearchResult(path=[], cost=math.inf, expanded=len(explored), explored=explored)

def cost_matrix(
    world: World,
    blocked: np.ndarray,
    coef: EnergyCoef,
    stops: list[Cell],
    payload: float,
    max_wind: float,
) -> tuple[np.ndarray, dict[tuple[int, int], list[Cell]]]:
    """Return directed A* energy costs and paths between every pair of stops.
    
    The matrix includes diagonal entries of zeros. For an unreachable directed
    leg, the corresponding matrix entry is ``np.inf`` and its path is empty.
    """
    count = len(stops)
    matrix = np.full((count, count), np.inf, dtype=float)
    paths: dict[tuple[int, int], list[Cell]] = {}
    for i, start in enumerate(stops):
        matrix[i,i] = 0.0
        paths[(i,i)] = []
        for j, goal in enumerate(stops): 
            if i == j: 
                continue
            result = astar(world, blocked, start, goal, coef, payload, max_wind)
            matrix[i,j] = result.cost 
            paths[(i,j)] = result.path
    return matrix, paths    