"""Uninformed search: BFS, iterative DFS, and uniform-cost search."""

from __future__ import annotations

from collections import deque
import heapq
import math

import numpy as np

from uavplanner.config import Config
from uavplanner.energy import headwind, move_energy
from uavplanner.models import Cell, EnergyCoef, SearchResult, World
from uavplanner.world import neighbors, reconstruct_path


def bfs(blocked: np.ndarray, start: Cell, goal: Cell) -> SearchResult:
    """Breadth-first search. Ignores energy; ``cost`` is the number of steps."""
    if start == goal:
        return SearchResult(path=[], cost=0.0, expanded=1, explored=[start])
    n = int(blocked.shape[0])
    parent: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    explored: list[Cell] = []
    found = False
    while queue:
        cell = queue.popleft()
        explored.append(cell)
        if cell == goal:
            found = True
            break
        for nb, _length in neighbors(cell, n):
            if blocked[nb] or nb in parent:
                continue
            parent[nb] = cell
            queue.append(nb)
    if not found:
        return SearchResult(path=[], cost=math.inf, expanded=len(explored), explored=explored)
    path = reconstruct_path(parent, goal)
    return SearchResult(
        path=path,
        cost=float(len(path) - 1),
        expanded=len(explored),
        explored=explored,
    )


def dfs(blocked: np.ndarray, start: Cell, goal: Cell) -> SearchResult:
    """Depth-first search to a goal using an explicit stack (not recursion)."""
    if start == goal:
        return SearchResult(path=[], cost=0.0, expanded=1, explored=[start])
    n = int(blocked.shape[0])
    parent: dict[Cell, Cell | None] = {start: None}
    stack: list[Cell] = [start]
    explored: list[Cell] = []
    found = False
    while stack:
        cell = stack.pop()
        explored.append(cell)
        if cell == goal:
            found = True
            break
        for nb, _length in neighbors(cell, n):
            if blocked[nb] or nb in parent:
                continue
            parent[nb] = cell
            stack.append(nb)
    if not found:
        return SearchResult(path=[], cost=math.inf, expanded=len(explored), explored=explored)
    path = reconstruct_path(parent, goal)
    return SearchResult(
        path=path,
        cost=float(len(path) - 1),
        expanded=len(explored),
        explored=explored,
    )


def dfs_reachable(blocked: np.ndarray, start: Cell) -> np.ndarray:
    """Flood-fill with an explicit stack. True cells can be reached from ``start``."""
    n = int(blocked.shape[0])
    reach = np.zeros((n, n), dtype=bool)
    if blocked[start]:
        return reach
    stack: list[Cell] = [start]
    reach[start] = True
    while stack:
        cell = stack.pop()
        for nb, _length in neighbors(cell, n):
            if blocked[nb] or reach[nb]:
                continue
            reach[nb] = True
            stack.append(nb)
    return reach


def ucs(
    world: World,
    blocked: np.ndarray,
    start: Cell,
    goal: Cell,
    coef: EnergyCoef,
    payload: float,
    floor: float | None = None,
    diag_step: float | None = None,
) -> SearchResult:
    """Uniform-cost (Dijkstra) search ordered by energy so far."""
    if floor is None:
        floor = Config.energy_floor
    if diag_step is None:
        diag_step = Config.diag_step
    if start == goal:
        return SearchResult(path=[], cost=0.0, expanded=1, explored=[start])
    n = int(blocked.shape[0])
    if blocked[start] or blocked[goal]:
        return SearchResult(path=[], cost=math.inf, expanded=0, explored=[])

    heap: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    best_g: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell | None] = {start: None}
    explored: list[Cell] = []
    tie = 1
    found = False
    while heap:
        g, _t, cell = heapq.heappop(heap)
        if g > best_g.get(cell, math.inf):
            continue
        explored.append(cell)
        if cell == goal:
            found = True
            break
        for nb, length in neighbors(cell, n, diag_step=diag_step):
            if blocked[nb]:
                continue
            climb = float(world.terrain[nb] - world.terrain[cell])
            hw = headwind(world.wind, (nb[0] - cell[0], nb[1] - cell[1]))
            step_cost = move_energy(coef, length, climb, hw, payload, floor=floor)
            ng = g + step_cost
            if ng < best_g.get(nb, math.inf):
                best_g[nb] = ng
                parent[nb] = cell
                heapq.heappush(heap, (ng, tie, nb))
                tie += 1
    if not found:
        return SearchResult(path=[], cost=math.inf, expanded=len(explored), explored=explored)
    path = reconstruct_path(parent, goal)
    return SearchResult(path=path, cost=best_g[goal], expanded=len(explored), explored=explored)