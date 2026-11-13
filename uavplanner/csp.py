"""Constraint-satisfaction assignment of static targets to drones."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from uavplanner.energy import headwind, move_energy
from uavplanner.informed import euclid
from uavplanner.models import Cell, Drone, EnergyCoef, Target, World
from uavplanner.world import neighbors


def avg_cell_cost(
    world: World,
    coef: EnergyCoef,
    rng: np.random.Generator,
    samples: int = 1000,
) -> float:
    """Estimate unloaded energy per traversable move by sampling valid edges.

    The CSP uses this only as a conservative first-pass assignment estimate;
    the later planner verifies actual A*-route tour energy.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    n = int(world.terrain.shape[0])
    edges = [
        (cell, neighbour, distance)
        for row in range(n)
        for col in range(n)
        for cell in [(row, col)]
        if not world.no_fly[cell]
        for neighbour, distance in neighbors(cell, n)
        if not world.no_fly[neighbour]
    ]
    if not edges:
        raise ValueError("world has no traversable moves")
    indices = rng.integers(0, len(edges), size=samples)
    costs = []
    for index in indices:
        start, goal, distance = edges[int(index)]
        climb = float(world.terrain[goal] - world.terrain[start])
        wind = headwind(world.wind, (goal[0] - start[0], goal[1] - start[1]))
        costs.append(move_energy(coef, distance, climb, wind, payload=0.0) / distance)
    return float(np.mean(costs))


def estimate_target_cost(base: Cell, target: Target, average_cell_cost: float) -> float:
    """Estimate a target's out-and-back energy from a mean cell cost."""
    if average_cell_cost < 0:
        raise ValueError("average_cell_cost must be non-negative")
    return 2.0 * euclid(base, target.pos) * average_cell_cost


def csp_assign(
    targets: list[Target],
    drones: list[Drone],
    est_cost: Mapping[int, float],
    energy_scale: float = 1.0,
    reserve: float = 0.2,
    use_mrv: bool = True,
    use_fc: bool = True,
) -> tuple[dict[int, int] | None, dict[str, int]]:
    """Assign every target while respecting drone capacity and energy budgets.

    ``est_cost`` maps a target id to its estimated out-and-back energy. The
    result maps target id to drone id. MRV chooses the target with the fewest
    currently feasible drones; forward checking rejects an assignment as soon
    as it leaves any unassigned target without a feasible drone.
    """
    if not 0.0 <= reserve < 1.0:
        raise ValueError("reserve must be in [0, 1)")
    if energy_scale <= 0.0:
        raise ValueError("energy_scale must be positive")
    if len({target.id for target in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    if len({drone.id for drone in drones}) != len(drones):
        raise ValueError("drone ids must be unique")
    missing = [target.id for target in targets if target.id not in est_cost]
    if missing:
        raise ValueError(f"missing estimated costs for target ids: {missing}")
    if any(est_cost[target.id] < 0.0 for target in targets):
        raise ValueError("estimated costs must be non-negative")

    stats = {"backtracks": 0, "nodes": 0}
    if not targets:
        return {}, stats
    drone_by_id = {drone.id: drone for drone in drones}
    mass_load = {drone.id: 0.0 for drone in drones}
    energy_load = {drone.id: 0.0 for drone in drones}
    assignment: dict[int, int] = {}

    def feasible_drone_ids(target: Target) -> list[int]:
        candidates = []
        for drone_id, drone in drone_by_id.items():
            mass_ok = mass_load[drone_id] + target.mass_kg <= drone.capacity_kg
            energy_limit = drone.battery * (1.0 - reserve) * energy_scale
            energy_ok = energy_load[drone_id] + est_cost[target.id] <= energy_limit
            if mass_ok and energy_ok:
                candidates.append(drone_id)
        return candidates

    def solve() -> bool:
        if len(assignment) == len(targets):
            return True
        unassigned = [target for target in targets if target.id not in assignment]
        if use_mrv:
            target = min(
                unassigned,
                key=lambda item: (len(feasible_drone_ids(item)), item.id),
            )
        else:
            target = unassigned[0]
        candidates = feasible_drone_ids(target)
        if not candidates:
            return False
        for drone_id in candidates:
            assignment[target.id] = drone_id
            mass_load[drone_id] += target.mass_kg
            energy_load[drone_id] += est_cost[target.id]
            stats["nodes"] += 1
            remaining = [item for item in unassigned if item.id != target.id]
            viable = not use_fc or all(feasible_drone_ids(item) for item in remaining)
            if viable and solve():
                return True
            del assignment[target.id]
            mass_load[drone_id] -= target.mass_kg
            energy_load[drone_id] -= est_cost[target.id]
            stats["backtracks"] += 1
        return False

    return (dict(assignment), stats) if solve() else (None, stats)
