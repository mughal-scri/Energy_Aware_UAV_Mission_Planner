"""Constraint-Assignment Checks."""

import numpy as np

from uavplanner.csp import avg_cell_cost, csp_assign, estimate_target_cost
from uavplanner.models import Drone, EnergyCoef, Target, World


def test_average_cell_cost_and_out_and_back_estimate() -> None:
    world = World(
        terrain=np.zeros((3, 3)), no_fly=np.zeros((3, 3), dtype=bool),
        wind=(0.0, 0.0), base=(0, 0)
    )
    average = avg_cell_cost(world, EnergyCoef(), np.random.default_rng(1), samples=100)
    target = Target(id=1, pos=(3, 4), mass_kg=1.0)
    assert average == 1.0
    assert estimate_target_cost(world.base, target, average) == 10.0


def test_csp_assignment_satisfies_mass_and_energy_constraints() -> None:
    drones = [
        Drone(id=1, battery=100.0, capacity_kg=3.0),
        Drone(id=2, battery=100.0, capacity_kg=3.0),
    ]
    targets = [
        Target(id=1, pos=(1, 1), mass_kg=1.0),
        Target(id=2, pos=(2, 2), mass_kg=1.5),
        Target(id=3, pos=(3, 3), mass_kg=1.0),
    ]
    costs = {1: 30.0, 2: 35.0, 3: 20.0}
    assignment, stats = csp_assign(targets, drones, costs, reserve=0.2)
    assert assignment is not None
    assert set(assignment) == {1, 2, 3}
    assert stats["nodes"] >= len(targets)
    for drone in drones:
        assigned = [target for target in targets if assignment[target.id] == drone.id]
        assert sum(target.mass_kg for target in assigned) <= drone.capacity_kg
        assert sum(costs[target.id] for target in assigned) <= drone.battery * 0.8


def test_csp_returns_none_for_an_impossible_target() -> None:
    drones = [Drone(id=1, capacity_kg=2.0), Drone(id=2, capacity_kg=2.5)]
    targets = [Target(id=1, pos=(1, 1), mass_kg=3.0)]
    assignment, stats = csp_assign(targets, drones, {1: 10.0})
    assert assignment is None
    assert stats == {"backtracks": 0, "nodes": 0}


def test_mrv_and_forward_checking_need_no_more_backtracks() -> None:
    drones = [
        Drone(id=1, battery=100.0, capacity_kg=3.0),
        Drone(id=2, battery=100.0, capacity_kg=1.0),
    ]
    targets = [
        Target(id=1, pos=(1, 1), mass_kg=1.0),
        Target(id=2, pos=(2, 2), mass_kg=3.0),
    ]
    costs = {1: 1.0, 2: 1.0}
    _plain_assignment, plain = csp_assign(
        targets, drones, costs, use_mrv=False, use_fc=False
    )
    smart_assignment, smart = csp_assign(targets, drones, costs, use_mrv=True, use_fc=True)
    assert smart_assignment is not None
    assert smart["backtracks"] <= plain["backtracks"]
