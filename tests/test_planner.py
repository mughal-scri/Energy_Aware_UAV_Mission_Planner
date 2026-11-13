"""Planning-pipeline tests."""

import numpy as np

from uavplanner.config import Config
from uavplanner.models import Drone, EnergyCoef, Target, World
from uavplanner.planner import format_plan, plan_mission
from uavplanner.world import make_world, neighbors


def _flat_world(n: int = 8, wind: tuple[float, float] = (0.0, 0.0)) -> World:
    return World(
        terrain=np.zeros((n, n)),
        no_fly=np.zeros((n, n), dtype=bool),
        wind=wind,
        base=(1, 1),
    )


def _cfg(**overrides) -> Config:
    values = dict(hc_restarts=2, ga_pop=8, ga_gens=5, ga_elite=1, csp_samples=20)
    values.update(overrides)
    return Config(**values)


def test_storm_preset_is_no_go() -> None:
    cfg = _cfg(grid_n=12, wind_preset="storm", n_hills_min=2, n_hills_max=2)
    rng = np.random.default_rng(7)
    world = make_world(cfg, rng)
    targets = [Target(id=1, pos=(2, 2), mass_kg=1.0)]
    drones = [Drone(id=1, pos=world.base)]
    plan = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(1))
    text = format_plan(plan)
    assert plan.verdict == "NO-GO"
    assert plan.assignment == {}
    assert any("wind above safe limit" in reason for reason in plan.reasons)
    assert "NO-GO" in text
    assert "wind above safe limit" in text


def test_open_mission_returns_routes_and_leaves_a_drone_parked() -> None:
    world = _flat_world()
    cfg = _cfg()
    drones = [
        Drone(id=1, battery=150.0, capacity_kg=3.0, pos=world.base),
        Drone(id=2, battery=150.0, capacity_kg=0.4, pos=world.base),
    ]
    targets = [
        Target(id=1, pos=(1, 5), mass_kg=1.0),
        Target(id=2, pos=(5, 1), mass_kg=1.0),
    ]
    plan = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(2))
    assert plan.verdict == "GO"
    assert plan.rejected == {}
    assert set(plan.assignment) == {1, 2}
    assert set(plan.assignment.values()) == {1}
    assert plan.order[2] == []
    assert plan.routes[2] == []
    assert plan.est_energy[2] == 0.0
    route = plan.routes[1]
    assert route[0] == world.base
    assert route[-1] == world.base
    assert {targets[0].pos, targets[1].pos} <= set(route)
    _assert_route_is_connected(route)
    assert plan.est_energy[1] <= drones[0].battery * (1.0 - cfg.reserve)
    assert plan.order_costs[1]["hill_climb"] <= plan.order_costs[1]["genetic"] or (
        plan.est_energy[1] == min(plan.order_costs[1].values())
    )


def test_restricted_target_is_rejected_and_the_other_is_planned() -> None:
    world = _flat_world()
    world.no_fly[4, 4] = True
    cfg = _cfg(restricted_buffer=0)
    drones = [Drone(id=1, pos=world.base)]
    targets = [
        Target(id=1, pos=(1, 4), mass_kg=1.0),
        Target(id=2, pos=(4, 4), mass_kg=1.0),
    ]
    plan = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(3))
    assert plan.verdict == "GO"
    assert plan.assignment == {1: 1}
    assert "R2" in plan.rejected[2]
    assert any("R2" in reason for reason in plan.reasons)


def test_heavy_target_is_dropped_without_cancelling_the_mission() -> None:
    world = _flat_world()
    cfg = _cfg()
    drones = [Drone(id=1, capacity_kg=2.0, pos=world.base)]
    targets = [
        Target(id=1, pos=(1, 3), mass_kg=1.0),
        Target(id=2, pos=(6, 6), mass_kg=5.0),
    ]
    plan = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(4))
    assert plan.verdict == "GO"
    assert plan.assignment == {1: 1}
    assert "fits no drone" in plan.rejected[2]


def test_same_seed_reproduces_the_plan() -> None:
    world = _flat_world()
    cfg = _cfg()
    drones = [Drone(id=1, pos=world.base), Drone(id=2, pos=world.base)]
    targets = [
        Target(id=1, pos=(2, 6), mass_kg=0.8),
        Target(id=2, pos=(6, 2), mass_kg=0.9),
        Target(id=3, pos=(6, 6), mass_kg=0.7),
    ]
    first = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(11))
    second = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(11))
    assert first == second


def test_detour_above_the_battery_limit_drops_the_farthest_target() -> None:
    n = 11
    no_fly = np.zeros((n, n), dtype=bool)
    no_fly[1:, 5] = True
    world = World(
        terrain=np.zeros((n, n)),
        no_fly=no_fly,
        wind=(0.0, 0.0),
        base=(5, 0),
    )
    cfg = _cfg(restricted_buffer=0, battery=37.5)
    drones = [Drone(id=1, battery=37.5, capacity_kg=3.0, pos=world.base)]
    targets = [
        Target(id=1, pos=(5, 2), mass_kg=0.5),
        Target(id=2, pos=(5, 10), mass_kg=0.5),
    ]
    plan = plan_mission(world, drones, targets, EnergyCoef(), cfg, np.random.default_rng(5))
    assert plan.verdict == "GO"
    assert plan.assignment == {1: 1}
    assert 2 in plan.rejected
    assert plan.routes[1][0] == world.base
    assert plan.routes[1][-1] == world.base
    assert targets[0].pos in plan.routes[1]
    assert plan.est_energy[1] <= 37.5 * 0.8 + 1e-6


def test_airborne_replan_starts_at_the_drone_and_returns_home() -> None:
    world = _flat_world()
    cfg = _cfg()
    drone = Drone(id=1, pos=(2, 2), mode="TRANSIT")
    target = Target(id=7, pos=(6, 6), mass_kg=1.0)
    blocked = np.zeros(world.terrain.shape, dtype=bool)
    blocked[3, 3] = True
    plan = plan_mission(
        world,
        [drone],
        [target],
        EnergyCoef(),
        cfg,
        np.random.default_rng(6),
        blocked_extra=blocked,
    )
    assert plan.verdict == "GO"
    assert plan.order == {1: [7]}
    assert plan.routes[1][0] == (2, 2)
    assert plan.routes[1][-1] == world.base
    assert target.pos in plan.routes[1]
    assert (3, 3) not in plan.routes[1]


def test_no_remaining_targets_is_no_go() -> None:
    world = _flat_world()
    plan = plan_mission(
        world,
        [Drone(id=1, pos=world.base)],
        [],
        EnergyCoef(),
        _cfg(),
        np.random.default_rng(8),
    )
    assert plan.verdict == "NO-GO"
    assert any("R7" in reason for reason in plan.reasons)


def _assert_route_is_connected(route: list[tuple[int, int]]) -> None:
    assert len(route) >= 2
    n = max(max(cell) for cell in route) + 1
    for left, right in zip(route, route[1:]):
        assert right in {cell for cell, _distance in neighbors(left, n)}
