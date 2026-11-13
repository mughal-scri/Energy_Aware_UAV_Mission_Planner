"""Tick-loop integration checks without interceptor behaviour."""

import numpy as np

from uavplanner.config import Config
from uavplanner.console import Console
from uavplanner.models import Drone, EnergyCoef, Interceptor, PerceptRow, Target, World
from uavplanner.planner import Plan
from uavplanner.simulator import run_mission


def _scenario(seed: int = 1):
    world = World(np.zeros((7, 7)), np.zeros((7, 7), dtype=bool), (0.0, 0.0), (3, 3))
    drones = [
        Drone(id=1, battery=100.0, capacity_kg=3.0, pos=world.base),
        Drone(id=2, battery=100.0, capacity_kg=3.0, pos=world.base),
    ]
    targets = [Target(id=1, pos=(2, 3), mass_kg=0.5), Target(id=2, pos=(4, 3), mass_kg=0.5)]
    plan = Plan(
        "GO", [], {}, {1: 1, 2: 2}, {1: [1], 2: [2]},
        {1: [world.base, (2, 3), world.base], 2: [world.base, (4, 3), world.base]},
        {1: 3.0, 2: 3.0},
    )
    cfg = Config(max_ticks=50, takeoff_stagger=2, noise_sd=0.0)
    return world, plan, drones, targets, cfg, np.random.default_rng(seed)


def test_mission_confirms_target_returns_and_logs_percepts() -> None:
    world, plan, drones, targets, cfg, rng = _scenario()
    # One drone keeps this targeted check independent of takeoff staggering.
    drones = drones[:1]
    drones[0].pos = world.base
    plan.assignment = {1: 1}
    plan.order = {1: [1]}
    plan.routes = {1: [world.base, (2, 3), world.base]}
    console = Console()
    percepts: list[PerceptRow] = []
    summary = run_mission(world, plan, drones, None, targets[:1], EnergyCoef(), EnergyCoef(), cfg,
                          rng, console, percepts)
    assert summary["missions_completed"] == 1
    assert summary["targets_confirmed"] == 1
    assert summary["collisions"] == 0
    assert summary["percepts"] == 2
    assert targets[0].status == "confirmed"
    assert any("confirmed" in row.event for row in console.rows())


def test_storm_no_go_does_not_consume_energy() -> None:
    world, plan, drones, targets, cfg, rng = _scenario()
    plan.verdict = "NO-GO"
    console = Console()
    summary = run_mission(world, plan, drones, None, targets, EnergyCoef(), EnergyCoef(), cfg,
                          rng, console, [])
    assert summary["missions_completed"] == 0
    assert summary["total_energy"] == 0.0
    assert all(drone.battery == 100.0 for drone in drones)
    assert all(row.event == "NO-GO" for row in console.rows())


def test_twenty_seeded_two_drone_missions_have_no_collisions() -> None:
    for seed in range(20):
        world, plan, drones, targets, cfg, rng = _scenario(seed)
        console = Console()
        summary = run_mission(world, plan, drones, None, targets, EnergyCoef(), EnergyCoef(), cfg,
                              rng, console, [])
        assert summary["collisions"] == 0
        assert summary["drones_lost"] == 0


def test_three_sensor_misses_mark_target_unconfirmed_and_continue() -> None:
    class AlwaysMiss:
        @staticmethod
        def random() -> float:
            return 0.99

        @staticmethod
        def normal(_mean: float, _sd: float) -> float:
            return 0.0

    world, plan, drones, targets, cfg, _rng = _scenario()
    drones = drones[:1]
    plan.assignment = {1: 1}
    plan.order = {1: [1]}
    plan.routes = {1: [world.base, (2, 3), world.base]}
    cfg.takeoff_stagger = 0
    console = Console()
    summary = run_mission(world, plan, drones, None, targets[:1], EnergyCoef(), EnergyCoef(), cfg,
                          AlwaysMiss(), console, [])
    assert summary["targets_unconfirmed"] == 1
    assert targets[0].status == "unconfirmed"
    assert summary["missions_completed"] == 1
    assert any("unconfirmed" in row.event for row in console.rows())


def test_r6_invokes_evasion_and_clear_threat_replans_with_astar() -> None:
    world = World(np.zeros((21, 21)), np.zeros((21, 21), dtype=bool), (0.0, 0.0), (10, 10))
    target = Target(id=1, pos=(10, 15), mass_kg=0.5)
    drone = Drone(id=1, battery=100.0, capacity_kg=3.0, pos=world.base)
    route = [(10, col) for col in range(10, 16)] + [(10, col) for col in range(14, 9, -1)]
    plan = Plan("GO", [], {}, {1: 1}, {1: [1]}, {1: route}, {1: 10.0})
    interceptor = Interceptor(pos=(10, 5), waypoints=[(0, 0)])
    cfg = Config(grid_n=21, max_ticks=60, takeoff_stagger=0, noise_sd=0.0,
                 interceptor_detection=5, minimax_depth=4)
    console = Console()
    summary = run_mission(
        world, plan, [drone], interceptor, [target], EnergyCoef(), EnergyCoef(), cfg,
        np.random.default_rng(2), console, [],
    )
    events = [row.event for row in console.rows()]
    assert summary["minimax_nodes"] > 0
    assert any("R6: EVADE" in event for event in events)
    assert any("R6 clear: A* route replanned" in event for event in events)
    assert summary["missions_completed"] == 1
    assert summary["targets_confirmed"] == 1
    assert summary["drones_lost"] == 0


def test_drone_already_evading_replans_when_threat_is_clear() -> None:
    world, plan, drones, targets, cfg, rng = _scenario()
    drones = drones[:1]
    drones[0].mode = "EVADE"
    plan.assignment = {1: 1}
    plan.order = {1: [1]}
    plan.routes = {1: [world.base, targets[0].pos, world.base]}
    interceptor = Interceptor(pos=(0, 0), waypoints=[(0, 0)])
    cfg.minimax_release = 2
    cfg.minimax_trigger = 1
    cfg.interceptor_detection = 0
    console = Console()
    summary = run_mission(world, plan, drones, interceptor, targets[:1], EnergyCoef(), EnergyCoef(),
                          cfg, rng, console, [])
    assert any("R6 clear: A* route replanned" in row.event for row in console.rows())
    assert summary["missions_completed"] == 1
    assert summary["drones_lost"] == 0
