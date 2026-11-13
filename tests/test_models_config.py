"""Smoke tests for the shared data model and config defaults."""

from uavplanner.config import Config
from uavplanner.models import Drone, EnergyCoef, Interceptor, Target


def test_energy_coef_defaults_match_spec() -> None:
    coef = EnergyCoef()
    assert (coef.a, coef.b, coef.c, coef.e) == (1.0, 0.05, 0.08, 0.15)


def test_config_defaults_match() -> None:
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.grid_n == 50
    assert cfg.cell_m == 100.0
    assert cfg.tick_s == 10.0
    assert cfg.n_drones == 3
    assert (cfg.n_targets_min, cfg.n_targets_max) == (6, 8)
    assert cfg.battery == 150.0
    assert cfg.reserve == 0.20
    assert cfg.capacity_kg == 3.0
    assert cfg.true_coef == EnergyCoef()
    assert cfg.energy_floor == 0.3
    assert (cfg.noise_sd, cfg.noise_clip) == (0.05, 0.15)
    assert cfg.hover_energy == 0.2
    assert cfg.wind_presets == {
        "calm": 2.0,
        "moderate": 6.0,
        "strong": 10.0,
        "storm": 14.0,
    }
    assert cfg.wind_limit == 12.0
    assert cfg.restricted_buffer == 1
    assert cfg.max_climb_m == 40.0
    assert cfg.interceptor_zone == 2
    assert (cfg.takeoff_stagger, cfg.replan_after_waits) == (2, 3)
    assert cfg.interceptor_detection == 6
    assert cfg.interceptor_speed == 1
    assert cfg.interceptor_capture == 1
    assert cfg.hc_restarts == 20
    assert (cfg.ga_pop, cfg.ga_gens, cfg.ga_pc, cfg.ga_pm) == (50, 100, 0.8, 0.1)
    assert (cfg.ga_tournament, cfg.ga_elite, cfg.ga_battery_penalty) == (3, 2, 1000.0)
    assert (cfg.minimax_depth, cfg.minimax_window_r) == (4, 4)
    assert (cfg.minimax_trigger, cfg.minimax_release) == (5, 8)
    assert (cfg.minimax_w_goal, cfg.minimax_w_threat, cfg.minimax_cap) == (
        1.0,
        1.5,
        5,
    )
    assert cfg.sensor_range == 3
    assert cfg.sensor_probs == {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
    assert cfg.sensor_max_looks == 3
    assert cfg.learning_lr == 0.01
    assert cfg.learning_spread == 0.4
    assert cfg.learning_scale == (1.0, 20.0, 5.0, 2.0)
    assert cfg.max_ticks == 300


def test_drone_mutable_defaults_are_independent() -> None:
    a = Drone(id=1)
    b = Drone(id=2)
    a.stops.append(7)
    a.route.append((1, 1))
    assert b.stops == []
    assert b.route == []
    assert a.mode == "PARKED"


def test_target_and_interceptor_defaults() -> None:
    target = Target(id=1, pos=(4, 5), mass_kg=1.0)
    threat = Interceptor(pos=(0, 0), waypoints=[(0, 0), (1, 1)])
    assert target.status == "pending"
    assert threat.mode == "PATROL"
    assert threat.last_seen is None
    assert threat.wp_index == 0
