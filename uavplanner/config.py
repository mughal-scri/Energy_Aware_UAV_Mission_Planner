"""All tunable parameters in one dataclass.

Later modules must read values from ``Config`` instead of hard-coding them.
Defaults match (and the world-generation ranges in
Section 6.1, so ``world.py`` will not introduce magic numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from uavplanner.models import EnergyCoef


def _wind_presets() -> dict[str, float]:
    return {
        "calm": 2.0,
        "moderate": 6.0,
        "strong": 10.0,
        "storm": 14.0,
    }


def _sensor_probs() -> dict[int, float]:
    return {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}


@dataclass
class Config:
    """Simulation, planning, and learning defaults."""

    seed: int = 42
    grid_n: int = 50
    cell_m: float = 100.0
    tick_s: float = 10.0
    n_drones: int = 3
    n_targets_min: int = 6
    n_targets_max: int = 8
    battery: float = 150.0
    reserve: float = 0.20
    capacity_kg: float = 3.0
    true_coef: EnergyCoef = field(default_factory=EnergyCoef)
    energy_floor: float = 0.3
    noise_sd: float = 0.05
    noise_clip: float = 0.15
    hover_energy: float = 0.2
    wind_presets: dict[str, float] = field(default_factory=_wind_presets)
    wind_preset: str = "moderate"
    wind_limit: float = 12.0
    restricted_buffer: int = 1
    max_climb_m: float = 40.0
    interceptor_zone: int = 2
    takeoff_stagger: int = 2
    replan_after_waits: int = 3
    interceptor_detection: int = 6
    interceptor_speed: int = 1
    interceptor_capture: int = 1
    hc_restarts: int = 20
    ga_pop: int = 50
    ga_gens: int = 100
    ga_pc: float = 0.8
    ga_pm: float = 0.1
    ga_tournament: int = 3
    ga_elite: int = 2
    ga_battery_penalty: float = 1000.0
    minimax_depth: int = 4
    minimax_window_r: int = 4
    minimax_trigger: int = 5
    minimax_release: int = 8
    minimax_w_goal: float = 1.0
    minimax_w_threat: float = 1.5
    minimax_cap: int = 5
    sensor_range: int = 3
    sensor_probs: dict[int, float] = field(default_factory=_sensor_probs)
    sensor_max_looks: int = 3
    learning_lr: float = 0.01
    learning_spread: float = 0.4
    learning_scale: tuple[float, float, float, float] = (1.0, 20.0, 5.0, 2.0)
    max_ticks: int = 300
    demo_rejections: bool = False
    n_hills_min: int = 5
    n_hills_max: int = 8
    hill_width_min: float = 5.0
    hill_width_max: float = 12.0
    hill_height_min: float = 50.0
    hill_height_max: float = 250.0
    terrain_max_m: float = 300.0
    terrain_noise_sd: float = 3.0
    diag_step: float = 1.414
    base_margin: int = 1
    n_nofly_min: int = 2
    n_nofly_max: int = 3
    nofly_radius_min: int = 3
    nofly_radius_max: int = 8
    target_min_separation: int = 5
    target_mass_min: float = 0.5
    target_mass_max: float = 2.0