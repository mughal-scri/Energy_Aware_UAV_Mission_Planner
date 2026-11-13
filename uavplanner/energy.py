"""Energy cost of a single move, plus the noisy true model.

The planner uses ``move_energy`` with estimated coefficients. After a step is
flown, ``true_energy`` charges the battery with the hidden coefficients plus
noise. ``e_min`` is the admissible per-cell lower bound used later by A*.
"""

from __future__ import annotations

import numpy as np

from uavplanner.models import Cell, EnergyCoef, World


def headwind(wind: tuple[float, float], step: tuple[int, int]) -> float:
    """Return signed headwind (m/s). Positive means the wind opposes the move."""
    length = (step[0] ** 2 + step[1] ** 2) ** 0.5
    if length == 0:
        return 0.0
    return -(wind[0] * step[0] + wind[1] * step[1]) / length


def move_energy(
    coef: EnergyCoef,
    d: float,
    climb: float,
    headwind: float,
    payload: float,
    floor: float = 0.3,
) -> float:
    """Predicted energy (EU) for one step, floored at ``floor * d``."""
    raw = (
        coef.a * d
        + coef.b * max(0.0, climb)
        + coef.c * d * headwind
        + coef.e * payload * d
    )
    return max(raw, floor * d)


def e_min(coef: EnergyCoef, max_wind: float, floor: float = 0.3) -> float:
    """Lowest energy per cell of travel when |headwind| is at most ``max_wind``."""
    return max(floor, coef.a - coef.c * max_wind)


def true_energy(
    true_coef: EnergyCoef,
    d: float,
    climb: float,
    headwind: float,
    payload: float,
    rng: np.random.Generator,
    floor: float = 0.3,
    noise_sd: float = 0.05,
    noise_clip: float = 0.15,
) -> float:
    """Actual energy used: true model, floor, then multiplicative noise."""
    base = move_energy(true_coef, d, climb, headwind, payload, floor=floor)
    noise = float(np.clip(rng.normal(0.0, noise_sd), -noise_clip, noise_clip))
    return base * (1.0 + noise)


def init_estimate(
    true_coef: EnergyCoef,
    rng: np.random.Generator,
    spread: float = 0.4,
) -> EnergyCoef:
    """Agent's starting coefficients, each off by up to ``spread`` (40% default)."""

    def _jitter(value: float) -> float:
        factor = 1.0 + float(rng.uniform(-spread, spread))
        return max(1e-6, value * factor)

    return EnergyCoef(
        a=_jitter(true_coef.a),
        b=_jitter(true_coef.b),
        c=_jitter(true_coef.c),
        e=_jitter(true_coef.e),
    )


def path_energy(
    world: World,
    path: list[Cell],
    coef: EnergyCoef,
    payload: float,
    floor: float = 0.3,
    diag_step: float = 1.414,
) -> float:
    """Sum ``move_energy`` along a cell path. An empty or single-cell path costs 0."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        d = diag_step if dr != 0 and dc != 0 else 1.0
        climb = float(world.terrain[b] - world.terrain[a])
        hw = headwind(world.wind, (dr, dc))
        total += move_energy(coef, d, climb, hw, payload, floor=floor)
    return total
