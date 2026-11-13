"""Energy-model checks."""

import numpy as np
import pytest

from uavplanner.energy import e_min, init_estimate, move_energy, true_energy
from uavplanner.models import EnergyCoef


def test_move_energy_check_values() -> None:
    coef = EnergyCoef()
    assert move_energy(coef, 1, 20, 4, 1) == pytest.approx(2.47, abs=1e-12)
    assert move_energy(coef, 1, -20, -4, 1) == pytest.approx(0.83, abs=1e-12)


def test_e_min_check_value() -> None:
    assert e_min(EnergyCoef(), 8) == pytest.approx(0.36, abs=1e-12)


def test_no_move_cheaper_than_e_min() -> None:
    coef = EnergyCoef()
    max_wind = 8.0
    floor_per_cell = e_min(coef, max_wind)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        d = float(rng.choice([1.0, 1.414]))
        climb = float(rng.uniform(-40.0, 40.0))
        hw = float(rng.uniform(-max_wind, max_wind))
        payload = float(rng.uniform(0.0, 3.0))
        cost = move_energy(coef, d, climb, hw, payload)
        assert cost >= floor_per_cell * d - 1e-12


def test_true_energy_noise_is_clipped() -> None:
    rng = np.random.default_rng(1)
    samples = [
        true_energy(EnergyCoef(), 1.0, 0.0, 0.0, 0.0, rng)
        for _ in range(400)
    ]
    base = move_energy(EnergyCoef(), 1.0, 0.0, 0.0, 0.0)
    assert min(samples) >= base * (1.0 - 0.15) - 1e-9
    assert max(samples) <= base * (1.0 + 0.15) + 1e-9


def test_init_estimate_stays_within_spread() -> None:
    true_coef = EnergyCoef()
    rng = np.random.default_rng(2)
    est = init_estimate(true_coef, rng, spread=0.4)
    for name in ("a", "b", "c", "e"):
        true_v = getattr(true_coef, name)
        est_v = getattr(est, name)
        assert 0.6 * true_v - 1e-9 <= est_v <= 1.4 * true_v + 1e-9
