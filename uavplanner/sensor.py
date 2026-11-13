"""Probabilistic confirmation sensor for known static targets."""

from __future__ import annotations

import numpy as np

from uavplanner.config import Config
from uavplanner.models import Cell
from uavplanner.world import chebyshev


def detect_prob(dist: int) -> float:
    """Return the configured default detection probability at a cell distance."""
    if dist < 0:
        raise ValueError("dist must be non-negative")
    return Config().sensor_probs.get(dist, 0.0)


def look(drone_pos: Cell, target_pos: Cell, rng: np.random.Generator) -> bool:
    """Make one seeded detection attempt using Chebyshev cell distance."""
    return bool(rng.random() < detect_prob(chebyshev(drone_pos, target_pos)))
