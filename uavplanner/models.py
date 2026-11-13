"""Shared dataclasses for the mission planner.

These types are the only objects modules should pass between each other.
Randomness, config, and algorithm state stay outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]
"""Grid coordinate ``(row, col)``. Row 0 is the top when plotted."""


@dataclass
class EnergyCoef:
    """Coefficients of the linear energy-per-move model.

    ``a`` is base cost per cell, ``b`` per metre climbed, ``c`` per cell per
    m/s of headwind, and ``e`` per kilogram of payload per cell.
    """

    a: float = 1.0
    b: float = 0.05
    c: float = 0.08
    e: float = 0.15


@dataclass
class World:
    """Static map: heights, no-fly mask, wind vector, and the base cell."""

    terrain: np.ndarray
    no_fly: np.ndarray
    wind: tuple[float, float]
    base: Cell
    nofly_circles: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class Target:
    """A known static target the fleet may visit."""

    id: int
    pos: Cell
    mass_kg: float
    status: str = "pending"


@dataclass
class Drone:
    """One simulated drone. ``id`` is also the priority (lower id wins)."""

    id: int
    battery: float = 150.0
    capacity_kg: float = 3.0
    pos: Cell = (0, 0)
    mode: str = "PARKED"
    stops: list[int] = field(default_factory=list)
    route: list[Cell] = field(default_factory=list)
    waited: int = 0
    looks: int = 0


@dataclass
class Interceptor:
    """Reactive threat that patrols waypoints and chases nearby drones."""

    pos: Cell
    waypoints: list[Cell]
    wp_index: int = 0
    mode: str = "PATROL"
    last_seen: Cell | None = None
    target_last_seen: Cell | None = None
    lost_ticks: int = 0


@dataclass
class SearchResult:
    """Path found by a grid search, plus accounting for comparisons."""

    path: list[Cell]
    cost: float
    expanded: int
    explored: list[Cell]


@dataclass
class PerceptRow:
    """One flown step, used later by the learning update."""

    mission: int
    drone: int
    d: float
    climb: float
    headwind: float
    payload: float
    predicted: float
    actual: float
