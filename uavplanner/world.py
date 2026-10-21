"""Scenario generation: terrain, wind, no-fly zones, base, and targets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from uavplanner.config import Config
from uavplanner.models import Cell, Target, World

_NEIGHBOUR_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def chebyshev(a: Cell, b: Cell) -> int:
    """Number of 8-neighbour steps between two cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def nofly_buffer_mask(no_fly: np.ndarray, buffer: int) -> np.ndarray:
    """Chebyshev dilation of the no-fly mask (the static buffer part of restricted cells)."""
    if buffer <= 0:
        return no_fly.copy()
    n = int(no_fly.shape[0])
    out = no_fly.copy()
    blocked = np.argwhere(no_fly)
    for r, c in blocked:
        r0, r1 = max(0, int(r) - buffer), min(n, int(r) + buffer + 1)
        c0, c1 = max(0, int(c) - buffer), min(n, int(c) + buffer + 1)
        out[r0:r1, c0:c1] = True
    return out


def neighbors(cell: Cell, n: int, diag_step: float = 1.414):
    """Yield ``(neighbour, step_length)`` for the eight cells around ``cell``."""
    row, col = cell
    for dr, dc in _NEIGHBOUR_DELTAS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < n and 0 <= nc < n:
            length = diag_step if dr != 0 and dc != 0 else 1.0
            yield (nr, nc), length


def reconstruct_path(parent: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    """Walk parent pointers from ``goal`` back to the start (whose parent is None)."""
    path: list[Cell] = []
    cur: Cell | None = goal
    seen: set[Cell] = set()
    while cur is not None:
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def make_world(cfg: Config, rng: np.random.Generator) -> World:
    """Build a reproducible hilly map with wind and no-fly circles."""
    n = cfg.grid_n
    terrain = _gaussian_hills(n, cfg, rng)
    base = _place_base(n, cfg.base_margin, rng)
    no_fly, circles = _place_nofly(n, base, cfg, rng)
    if cfg.demo_rejections:
        _wall_in_a_cell(no_fly, base, rng)
    speed = cfg.wind_presets[cfg.wind_preset]
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    wind = (speed * math.sin(angle), speed * math.cos(angle))
    return World(
        terrain=terrain,
        no_fly=no_fly,
        wind=wind,
        base=base,
        nofly_circles=circles,
    )


def make_targets(world: World, cfg: Config, rng: np.random.Generator) -> list[Target]:
    """Place 6 to 8 targets on free cells, at least ``target_min_separation`` apart."""
    n = int(world.terrain.shape[0])
    n_targets = int(rng.integers(cfg.n_targets_min, cfg.n_targets_max + 1))
    targets: list[Target] = []
    next_id = 1

    if cfg.demo_rejections:
        enclosed = _find_enclosed_cell(world)
        if enclosed is not None:
            targets.append(
                Target(
                    id=next_id,
                    pos=enclosed,
                    mass_kg=_sample_mass(cfg, rng),
                    status="pending",
                )
            )
            next_id += 1
        blocked = _find_nofly_cell(world)
        if blocked is not None:
            targets.append(
                Target(
                    id=next_id,
                    pos=blocked,
                    mass_kg=_sample_mass(cfg, rng),
                    status="pending",
                )
            )
            next_id += 1

    occupied = [t.pos for t in targets]
    free = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if not world.no_fly[r, c] and (r, c) != world.base and (r, c) not in occupied
    ]
    rng.shuffle(free)
    for cell in free:
        if len(targets) >= n_targets:
            break
        if any(chebyshev(cell, pos) < cfg.target_min_separation for pos in occupied):
            continue
        occupied.append(cell)
        targets.append(
            Target(id=next_id, pos=cell, mass_kg=_sample_mass(cfg, rng), status="pending")
        )
        next_id += 1
    return targets


def save_scenario(
    directory: str | Path,
    name: str,
    world: World,
    targets: list[Target],
    cfg: Config,
) -> None:
    """Write ``<name>.npz`` (grids) and ``<name>.json`` (parameters and targets)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / f"{name}.npz",
        terrain=world.terrain,
        no_fly=world.no_fly,
        wind=np.array(world.wind),
        base=np.array(world.base),
        nofly_circles=np.array(world.nofly_circles, dtype=float),
    )
    payload = {
        "seed": cfg.seed,
        "wind_preset": cfg.wind_preset,
        "base": list(world.base),
        "wind": list(world.wind),
        "targets": [
            {"id": t.id, "pos": list(t.pos), "mass_kg": t.mass_kg, "status": t.status}
            for t in targets
        ],
    }
    (directory / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scenario(directory: str | Path, name: str) -> tuple[World, list[Target]]:
    """Load a scenario saved by ``save_scenario``."""
    directory = Path(directory)
    data = np.load(directory / f"{name}.npz")
    circles_arr = data["nofly_circles"]
    circles = [tuple(row) for row in circles_arr] if circles_arr.size else []
    world = World(
        terrain=data["terrain"],
        no_fly=data["no_fly"].astype(bool),
        wind=(float(data["wind"][0]), float(data["wind"][1])),
        base=(int(data["base"][0]), int(data["base"][1])),
        nofly_circles=circles,
    )
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    targets = [
        Target(
            id=int(row["id"]),
            pos=(int(row["pos"][0]), int(row["pos"][1])),
            mass_kg=float(row["mass_kg"]),
            status=str(row["status"]),
        )
        for row in meta["targets"]
    ]
    return world, targets


def _sample_mass(cfg: Config, rng: np.random.Generator) -> float:
    return float(rng.uniform(cfg.target_mass_min, cfg.target_mass_max))


def _gaussian_hills(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    n_hills = int(rng.integers(cfg.n_hills_min, cfg.n_hills_max + 1))
    rows, cols = np.indices((n, n), dtype=float)
    terrain = np.zeros((n, n), dtype=float)
    for _ in range(n_hills):
        cy = float(rng.uniform(0.0, n - 1))
        cx = float(rng.uniform(0.0, n - 1))
        width = float(rng.uniform(cfg.hill_width_min, cfg.hill_width_max))
        height = float(rng.uniform(cfg.hill_height_min, cfg.hill_height_max))
        terrain += height * np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2.0 * width**2))
    terrain += rng.normal(0.0, cfg.terrain_noise_sd, size=terrain.shape)
    terrain -= float(terrain.min())
    peak = float(terrain.max())
    if peak > 0:
        terrain *= cfg.terrain_max_m / peak
    return terrain


def _place_base(n: int, margin: int, rng: np.random.Generator) -> Cell:
    edge = int(rng.integers(0, 4))
    k = int(rng.integers(margin, n - margin))
    if edge == 0:
        return (margin, k)
    if edge == 1:
        return (n - 1 - margin, k)
    if edge == 2:
        return (k, margin)
    return (k, n - 1 - margin)


def _place_nofly(
    n: int, base: Cell, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    n_zones = int(rng.integers(cfg.n_nofly_min, cfg.n_nofly_max + 1))
    mask = np.zeros((n, n), dtype=bool)
    circles: list[tuple[float, float, float]] = []
    rows, cols = np.indices((n, n), dtype=float)
    for _ in range(n_zones):
        for _attempt in range(80):
            radius = float(rng.integers(cfg.nofly_radius_min, cfg.nofly_radius_max + 1))
            cr = float(rng.uniform(0.0, n - 1))
            cc = float(rng.uniform(0.0, n - 1))
            if math.hypot(base[0] - cr, base[1] - cc) <= radius + 1.0:
                continue
            disk = (rows - cr) ** 2 + (cols - cc) ** 2 <= radius**2
            if disk[base]:
                continue
            mask |= disk
            circles.append((cr, cc, radius))
            break
    return mask, circles


def _wall_in_a_cell(no_fly: np.ndarray, base: Cell, rng: np.random.Generator) -> None:
    """Surround one free cell with a no-fly ring (for the expert-system demo)."""
    n = no_fly.shape[0]
    for _attempt in range(80):
        r = int(rng.integers(3, n - 3))
        c = int(rng.integers(3, n - 3))
        if chebyshev((r, c), base) < 6:
            continue
        ring = [
            (r + dr, c + dc)
            for dr, dc in _NEIGHBOUR_DELTAS
        ]
        if any(cell == base or not (0 <= cell[0] < n and 0 <= cell[1] < n) for cell in ring):
            continue
        for cell in ring:
            no_fly[cell] = True
        no_fly[r, c] = False
        return


def _find_enclosed_cell(world: World) -> Cell | None:
    n = int(world.terrain.shape[0])
    for r in range(n):
        for c in range(n):
            if world.no_fly[r, c] or (r, c) == world.base:
                continue
            nbs = list(neighbors((r, c), n))
            if nbs and all(world.no_fly[nb] for nb, _ in nbs):
                return (r, c)
    return None


def _find_nofly_cell(world: World) -> Cell | None:
    found = np.argwhere(world.no_fly)
    if found.size == 0:
        return None
    r, c = found[0]
    return (int(r), int(c))
