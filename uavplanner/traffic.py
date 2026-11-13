"""Traffic manager: takeoff staggering and collision resolution."""

from __future__ import annotations

from collections.abc import Mapping
import itertools

from uavplanner.models import Cell


def takeoff_tick(drone_id: int, stagger: int = 2) -> int:
    """Return the scheduled takeoff tick for a drone.

    Drones leave base in ascending order of their IDs with a fixed stagger
    gap to prevent immediate congestion at the launch point.
    """
    if drone_id < 1:
        raise ValueError("drone_id must be >= 1")
    if stagger < 0:
        raise ValueError("stagger must be non-negative")
    return (drone_id - 1) * stagger


def resolve(
    positions: Mapping[int, Cell],
    proposals: Mapping[int, Cell],
) -> dict[int, Cell]:
    """Resolve destination conflicts and pairwise head-on swaps among drones.

    Applies Level 1 priority-based rules:
    1. If multiple drones want the same cell, any drone already waiting at that
       cell retains it, and all movers must wait. If all are movers, the drone
       with the highest priority (lowest id) wins, and the others wait at their
       current positions.
    2. If two drones propose to swap cells (head-on move), the lower-priority
       drone (higher id) waits.
    3. The resolution iterates until no further conflicts or swaps remain.
    """
    if not proposals:
        return {}

    for d in proposals:
        if d not in positions:
            raise KeyError(f"drone {d} not found in positions")

    # Validate distinct starting positions for active drones
    start_cells = [positions[d] for d in proposals]
    if len(set(start_cells)) != len(start_cells):
        raise ValueError("active drones cannot occupy the same initial cell")

    approved: dict[int, Cell] = {d: proposals[d] for d in proposals}
    changed = True
    while changed:
        changed = False
        by_dest: dict[Cell, list[int]] = {}
        for d, cell in approved.items():
            by_dest.setdefault(cell, []).append(d)

        for cell, ids in by_dest.items():
            if len(ids) > 1:
                waiting = [d for d in ids if approved[d] == positions[d]]
                # someone already stays here: every mover into this cell waits;
                # otherwise the highest priority (lowest id) wins
                losers = [d for d in ids if d not in waiting] if waiting else sorted(ids)[1:]
                for d in losers:
                    approved[d] = positions[d]
                    changed = True

        for i, j in itertools.combinations(sorted(approved), 2):
            if (
                positions[i] != positions[j]
                and approved[i] == positions[j]
                and approved[j] == positions[i]
            ):
                approved[j] = positions[j]
                changed = True

    return approved
