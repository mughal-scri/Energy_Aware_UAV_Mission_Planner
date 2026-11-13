"""Traffic manager tests."""

import itertools
import pytest

import numpy as np

from uavplanner.traffic import resolve, takeoff_tick


def test_takeoff_tick() -> None:
    # Default stagger = 2
    assert takeoff_tick(1) == 0
    assert takeoff_tick(2) == 2
    assert takeoff_tick(3) == 4

    # Custom stagger
    assert takeoff_tick(1, stagger=3) == 0
    assert takeoff_tick(2, stagger=3) == 3
    assert takeoff_tick(3, stagger=3) == 6

    # Edge cases
    with pytest.raises(ValueError, match="drone_id"):
        takeoff_tick(0)
    with pytest.raises(ValueError, match="drone_id"):
        takeoff_tick(-1)
    with pytest.raises(ValueError, match="stagger"):
        takeoff_tick(1, stagger=-1)


def test_resolve_empty_and_single() -> None:
    assert resolve({}, {}) == {}
    assert resolve({1: (0, 0)}, {1: (0, 1)}) == {1: (0, 1)}


def test_resolve_independent_moves() -> None:
    positions = {1: (0, 0), 2: (5, 5)}
    proposals = {1: (0, 1), 2: (5, 6)}
    assert resolve(positions, proposals) == {1: (0, 1), 2: (5, 6)}


def test_resolve_destination_conflict_priority() -> None:
    # Both drones want (1, 1). Drone 1 has higher priority (lower ID).
    positions = {1: (1, 0), 2: (0, 1)}
    proposals = {1: (1, 1), 2: (1, 1)}
    approved = resolve(positions, proposals)
    assert approved[1] == (1, 1)
    assert approved[2] == (0, 1)  # drone 2 waits at current position


def test_resolve_destination_occupied_by_waiting_drone() -> None:
    # Drone 2 is already at (1, 1) and stays there.
    # Drone 1 (higher priority) wants to move into (1, 1).
    positions = {1: (1, 0), 2: (1, 1)}
    proposals = {1: (1, 1), 2: (1, 1)}
    approved = resolve(positions, proposals)
    # Drone 2 stays; Drone 1 cannot displace a waiting drone and must wait at (1, 0)
    assert approved[2] == (1, 1)
    assert approved[1] == (1, 0)


def test_resolve_head_on_swap() -> None:
    # Drone 1 at (0, 0) wants (0, 1); Drone 2 at (0, 1) wants (0, 0).
    positions = {1: (0, 0), 2: (0, 1)}
    proposals = {1: (0, 1), 2: (0, 0)}
    approved = resolve(positions, proposals)
    # Both must wait to avoid colliding head-on
    assert approved[1] == (0, 0)
    assert approved[2] == (0, 1)


def test_resolve_cyclic_movement() -> None:
    # Three drones rotating around a triangle: A -> B, B -> C, C -> A
    a, b, c = (0, 0), (0, 1), (1, 0)
    positions = {1: a, 2: b, 3: c}
    proposals = {1: b, 2: c, 3: a}
    approved = resolve(positions, proposals)
    # Since destinations are all distinct and no pairwise swaps exist, all move safely
    assert approved == {1: b, 2: c, 3: a}


def test_resolve_input_validation() -> None:
    # Missing position key
    with pytest.raises(KeyError, match="not found in positions"):
        resolve({1: (0, 0)}, {1: (0, 1), 2: (1, 1)})

    # Active drones with duplicate starting position
    with pytest.raises(ValueError, match="initial cell"):
        resolve({1: (0, 0), 2: (0, 0)}, {1: (0, 1), 2: (1, 0)})


def test_resolve_random_cases_property() -> None:
    """Verify no shared cells and no swaps over 20,000 random cases."""
    rng = np.random.default_rng(42)
    drone_ids = (1, 2, 3)
    grid_dim = 10  # small grid to force frequent contention

    for _ in range(20000):
        # Choose 3 distinct initial positions
        cells = set()
        while len(cells) < 3:
            cells.add((int(rng.integers(0, grid_dim)), int(rng.integers(0, grid_dim))))
        cell_list = list(cells)
        positions = {d: cell_list[idx] for idx, d in enumerate(drone_ids)}

        # Proposals: each drone either stays or steps to one of 8 neighbours
        proposals = {}
        for d in drone_ids:
            r, c = positions[d]
            dr = int(rng.integers(-1, 2))
            dc = int(rng.integers(-1, 2))
            proposals[d] = (r + dr, c + dc)

        approved = resolve(positions, proposals)

        # 1. No two approved cells are equal
        assert len(set(approved.values())) == len(approved)

        # 2. No pairwise swaps survived
        for i, j in itertools.combinations(drone_ids, 2):
            is_swap = (
                positions[i] != positions[j]
                and approved[i] == positions[j]
                and approved[j] == positions[i]
            )
            assert not is_swap

        # 3. Each drone either gets its proposal or waits at its current position
        for d in drone_ids:
            assert approved[d] in (positions[d], proposals[d])
