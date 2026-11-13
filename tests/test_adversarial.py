"""Minimax and Alpha-Beta equivalence and evasion behavior."""

import numpy as np

from uavplanner.adversarial import choose_move
from uavplanner.config import Config


def test_alpha_beta_matches_minimax_over_100_seeded_positions() -> None:
    rng = np.random.default_rng(19)
    mask = np.zeros((9, 9), dtype=bool)
    cfg = Config()
    fewer = 0
    for _ in range(100):
        drone, interceptor, waypoint = [tuple(int(v) for v in rng.integers(0, 9, size=2)) for _ in range(3)]
        plain = choose_move(drone, interceptor, waypoint, mask, cfg=cfg, use_alpha_beta=False)
        pruned = choose_move(drone, interceptor, waypoint, mask, cfg=cfg, use_alpha_beta=True)
        assert pruned[:2] == plain[:2]
        fewer += pruned[2] < plain[2]
    assert fewer > 50


def test_drone_one_cell_outside_capture_moves_away() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    move, _value, _nodes = choose_move((4, 4), (4, 2), (4, 8), mask, depth=4)
    assert max(abs(move[0] - 4), abs(move[1] - 2)) > 2


def test_blocked_and_terminal_inputs_are_handled() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    assert choose_move((2, 2), (2, 3), (0, 0), mask)[0] == (2, 2)
    assert choose_move((2, 2), (4, 4), (2, 2), mask)[:2] == ((2, 2), 1000.0)
    mask[1, :] = True
    mask[3, :] = True
    mask[2, [1, 3]] = True
    move, _value, _nodes = choose_move((2, 2), (0, 0), (4, 4), mask, depth=2)
    assert move == (2, 2)
