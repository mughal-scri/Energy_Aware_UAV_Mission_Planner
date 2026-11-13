"""Genetic-order checks."""

import numpy as np
import pytest

from uavplanner.genetic import genetic_order
from uavplanner.hill_climbing import brute_force_order, tour_cost


def _cost_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 7.0, 3.0, 9.0, 5.0, 8.0, 4.0],
            [4.0, 0.0, 6.0, 2.0, 8.0, 5.0, 7.0],
            [6.0, 5.0, 0.0, 4.0, 3.0, 7.0, 2.0],
            [8.0, 3.0, 5.0, 0.0, 6.0, 2.0, 4.0],
            [2.0, 7.0, 4.0, 5.0, 0.0, 3.0, 6.0],
            [5.0, 4.0, 8.0, 3.0, 6.0, 0.0, 2.0],
            [7.0, 2.0, 6.0, 5.0, 4.0, 3.0, 0.0],
        ]
    )


def test_genetic_order_children_remain_valid_and_elitism_preserves_best() -> None:
    cost = _cost_matrix()
    stops = [1, 2, 3, 4, 5, 6]
    order, value, history = genetic_order(
        cost, stops, pop=30, gens=40, rng=np.random.default_rng(8)
    )
    assert sorted(order) == stops
    assert value == pytest.approx(tour_cost(order, cost))
    assert all(next_cost <= previous_cost for previous_cost, next_cost in zip(history, history[1:]))


def test_genetic_order_is_near_brute_force_at_six_stops() -> None:
    cost = _cost_matrix()
    stops = [1, 2, 3, 4, 5, 6]
    _exact_order, exact_cost = brute_force_order(cost, stops)
    _order, genetic_cost, _history = genetic_order(
        cost, stops, pop=50, gens=100, rng=np.random.default_rng(2)
    )
    assert genetic_cost <= exact_cost * 1.10


def test_genetic_order_validates_settings() -> None:
    cost = _cost_matrix()
    with pytest.raises(ValueError, match="at least 2"):
        genetic_order(cost, [1, 2], pop=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        genetic_order(cost, [1, 2], pm=2.0)
