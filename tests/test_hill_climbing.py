"""Checks for directed tour-cost calculation."""

import math

import numpy as np
import pytest

from uavplanner.hill_climbing import brute_force_order, hill_climb, tour_cost


def test_tour_cost_includes_outbound_intermediate_and_return_legs() -> None:
    cost = np.array(
        [
            [0.0, 2.0, 4.0, 10.0],
            [3.0, 0.0, 5.0, 7.0],
            [4.0, 2.0, 0.0, 1.0],
            [8.0, 2.0, 3.0, 0.0],
        ]
    )
    assert tour_cost([1, 2, 3], cost) == pytest.approx(16.0)


def test_tour_cost_preserves_direction_and_empty_tour() -> None:
    cost = np.array(
        [
            [0.0, 2.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 2.0, 0.0],
        ]
    )
    assert tour_cost([1, 2], cost) == pytest.approx(11.0)
    assert tour_cost([2, 1], cost) == pytest.approx(9.0)
    assert tour_cost([], cost) == 0.0


def test_tour_cost_keeps_unreachable_leg_and_rejects_bad_inputs() -> None:
    cost = np.array([[0.0, math.inf], [1.0, 0.0]])
    assert math.isinf(tour_cost([1], cost))
    with pytest.raises(ValueError, match="square"):
        tour_cost([1], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="inside"):
        tour_cost([2], cost)


def test_hill_climb_beats_its_initial_order_and_matches_brute_force() -> None:
    cost = np.array(
        [
            [0.0, 8.0, 2.0, 6.0],
            [3.0, 0.0, 7.0, 1.0],
            [5.0, 4.0, 0.0, 2.0],
            [1.0, 6.0, 3.0, 0.0],
        ]
    )
    stops = [1, 2, 3]
    baseline = tour_cost(stops, cost)
    best_order, best_cost, history = hill_climb(
        cost, stops, restarts=8, rng=np.random.default_rng(4)
    )
    exact_order, exact_cost = brute_force_order(cost, stops)
    assert sorted(best_order) == stops
    assert best_cost <= baseline
    assert best_cost == pytest.approx(exact_cost)
    assert tour_cost(exact_order, cost) == pytest.approx(exact_cost)
    assert history[0] == pytest.approx(baseline)
    assert all(next_cost <= previous_cost for previous_cost, next_cost in zip(history, history[1:]))


def test_brute_force_is_bounded_and_hill_climb_rejects_invalid_starts() -> None:
    cost = np.zeros((9, 9))
    with pytest.raises(ValueError, match="seven"):
        brute_force_order(cost, list(range(1, 9)))
    with pytest.raises(ValueError, match="unique"):
        hill_climb(cost, [1, 1])
    with pytest.raises(ValueError, match="at least"):
        hill_climb(cost, [1], restarts=0)
