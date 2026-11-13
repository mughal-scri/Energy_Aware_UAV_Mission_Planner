"""Tour-cost primitives for ordering algorithms.

The cost matrix is directed: travelling from stop A to B can cost differently
from travelling B to A. ``tour_cost`` therefore preserves the supplied order
and includes both the outbound leg from base and the final return leg.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def tour_cost(order: list[int], cost: np.ndarray, base: int = 0) -> float:
    """Return the directed cost of ``base -> order -> base``.

    ``order`` contains cost-matrix indices and does not include ``base``.
    An empty order is an empty tour and costs zero. Infinite leg costs are
    returned unchanged, allowing later ordering algorithms to reject an
    unreachable tour.

    Raises:
        ValueError: If ``cost`` is not square or an index is outside it.
    """
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    n_stops = cost.shape[0]
    indices = [base, *order]
    if any(index < 0 or index >= n_stops for index in indices):
        raise ValueError("base and order indices must be inside the cost matrix")
    if not order:
        return 0.0

    route = [base, *order, base]
    return float(sum(cost[start, goal] for start, goal in zip(route, route[1:])))


def hill_climb(
    cost: np.ndarray,
    stops: list[int],
    restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], float, list[float]]:
    """Find a low-cost order with steepest swap and 2-opt improvements.

    The supplied ``stops`` order is the first start, so the result can never
    be worse than that baseline. Each later restart begins from a shuffled
    permutation. ``history`` records the best cost seen after every local
    search step and is suitable for a convergence plot.

    Raises:
        ValueError: If ``restarts`` is less than one or stops are not unique.
    """
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    if len(set(stops)) != len(stops):
        raise ValueError("stops must contain unique matrix indices")
    if rng is None:
        rng = np.random.default_rng()
    if not stops:
        return [], 0.0, [0.0]

    best_order = list(stops)
    best_cost = tour_cost(best_order, cost)
    history = [best_cost]
    for restart in range(restarts):
        order = list(stops) if restart == 0 else list(rng.permutation(stops))
        current_cost = tour_cost(order, cost)
        while True:
            candidate_order = order
            candidate_cost = current_cost
            for neighbour in _neighbours(order):
                neighbour_cost = tour_cost(neighbour, cost)
                if neighbour_cost < candidate_cost:
                    candidate_order = neighbour
                    candidate_cost = neighbour_cost
            if candidate_cost >= current_cost:
                break
            order = candidate_order
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_order = list(order)
                best_cost = current_cost
            history.append(best_cost)
        if current_cost < best_cost:
            best_order = list(order)
            best_cost = current_cost
        history.append(best_cost)
    return best_order, best_cost, history


def brute_force_order(
    cost: np.ndarray, stops: list[int], base: int = 0
) -> tuple[list[int], float]:
    """Return the exact best directed tour for at most seven non-base stops.

    This deliberately small oracle is for tests and comparisons, not
    production-sized target sets: it evaluates ``len(stops)!`` permutations.
    """
    if len(stops) > 7:
        raise ValueError("brute force is limited to seven stops")
    if len(set(stops)) != len(stops):
        raise ValueError("stops must contain unique matrix indices")
    if not stops:
        return [], 0.0
    best_order: list[int] | None = None
    best_cost = float("inf")
    for order_tuple in permutations(stops):
        order = list(order_tuple)
        value = tour_cost(order, cost, base=base)
        if value < best_cost:
            best_order = order
            best_cost = value
    return best_order or [], best_cost


def _neighbours(order: list[int]):
    """Yield unique swap and inclusive-segment 2-opt neighbours."""
    seen: set[tuple[int, ...]] = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            swapped = list(order)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            reversed_segment = order[:i] + list(reversed(order[i : j + 1])) + order[j + 1 :]
            for neighbour in (swapped, reversed_segment):
                key = tuple(neighbour)
                if key not in seen:
                    seen.add(key)
                    yield neighbour
