"""Genetic ordering search for directed drone-tour cost matrices."""

from __future__ import annotations

import numpy as np

from uavplanner.config import Config
from uavplanner.hill_climbing import tour_cost


def genetic_order(
    cost: np.ndarray,
    stops: list[int],
    pop: int = Config.ga_pop,
    gens: int = Config.ga_gens,
    pc: float = Config.ga_pc,
    pm: float = Config.ga_pm,
    tournament: int = Config.ga_tournament,
    elite: int = Config.ga_elite,
    battery_limit: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], float, list[float]]:
    """Optimize a directed tour using tournament selection, OX, and mutation.

    The returned cost is the physical tour energy, not its penalized fitness.
    When ``battery_limit`` is supplied, a configurable penalty makes over-limit
    tours lose selection contests to otherwise comparable feasible tours.
    ``history`` is the best physical cost in each generation and cannot rise
    because the best ``elite`` members are copied unchanged.
    """
    _validate_settings(pop, gens, pc, pm, tournament, elite, stops)
    if rng is None:
        rng = np.random.default_rng()
    if not stops:
        return [], 0.0, [0.0]

    population = [list(stops)] + [list(rng.permutation(stops)) for _ in range(pop - 1)]

    def raw_cost(order: list[int]) -> float:
        return tour_cost(order, cost)

    def fitness(order: list[int]) -> float:
        value = raw_cost(order)
        if battery_limit is not None and value > battery_limit:
            return value + Config.ga_battery_penalty
        return value

    def ranked(members: list[list[int]]) -> list[list[int]]:
        return sorted(members, key=fitness)

    population = ranked(population)
    best_order = list(population[0])
    best_cost = raw_cost(best_order)
    history = [best_cost]
    for _generation in range(gens):
        ranked_population = ranked(population)
        next_population = [list(member) for member in ranked_population[:elite]]
        while len(next_population) < pop:
            first_parent = _tournament_pick(ranked_population, fitness, tournament, rng)
            second_parent = _tournament_pick(ranked_population, fitness, tournament, rng)
            if rng.random() < pc:
                first_child, second_child = _order_crossover(first_parent, second_parent, rng)
            else:
                first_child, second_child = list(first_parent), list(second_parent)
            _swap_mutate(first_child, pm, rng)
            _swap_mutate(second_child, pm, rng)
            next_population.append(first_child)
            if len(next_population) < pop:
                next_population.append(second_child)
        population = next_population
        generation_best = ranked(population)[0]
        generation_cost = raw_cost(generation_best)
        if fitness(generation_best) < fitness(best_order):
            best_order = list(generation_best)
            best_cost = generation_cost
        history.append(best_cost)
    return best_order, best_cost, history


def _validate_settings(
    pop: int, gens: int, pc: float, pm: float, tournament: int, elite: int, stops: list[int]
) -> None:
    if pop < 2:
        raise ValueError("pop must be at least 2")
    if gens < 1:
        raise ValueError("gens must be at least 1")
    if not 0.0 <= pc <= 1.0 or not 0.0 <= pm <= 1.0:
        raise ValueError("pc and pm must be between 0 and 1")
    if tournament < 1 or tournament > pop:
        raise ValueError("tournament must be between 1 and pop")
    if elite < 1 or elite > pop:
        raise ValueError("elite must be between 1 and pop")
    if len(set(stops)) != len(stops):
        raise ValueError("stops must contain unique matrix indices")


def _tournament_pick(
    population: list[list[int]], fitness, tournament: int, rng: np.random.Generator
) -> list[int]:
    """Return the best of a random candidate subset."""
    choices = rng.choice(len(population), size=tournament, replace=False)
    return min((population[int(index)] for index in choices), key=fitness)


def _order_crossover(
    first: list[int], second: list[int], rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Create two valid order-crossover children from permutation parents."""
    if len(first) < 2:
        return list(first), list(second)
    left, right = sorted(int(index) for index in rng.choice(len(first), size=2, replace=False))
    return _ox_child(first, second, left, right), _ox_child(second, first, left, right)


def _ox_child(source: list[int], fill: list[int], left: int, right: int) -> list[int]:
    child: list[int | None] = [None] * len(source)
    child[left : right + 1] = source[left : right + 1]
    remaining = [stop for stop in fill if stop not in child]
    positions = [index for index, value in enumerate(child) if value is None]
    for index, stop in zip(positions, remaining):
        child[index] = stop
    return [int(stop) for stop in child if stop is not None]


def _swap_mutate(order: list[int], probability: float, rng: np.random.Generator) -> None:
    """Swap two genes in place with the supplied probability."""
    if len(order) < 2 or rng.random() >= probability:
        return
    first, second = rng.choice(len(order), size=2, replace=False)
    order[int(first)], order[int(second)] = order[int(second)], order[int(first)]
