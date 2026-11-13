"""Planning service: safety verdict, assignment, order, and routes.

The planning pipeline coordinates safety, assignment, ordering, and routes. Parked drones start at the
world base. An airborne drone is a replan: its route starts at ``drone.pos``,
visits the remaining targets, and still ends at the base. Tour energy is
checked against ``battery * (1 - reserve)``. When a tour is too expensive the
CSP energy scale is multiplied by ``csp_scale_shrink`` up to
``csp_verify_retries`` times, and then the farthest overloaded target is
dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from uavplanner.config import Config
from uavplanner.csp import avg_cell_cost, csp_assign, estimate_target_cost
from uavplanner.expert import default_rules, forward_chain, static_restricted
from uavplanner.genetic import genetic_order
from uavplanner.hill_climbing import hill_climb, tour_cost
from uavplanner.informed import astar, cost_matrix, euclid
from uavplanner.models import Cell, Drone, EnergyCoef, Target, World
from uavplanner.uninformed import dfs_reachable

_AIRBORNE = {"TRANSIT", "HOVER", "WAIT", "EVADE", "RETURN"}
_COST_TOLERANCE = 1e-6


@dataclass
class Plan:
    """One complete pre-flight decision for the fleet."""

    verdict: str
    reasons: list[str]
    rejected: dict[int, str]
    assignment: dict[int, int]
    order: dict[int, list[int]]
    routes: dict[int, list[Cell]]
    est_energy: dict[int, float]
    order_costs: dict[int, dict[str, float]] = field(default_factory=dict)


def format_plan(plan: Plan) -> str:
    """Render a plan as the explanation an operator would read."""
    lines = [f"Verdict: {plan.verdict}"]
    if plan.reasons:
        lines.append("Explanation:")
        lines.extend(f"  - {reason}" for reason in plan.reasons)
    if plan.rejected:
        lines.append("Rejected targets:")
        for target_id, reason in sorted(plan.rejected.items()):
            lines.append(f"  - target {target_id}: {reason}")
    if plan.verdict != "GO":
        return "\n".join(lines)
    lines.append("Assignment:")
    for target_id, drone_id in sorted(plan.assignment.items()):
        lines.append(f"  - target {target_id} -> drone {drone_id}")
    for drone_id, visiting in sorted(plan.order.items()):
        energy = plan.est_energy.get(drone_id, 0.0)
        if not visiting:
            lines.append(f"Drone {drone_id}: unused, stays parked")
            continue
        cells = len(plan.routes.get(drone_id, []))
        lines.append(
            f"Drone {drone_id}: visit {visiting}, estimated energy {energy:.2f} EU, "
            f"route cells {cells}"
        )
    return "\n".join(lines)


def plan_mission(
    world: World,
    drones: list[Drone],
    targets: list[Target],
    coef: EnergyCoef,
    cfg: Config,
    rng: np.random.Generator,
    blocked_extra: np.ndarray | None = None,
) -> Plan:
    """Plan a mission, or replan one airborne drone, and explain the verdict.

    ``blocked_extra`` is combined with the static restricted mask. It is how a
    replan keeps other drones' cells, or a known threat zone, out of the route.
    """
    _validate(world, drones, targets, cfg, blocked_extra)
    rules = {rule.id: rule for rule in default_rules(cfg)}
    blocked = _blocked_mask(world, cfg, blocked_extra)
    wind_facts, wind_fired = forward_chain(
        {"wind_speed": _wind_speed(world)},
        [rules["R1"]],
    )
    if "R1" in wind_fired:
        return _no_go(drones, [f"R1: {wind_facts['reasons'][-1]}"], {})

    reachable = dfs_reachable(blocked, world.base)
    accepted: list[Target] = []
    rejected: dict[int, str] = {}
    reasons: list[str] = []
    for target in targets:
        verdict_bits = _screen_target(target, blocked, reachable, rules)
        if verdict_bits is None:
            accepted.append(target)
            continue
        rejected[target.id] = verdict_bits
        reasons.append(f"target {target.id}: {verdict_bits}")

    if not accepted:
        return _finish_no_go(drones, reasons, rejected, rules)

    average = avg_cell_cost(world, coef, rng, samples=cfg.csp_samples)
    estimates = {
        target.id: estimate_target_cost(world.base, target, average) for target in accepted
    }
    scale = cfg.csp_energy_scale
    re_runs = 0
    steps = 0
    step_limit = (len(targets) + 1) * (cfg.csp_verify_retries + 2)
    drones_by_id = sorted(drones, key=lambda drone: drone.id)

    while steps <= step_limit:
        steps += 1
        if not accepted:
            return _finish_no_go(drones, reasons, rejected, rules)
        assignment, _stats = csp_assign(
            accepted,
            drones_by_id,
            estimates,
            energy_scale=scale,
            reserve=cfg.reserve,
        )
        if assignment is None:
            victim, why = _drop_for_failed_assignment(
                accepted, drones_by_id, estimates, scale, cfg.reserve, world.base
            )
            _drop(accepted, rejected, reasons, victim, why)
            scale = cfg.csp_energy_scale
            re_runs = 0
            continue

        built = {
            drone.id: _plan_drone(
                drone, accepted, assignment, world, blocked, coef, cfg, rng
            )
            for drone in drones_by_id
        }
        overloaded = [
            drone
            for drone in drones_by_id
            if built[drone.id]["energy"] > _usable(drone, cfg) + _COST_TOLERANCE
        ]
        if not overloaded:
            return _go_plan(drones_by_id, assignment, built, reasons, rejected)
        if re_runs < cfg.csp_verify_retries:
            scale *= cfg.csp_scale_shrink
            re_runs += 1
            drone_ids = ", ".join(str(drone.id) for drone in overloaded)
            reasons.append(
                "R4: estimated energy above planning limit for "
                f"drone {drone_ids}; retry assignment at energy_scale {scale:.2f}"
            )
            continue
        victim = _farthest(
            [target for target in accepted if assignment[target.id] in {drone.id for drone in overloaded}],
            world.base,
        )
        _drop(
            accepted,
            rejected,
            reasons,
            victim,
            "tour energy stayed above the battery limit after verification retries, "
            "so the farthest target was dropped",
        )
        scale = cfg.csp_energy_scale
        re_runs = 0

    raise RuntimeError("planning verification did not finish")


def _go_plan(
    drones: list[Drone],
    assignment: dict[int, int],
    built: dict[int, dict],
    reasons: list[str],
    rejected: dict[int, str],
) -> Plan:
    order = {drone.id: [] for drone in drones}
    routes = {drone.id: [] for drone in drones}
    energy = {drone.id: 0.0 for drone in drones}
    costs: dict[int, dict[str, float]] = {}
    for drone in drones:
        item = built[drone.id]
        order[drone.id] = item["order"]
        routes[drone.id] = item["route"]
        energy[drone.id] = 0.0 if not item["order"] else item["energy"]
        costs[drone.id] = {"hill_climb": item["hc_cost"], "genetic": item["ga_cost"]}
        if not item["order"]:
            reasons.append(f"drone {drone.id}: no targets, stays parked")
            continue
        reasons.append(
            f"drone {drone.id}: kept {item['chosen']} at {item['energy']:.2f} EU "
            f"(hill climb {item['hc_cost']:.2f}, genetic {item['ga_cost']:.2f})"
        )
    return Plan(
        verdict="GO",
        reasons=reasons,
        rejected=rejected,
        assignment=dict(assignment),
        order=order,
        routes=routes,
        est_energy=energy,
        order_costs=costs,
    )


def _plan_drone(
    drone: Drone,
    accepted: list[Target],
    assignment: dict[int, int],
    world: World,
    blocked: np.ndarray,
    coef: EnergyCoef,
    cfg: Config,
    rng: np.random.Generator,
) -> dict:
    assigned = sorted(
        (target for target in accepted if assignment[target.id] == drone.id),
        key=lambda target: target.id,
    )
    if not assigned:
        return {
            "order": [],
            "route": [],
            "energy": 0.0,
            "hc_cost": 0.0,
            "ga_cost": 0.0,
            "chosen": "unused",
        }
    start = drone.pos if drone.mode in _AIRBORNE else world.base
    payload = float(sum(target.mass_kg for target in assigned))
    max_wind = _wind_speed(world)
    matrix, paths = cost_matrix(
        world,
        blocked,
        coef,
        [start, *[target.pos for target in assigned]],
        payload,
        max_wind,
    )
    if start != world.base:
        for index, target in enumerate(assigned, start=1):
            home = astar(
                world, blocked, target.pos, world.base, coef, payload, max_wind
            )
            matrix[index, 0] = home.cost
            paths[(index, 0)] = home.path
    stops = list(range(1, len(assigned) + 1))
    hc_order, hc_cost, _hc_history = hill_climb(
        matrix, stops, restarts=cfg.hc_restarts, rng=rng
    )
    ga_order, ga_cost, _ga_history = genetic_order(
        matrix,
        stops,
        pop=cfg.ga_pop,
        gens=cfg.ga_gens,
        pc=cfg.ga_pc,
        pm=cfg.ga_pm,
        tournament=cfg.ga_tournament,
        elite=cfg.ga_elite,
        battery_limit=_usable(drone, cfg),
        rng=rng,
    )
    if ga_cost < hc_cost:
        chosen_order, chosen_cost, chosen_name = ga_order, ga_cost, "genetic"
    else:
        chosen_order, chosen_cost, chosen_name = hc_order, hc_cost, "hill_climb"
    route = _stitch(paths, chosen_order, start)
    if not math.isfinite(chosen_cost):
        route = []
        chosen_cost = math.inf
    return {
        "order": [assigned[index - 1].id for index in chosen_order],
        "route": route,
        "energy": float(tour_cost(chosen_order, matrix)),
        "hc_cost": float(hc_cost),
        "ga_cost": float(ga_cost),
        "chosen": chosen_name,
    }


def _stitch(paths: dict[tuple[int, int], list[Cell]], order: list[int], start: Cell) -> list[Cell]:
    """Join directed legs, dropping the repeated cell where two legs meet."""
    route: list[Cell] = []
    sequence = [0, *order, 0]
    for left, right in zip(sequence, sequence[1:]):
        leg = paths[(left, right)]
        if not leg:
            continue
        route.extend(leg if not route else leg[1:])
    if not route and order:
        return [start]
    return route


def _screen_target(
    target: Target,
    blocked: np.ndarray,
    reachable: np.ndarray,
    rules: dict,
) -> str | None:
    facts, fired = forward_chain(
        {
            "target_restricted": bool(blocked[target.pos]),
            "target_reachable": bool(reachable[target.pos]),
        },
        [rules["R2"], rules["R3"]],
    )
    if "R2" not in fired and "R3" not in fired:
        return None
    texts = facts.get("reasons", [])
    return "; ".join(f"{rule_id}: {text}" for rule_id, text in zip(fired, texts))


def _drop_for_failed_assignment(
    accepted: list[Target],
    drones: list[Drone],
    estimates: dict[int, float],
    scale: float,
    reserve: float,
    base: Cell,
) -> tuple[Target, str]:
    infeasible = [
        target
        for target in accepted
        if not any(_drone_can_take(drone, target, estimates, scale, reserve) for drone in drones)
    ]
    if infeasible:
        return (
            _farthest(infeasible, base),
            "fits no drone at the current mass and energy limits",
        )
    return (
        _farthest(accepted, base),
        "assignment failed, so the farthest target was dropped",
    )


def _drone_can_take(
    drone: Drone,
    target: Target,
    estimates: dict[int, float],
    scale: float,
    reserve: float,
) -> bool:
    mass_ok = target.mass_kg <= drone.capacity_kg
    energy_ok = estimates[target.id] <= drone.battery * (1.0 - reserve) * scale
    return mass_ok and energy_ok


def _drop(
    accepted: list[Target],
    rejected: dict[int, str],
    reasons: list[str],
    target: Target,
    why: str,
) -> None:
    accepted.remove(target)
    rejected[target.id] = why
    reasons.append(f"target {target.id}: {why}")


def _farthest(targets: list[Target], base: Cell) -> Target:
    return max(targets, key=lambda target: (euclid(base, target.pos), target.id))


def _finish_no_go(
    drones: list[Drone],
    reasons: list[str],
    rejected: dict[int, str],
    rules: dict,
) -> Plan:
    facts, fired = forward_chain({"remaining_targets": 0}, [rules["R7"]])
    if "R7" in fired:
        reasons.append(f"R7: {facts['reasons'][-1]}")
    return _no_go(drones, reasons, rejected)


def _no_go(drones: list[Drone], reasons: list[str], rejected: dict[int, str]) -> Plan:
    order, routes, energy = _blank(drones)
    return Plan(
        verdict="NO-GO",
        reasons=reasons,
        rejected=rejected,
        assignment={},
        order=order,
        routes=routes,
        est_energy=energy,
    )


def _blank(drones: list[Drone]) -> tuple[dict[int, list[int]], dict[int, list[Cell]], dict[int, float]]:
    return (
        {drone.id: [] for drone in drones},
        {drone.id: [] for drone in drones},
        {drone.id: 0.0 for drone in drones},
    )


def _blocked_mask(world: World, cfg: Config, blocked_extra: np.ndarray | None) -> np.ndarray:
    blocked = static_restricted(world, cfg)
    if blocked_extra is None:
        return blocked
    extra = np.asarray(blocked_extra, dtype=bool)
    if extra.shape != blocked.shape:
        raise ValueError("blocked_extra must match the world grid")
    return blocked | extra


def _usable(drone: Drone, cfg: Config) -> float:
    return float(drone.battery * (1.0 - cfg.reserve))


def _wind_speed(world: World) -> float:
    return float(math.hypot(world.wind[0], world.wind[1]))


def _validate(
    world: World,
    drones: list[Drone],
    targets: list[Target],
    cfg: Config,
    blocked_extra: np.ndarray | None,
) -> None:
    if world.terrain.shape != world.no_fly.shape or world.terrain.ndim != 2:
        raise ValueError("terrain and no_fly must be matching 2-D grids")
    if not drones:
        raise ValueError("at least one drone is required")
    if len({drone.id for drone in drones}) != len(drones):
        raise ValueError("drone ids must be unique")
    if len({target.id for target in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    if cfg.csp_verify_retries < 0:
        raise ValueError("csp_verify_retries must be non-negative")
    if not 0.0 < cfg.csp_scale_shrink < 1.0:
        raise ValueError("csp_scale_shrink must be in (0, 1)")
    n = int(world.terrain.shape[0])
    for target in targets:
        row, col = target.pos
        if not (0 <= row < n and 0 <= col < n):
            raise ValueError(f"target {target.id} is outside the grid")
    if blocked_extra is not None and np.asarray(blocked_extra).shape != world.terrain.shape:
        raise ValueError("blocked_extra must match the world grid")
