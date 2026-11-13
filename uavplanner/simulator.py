"""Tick-based mission execution with traffic and interceptor evasion."""

from __future__ import annotations

import math

import numpy as np

from uavplanner.config import Config
from uavplanner.console import Console
from uavplanner.adversarial import choose_move
from uavplanner.energy import headwind, move_energy, true_energy
from uavplanner.expert import dynamic_restricted, static_restricted
from uavplanner.informed import astar
from uavplanner.interceptor import step_interceptor
from uavplanner.models import Drone, EnergyCoef, Interceptor, PerceptRow, Target, World
from uavplanner.planner import Plan
from uavplanner.sensor import look
from uavplanner.traffic import resolve, takeoff_tick
from uavplanner.world import chebyshev

_AIRBORNE = {"TRANSIT", "HOVER", "WAIT", "EVADE", "RETURN"}


def run_mission(
    world: World,
    plan: Plan,
    drones: list[Drone],
    interceptor: Interceptor | None,
    targets: list[Target],
    true_coef: EnergyCoef,
    model_coef: EnergyCoef,
    cfg: Config,
    rng: np.random.Generator,
    console: Console,
    percepts: list[PerceptRow],
) -> dict[str, int | float | list[int]]:
    """Fly planned routes, confirm targets, evade threats, and collect metrics."""
    _validate(world, plan, drones, targets, cfg)
    if plan.verdict != "GO":
        for drone in drones:
            console.log(0, drone.id, drone.pos, _battery_pct(drone), drone.mode, "NO-GO")
        summary = _summary(drones, targets, percepts, 0, 0)
        summary["total_energy"] = 0.0
        summary["minimax_nodes"] = 0
        return summary

    by_target = {target.id: target for target in targets}
    route_index = {drone.id: 0 for drone in drones}
    stop_index = {drone.id: 0 for drone in drones}
    initial_battery = {drone.id: drone.battery for drone in drones}
    event_by_drone = {drone.id: "" for drone in drones}
    returning_ids: set[int] = set()
    collisions = 0
    waiting_ticks = 0
    minimax_nodes = 0

    for tick in range(cfg.max_ticks):
        event_by_drone = {drone.id: "" for drone in drones}
        for drone in drones:
            if drone.mode == "PARKED" and tick >= takeoff_tick(drone.id, cfg.takeoff_stagger):
                if plan.order.get(drone.id):
                    drone.route = list(plan.routes.get(drone.id, []))
                    drone.pos = world.base
                    drone.mode = "TRANSIT"
                    route_index[drone.id] = 1 if drone.route and drone.route[0] == drone.pos else 0
                    event_by_drone[drone.id] = "takeoff"
                else:
                    drone.mode = "LANDED"

        active = [drone for drone in drones if drone.mode in _AIRBORNE]
        if not active:
            break

        positions = {drone.id: drone.pos for drone in active}
        proposals: dict[int, tuple[int, int]] = {}
        for drone in sorted(active, key=lambda item: item.id):
            assigned_ids = plan.order.get(drone.id, [])
            next_target = by_target.get(assigned_ids[stop_index[drone.id]]) if stop_index[drone.id] < len(assigned_ids) else None

            evading = False
            if interceptor is not None:
                threat_distance = chebyshev(drone.pos, interceptor.pos)
                if drone.mode == "EVADE":
                    if threat_distance > cfg.minimax_release:
                        stops_left = (
                            [] if drone.id in returning_ids
                            else assigned_ids[stop_index[drone.id]:]
                        )
                        replanned = _route_from_stops(
                            world, drone.pos, stops_left,
                            by_target, interceptor, [other.pos for other in active if other.id != drone.id],
                            model_coef, cfg,
                        )
                        if replanned is None:
                            drone.mode = "LOST"
                            event_by_drone[drone.id] = _join_event(
                                event_by_drone[drone.id], "R9: no clear-threat route"
                            )
                            continue
                        drone.route = replanned
                        route_index[drone.id] = 1 if replanned and replanned[0] == drone.pos else 0
                        drone.mode = "TRANSIT" if stops_left else "RETURN"
                        event_by_drone[drone.id] = _join_event(
                            event_by_drone[drone.id], "R6 clear: A* route replanned"
                        )
                    else:
                        evading = True
                elif threat_distance <= cfg.minimax_trigger:
                    drone.mode = "EVADE"
                    evading = True
                    event_by_drone[drone.id] = _join_event(event_by_drone[drone.id], "R6: EVADE")

            if evading and interceptor is not None:
                home_cost = _home_cost(world, drone, model_coef, cfg)
                return_trigger = max(
                    initial_battery[drone.id] * cfg.reserve, 1.1 * home_cost
                )
                if drone.battery <= return_trigger:
                    returning_ids.add(drone.id)
                goal = _evasion_waypoint(drone.pos, interceptor.pos, world.terrain.shape)
                blocked = static_restricted(world, cfg)
                blocked |= dynamic_restricted(world, interceptor.last_seen, cfg=cfg)
                blocked[drone.pos] = False
                blocked[goal] = False
                move, _value, visited = choose_move(
                    drone.pos,
                    interceptor.pos,
                    goal,
                    blocked,
                    cfg=cfg,
                )
                minimax_nodes += visited
                proposals[drone.id] = move
                continue

            if drone.mode == "HOVER":
                if next_target is None:
                    drone.mode = "TRANSIT"
                    drone.looks = 0
                    continue
                found = look(drone.pos, next_target.pos, rng)
                drone.looks += 1
                if found:
                    next_target.status = "confirmed"
                    event_by_drone[drone.id] = f"target {next_target.id} confirmed"
                    stop_index[drone.id] += 1
                    drone.looks = 0
                    drone.mode = "TRANSIT"
                    next_target = None
                elif drone.looks >= cfg.sensor_max_looks:
                    next_target.status = "unconfirmed"
                    event_by_drone[drone.id] = f"target {next_target.id} unconfirmed"
                    stop_index[drone.id] += 1
                    drone.looks = 0
                    drone.mode = "TRANSIT"
                    next_target = None
                else:
                    drone.battery = max(0.0, drone.battery - cfg.hover_energy)
                    if drone.battery <= 0.0:
                        drone.mode = "LOST"
                    proposals[drone.id] = drone.pos
                    continue

            if drone.mode == "LOST":
                continue
            home_cost = _home_cost(world, drone, model_coef, cfg)
            reserve_floor = initial_battery[drone.id] * cfg.reserve
            return_trigger = max(reserve_floor, 1.1 * home_cost)
            if drone.mode != "RETURN" and drone.battery <= return_trigger:
                _set_return_route(world, drone, model_coef, cfg)
                route_index[drone.id] = 0
                event_by_drone[drone.id] = "R5: return to base"
                if drone.mode in {"LANDED", "LOST"}:
                    continue

            if drone.mode == "RETURN" and drone.pos == world.base:
                drone.mode = "LANDED"
                event_by_drone[drone.id] = _join_event(event_by_drone[drone.id], "landed")
                continue

            if next_target is not None and drone.pos == next_target.pos and drone.mode != "RETURN":
                drone.mode = "HOVER"
                drone.looks = 0
                drone.battery = max(0.0, drone.battery - cfg.hover_energy)
                if drone.battery <= 0.0:
                    drone.mode = "LOST"
                    continue
                proposals[drone.id] = drone.pos
                continue

            if drone.mode == "TRANSIT" and stop_index[drone.id] >= len(assigned_ids):
                if drone.pos == world.base:
                    drone.mode = "LANDED"
                    event_by_drone[drone.id] = "landed"
                    continue
                _set_return_route(world, drone, model_coef, cfg)
                route_index[drone.id] = 0
                drone.mode = "RETURN"

            route = drone.route
            index = route_index[drone.id]
            while index < len(route) and route[index] == drone.pos:
                index += 1
            route_index[drone.id] = index
            proposals[drone.id] = route[index] if index < len(route) else drone.pos

        if not proposals:
            continue
        active_positions = {drone.id: drone.pos for drone in drones if drone.id in proposals}
        approved = resolve(active_positions, proposals)
        if len(set(approved.values())) != len(approved):
            collisions += 1
        for drone in sorted((d for d in drones if d.id in approved), key=lambda item: item.id):
            wanted = approved[drone.id]
            if wanted == drone.pos:
                if drone.mode != "HOVER":
                    waiting_ticks += 1
                    drone.waited += 1
                    drone.mode = "WAIT" if drone.mode not in {"RETURN", "EVADE"} else drone.mode
                    drone.battery = max(0.0, drone.battery - cfg.hover_energy)
                if drone.battery <= 0.0:
                    drone.mode = "LOST"
                continue
            previous = drone.pos
            dr, dc = wanted[0] - previous[0], wanted[1] - previous[1]
            distance = cfg.diag_step if dr and dc else 1.0
            climb = float(world.terrain[wanted] - world.terrain[previous])
            wind_cost = headwind(world.wind, (dr, dc))
            payload = _payload(drone, plan, by_target, stop_index[drone.id])
            predicted = move_energy(model_coef, distance, climb, wind_cost, payload, floor=cfg.energy_floor)
            actual = true_energy(
                true_coef, distance, climb, wind_cost, payload, rng,
                floor=cfg.energy_floor, noise_sd=cfg.noise_sd, noise_clip=cfg.noise_clip,
            )
            percepts.append(PerceptRow(1, drone.id, distance, climb, wind_cost, payload, predicted, actual))
            drone.battery = max(0.0, drone.battery - actual)
            drone.pos = wanted
            drone.waited = 0
            if drone.mode == "WAIT":
                drone.mode = "TRANSIT"
            if drone.mode != "EVADE":
                route_index[drone.id] += 1
            if drone.battery <= 0.0:
                drone.mode = "LOST"
                event_by_drone[drone.id] = "battery depleted"
            elif drone.mode == "RETURN" and drone.pos == world.base:
                drone.mode = "LANDED"
                event_by_drone[drone.id] = _join_event(event_by_drone[drone.id], "landed")

        if interceptor is not None:
            participants = [drone for drone in drones if drone.mode in _AIRBORNE]
            captured = step_interceptor(interceptor, [drone.pos for drone in participants], cfg)
            for index in captured:
                caught = participants[index]
                caught.mode = "LOST"
                event_by_drone[caught.id] = _join_event(
                    event_by_drone[caught.id], "captured by interceptor"
                )

        for drone in drones:
            if drone.mode in _AIRBORNE or event_by_drone[drone.id]:
                console.log(tick, drone.id, drone.pos,
                            _battery_pct(drone, initial_battery[drone.id]),
                            drone.mode, event_by_drone[drone.id])

    completed = sum(1 for drone in drones if drone.mode == "LANDED")
    total_energy = float(sum(initial_battery[d.id] - d.battery for d in drones))
    confirmed = sum(target.status == "confirmed" for target in targets)
    unconfirmed = sum(target.status == "unconfirmed" for target in targets)
    lost = [drone.id for drone in drones if drone.mode == "LOST"]
    return {
        "missions_completed": completed,
        "targets_confirmed": confirmed,
        "targets_unconfirmed": unconfirmed,
        "total_energy": total_energy,
        "collisions": collisions,
        "drones_lost": len(lost),
        "lost_drone_ids": lost,
        "waiting_ticks": waiting_ticks,
        "ticks": max((row.tick for row in console.rows()), default=-1) + 1,
        "percepts": len(percepts),
        "minimax_nodes": minimax_nodes,
    }


def _validate(world: World, plan: Plan, drones: list[Drone], targets: list[Target], cfg: Config) -> None:
    if world.terrain.shape != world.no_fly.shape or world.terrain.ndim != 2:
        raise ValueError("terrain and no_fly must be matching 2-D grids")
    if cfg.max_ticks < 1 or cfg.sensor_max_looks < 1:
        raise ValueError("max_ticks and sensor_max_looks must be positive")
    if len({drone.id for drone in drones}) != len(drones):
        raise ValueError("drone ids must be unique")
    if len({target.id for target in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    n = world.terrain.shape[0]
    if any(not (0 <= r < n and 0 <= c < n) for r, c in [world.base, *(d.pos for d in drones)]):
        raise ValueError("base and drone positions must be inside the world")
    for drone_id, route in plan.routes.items():
        if drone_id not in {drone.id for drone in drones}:
            raise ValueError(f"plan route references unknown drone {drone_id}")
        if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1 for a, b in zip(route, route[1:])):
            raise ValueError(f"route for drone {drone_id} contains a disconnected step")


def _home_cost(world: World, drone: Drone, coef: EnergyCoef, cfg: Config) -> float:
    if drone.pos == world.base:
        return 0.0
    blocked = static_restricted(world, cfg)
    blocked[drone.pos] = False
    blocked[world.base] = False
    result = astar(world, blocked, drone.pos, world.base, coef, 0.0,
                   math.hypot(*world.wind), floor=cfg.energy_floor, diag_step=cfg.diag_step)
    return result.cost


def _set_return_route(world: World, drone: Drone, coef: EnergyCoef, cfg: Config) -> None:
    blocked = static_restricted(world, cfg)
    blocked[drone.pos] = False
    blocked[world.base] = False
    result = astar(world, blocked, drone.pos, world.base, coef, 0.0,
                   math.hypot(*world.wind), floor=cfg.energy_floor, diag_step=cfg.diag_step)
    if result.path:
        drone.route = result.path
        drone.mode = "RETURN"
    elif drone.pos == world.base:
        drone.route = []
        drone.mode = "LANDED"
    else:
        drone.mode = "LOST"


def _route_from_stops(
    world: World,
    start: tuple[int, int],
    stop_ids: list[int],
    targets: dict[int, Target],
    interceptor: Interceptor,
    other_drone_cells: list[tuple[int, int]],
    coef: EnergyCoef,
    cfg: Config,
) -> list[tuple[int, int]] | None:
    """A* replan through remaining stops and back to base after a threat clears."""
    remaining = [targets[target_id] for target_id in stop_ids if target_id in targets]
    route = [start]
    current = start
    max_wind = math.hypot(*world.wind)
    for index, target in enumerate(remaining):
        blocked = static_restricted(world, cfg)
        blocked |= dynamic_restricted(
            world, interceptor.last_seen, other_drone_cells, cfg=cfg
        )
        blocked[current] = False
        blocked[target.pos] = False
        payload = sum(item.mass_kg for item in remaining[index:])
        leg = astar(
            world, blocked, current, target.pos, coef, payload, max_wind,
            floor=cfg.energy_floor, diag_step=cfg.diag_step,
        )
        if not leg.path and current != target.pos:
            return None
        route.extend(leg.path[1:] if len(leg.path) > 1 else [])
        current = target.pos

    blocked = static_restricted(world, cfg)
    blocked |= dynamic_restricted(
        world, interceptor.last_seen, other_drone_cells, cfg=cfg
    )
    blocked[current] = False
    blocked[world.base] = False
    home = astar(
        world, blocked, current, world.base, coef, 0.0, max_wind,
        floor=cfg.energy_floor, diag_step=cfg.diag_step,
    )
    if current != world.base and not home.path:
        return None
    if len(home.path) > 1:
        route.extend(home.path[1:])
    return route


def _evasion_waypoint(
    drone_pos: tuple[int, int], interceptor_pos: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    """Pick the grid corner farthest from the threat as a temporary flee goal."""
    rows, columns = shape
    corners = ((0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1))
    return max(corners, key=lambda cell: (chebyshev(cell, interceptor_pos), -chebyshev(cell, drone_pos)))


def _payload(drone: Drone, plan: Plan, targets: dict[int, Target], next_index: int) -> float:
    return float(sum(targets[target_id].mass_kg for target_id in plan.order.get(drone.id, [])[next_index:]
                     if target_id in targets))


def _battery_pct(drone: Drone, initial: float | None = None) -> float:
    capacity = drone.battery if initial is None else initial
    return 0.0 if capacity <= 0.0 else min(100.0, 100.0 * drone.battery / capacity)


def _join_event(existing: str, event: str) -> str:
    """Keep same-tick events visible when another event also occurs."""
    return f"{existing}; {event}" if existing else event


def _summary(drones: list[Drone], targets: list[Target], percepts: list[PerceptRow], collisions: int,
             waiting_ticks: int) -> dict[str, int | float | list[int]]:
    return {
        "missions_completed": sum(drone.mode == "LANDED" for drone in drones),
        "targets_confirmed": sum(target.status == "confirmed" for target in targets),
        "targets_unconfirmed": sum(target.status == "unconfirmed" for target in targets),
        "total_energy": float(sum(max(0.0, drone.battery) for drone in drones)),
        "collisions": collisions,
        "drones_lost": sum(drone.mode == "LOST" for drone in drones),
        "lost_drone_ids": [drone.id for drone in drones if drone.mode == "LOST"],
        "waiting_ticks": waiting_ticks,
        "ticks": 0,
        "percepts": len(percepts),
    }
