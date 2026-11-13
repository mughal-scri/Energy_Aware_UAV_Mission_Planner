"""Forward-chaining safety rules and restricted-cell masks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from uavplanner.config import Config
from uavplanner.models import Cell, World
from uavplanner.world import chebyshev, neighbors, nofly_buffer_mask


@dataclass(frozen=True)
class Rule:
    """One explainable IF/THEN rule used by ``forward_chain``."""

    id: str
    text: str
    cond: Callable[[dict], bool]
    act: Callable[[dict], None]


def static_restricted(world: World, cfg: Config) -> np.ndarray:
    """Return no-fly-buffer and excessive-climb cells blocked for all planning.

    A cell is too steep when entering it from at least one free neighbouring
    cell would require a climb greater than ``cfg.max_climb_m``.
    """
    restricted = nofly_buffer_mask(world.no_fly, cfg.restricted_buffer)
    n = int(world.terrain.shape[0])
    for row in range(n):
        for col in range(n):
            cell = (row, col)
            if restricted[cell]:
                continue
            for neighbour, _distance in neighbors(cell, n, diag_step=cfg.diag_step):
                if world.no_fly[neighbour]:
                    continue
                climb = float(world.terrain[cell] - world.terrain[neighbour])
                if climb > cfg.max_climb_m:
                    restricted[cell] = True
                    break
    return restricted


def dynamic_restricted(
    world: World,
    interceptor_last_seen: Cell | None,
    other_drone_cells: Iterable[Cell] = (),
    cfg: Config | None = None,
) -> np.ndarray:
    """Return current interceptor-zone and other-drone replan restrictions."""
    if cfg is None:
        cfg = Config()
    n = int(world.terrain.shape[0])
    restricted = np.zeros((n, n), dtype=bool)
    if interceptor_last_seen is not None:
        for row in range(n):
            for col in range(n):
                if chebyshev((row, col), interceptor_last_seen) <= cfg.interceptor_zone:
                    restricted[row, col] = True
    for row, col in other_drone_cells:
        if 0 <= row < n and 0 <= col < n:
            restricted[row, col] = True
    return restricted


def forward_chain(facts: dict, rules: list[Rule]) -> tuple[dict, list[str]]:
    """Fire each matching rule once until no rule produces a new conclusion.

    The returned facts object is a shallow copy so callers retain their input.
    Rule ids must be unique because the firing list is also the console-ready
    explanation for a decision.
    """
    if len({rule.id for rule in rules}) != len(rules):
        raise ValueError("rule ids must be unique")
    inferred = dict(facts)
    fired: list[str] = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.id in fired or not rule.cond(inferred):
                continue
            rule.act(inferred)
            fired.append(rule.id)
            changed = True
    return inferred, fired


def default_rules(cfg: Config | None = None) -> list[Rule]:
    """Build R1--R7 and R10 using the documented facts-dictionary keys.

    Call pre-flight rules once per target with ``target_restricted`` and
    ``target_reachable`` facts. Call monitoring rules with the corresponding
    battery and interceptor facts. Actions write simple flags plus an
    explanatory ``reasons`` list, which later console code can display.
    """
    if cfg is None:
        cfg = Config()

    def add_reason(facts: dict, reason: str) -> None:
        facts.setdefault("reasons", []).append(reason)

    def reject_target(facts: dict, reason: str) -> None:
        facts["target_rejected"] = True
        add_reason(facts, reason)

    return [
        Rule(
            "R1",
            "IF wind exceeds the safe limit THEN mission is NO-GO.",
            lambda facts: facts.get("wind_speed", 0.0) > cfg.wind_limit,
            lambda facts: (facts.__setitem__("verdict", "NO-GO"), add_reason(facts, "wind above safe limit")),
        ),
        Rule(
            "R2",
            "IF a target is restricted THEN reject it.",
            lambda facts: bool(facts.get("target_restricted", False)),
            lambda facts: reject_target(facts, "target inside restricted cell"),
        ),
        Rule(
            "R3",
            "IF a target is unreachable THEN reject it.",
            lambda facts: facts.get("target_reachable", True) is False,
            lambda facts: reject_target(facts, "target not reachable from base"),
        ),
        Rule(
            "R4",
            "IF estimated energy exceeds the planning limit THEN mark overloaded.",
            lambda facts: facts.get("estimated_mission_energy", 0.0)
            > facts.get("battery", cfg.battery) * (1.0 - facts.get("reserve", cfg.reserve)),
            lambda facts: (facts.__setitem__("overloaded", True), add_reason(facts, "estimated energy above planning limit")),
        ),
        Rule(
            "R5",
            "IF battery is below reserve THEN return to base.",
            lambda facts: facts.get("battery_remaining", facts.get("battery", cfg.battery))
            < facts.get("battery_capacity", cfg.battery) * facts.get("reserve", cfg.reserve),
            lambda facts: (facts.__setitem__("action", "RETURN-TO-BASE"), add_reason(facts, "battery below reserve")),
        ),
        Rule(
            "R6",
            "IF interceptor is within the trigger range THEN evade.",
            lambda facts: facts.get("interceptor_distance", float("inf")) <= cfg.minimax_trigger,
            lambda facts: (facts.__setitem__("mode", "EVADE"), add_reason(facts, "interceptor within evasion range")),
        ),
        Rule(
            "R7",
            "IF no targets remain THEN mission is NO-GO.",
            lambda facts: facts.get("remaining_targets", 1) <= 0,
            lambda facts: (facts.__setitem__("verdict", "NO-GO"), add_reason(facts, "no targets remain")),
        ),
        Rule(
            "R10",
            "IF a cell is no-fly, steep, or threat-adjacent THEN mark it restricted.",
            lambda facts: any(
                bool(facts.get(key, False))
                for key in ("in_nofly_buffer", "too_steep", "near_interceptor")
            ),
            lambda facts: facts.__setitem__("restricted", True),
        ),
    ]
