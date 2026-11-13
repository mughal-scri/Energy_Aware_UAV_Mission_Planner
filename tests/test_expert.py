"""Expert-system rule and restricted-cell tests."""

import numpy as np
import pytest

from uavplanner.config import Config
from uavplanner.expert import (
    Rule,
    default_rules,
    dynamic_restricted,
    forward_chain,
    static_restricted,
)
from uavplanner.models import World
from uavplanner.uninformed import dfs_reachable
from uavplanner.world import make_world


def test_r1_wind_limit_fires_at_14_ms_only() -> None:
    cfg = Config(wind_limit=12.0)
    rules = default_rules(cfg)

    # 14 m/s: above safe limit, must trigger R1 and NO-GO
    facts_high = {"wind_speed": 14.0}
    result_high, fired_high = forward_chain(facts_high, rules)
    assert "R1" in fired_high
    assert result_high["verdict"] == "NO-GO"
    assert "wind above safe limit" in result_high["reasons"]

    # 6 m/s: safe, R1 must not fire
    facts_safe = {"wind_speed": 6.0}
    result_safe, fired_safe = forward_chain(facts_safe, rules)
    assert "R1" not in fired_safe
    assert result_safe.get("verdict") is None

    # 12 m/s: exact boundary, R1 must not fire
    facts_bound = {"wind_speed": 12.0}
    result_bound, fired_bound = forward_chain(facts_bound, rules)
    assert "R1" not in fired_bound
    assert result_bound.get("verdict") is None


def test_r2_and_r3_target_rejection() -> None:
    rules = default_rules()

    # R2: target inside restricted cell
    facts_r2 = {"target_restricted": True, "target_reachable": True}
    res_r2, fired_r2 = forward_chain(facts_r2, rules)
    assert "R2" in fired_r2
    assert "R3" not in fired_r2
    assert res_r2["target_rejected"] is True
    assert "target inside restricted cell" in res_r2["reasons"]

    # R3: target unreachable from base
    facts_r3 = {"target_restricted": False, "target_reachable": False}
    res_r3, fired_r3 = forward_chain(facts_r3, rules)
    assert "R3" in fired_r3
    assert "R2" not in fired_r3
    assert res_r3["target_rejected"] is True
    assert "target not reachable from base" in res_r3["reasons"]

    # Valid target: neither R2 nor R3 fires
    facts_ok = {"target_restricted": False, "target_reachable": True}
    res_ok, fired_ok = forward_chain(facts_ok, rules)
    assert "R2" not in fired_ok
    assert "R3" not in fired_ok
    assert "target_rejected" not in res_ok


def test_fired_list_explains_each_verdict() -> None:
    rules = default_rules()

    # When all targets are rejected, R7 marks mission NO-GO
    facts = {"remaining_targets": 0}
    res, fired = forward_chain(facts, rules)
    assert fired == ["R7"]
    assert res["verdict"] == "NO-GO"
    assert res["reasons"] == ["no targets remain"]

    # R5 (battery low) and R6 (interceptor near) firing
    facts_inflight = {
        "battery_remaining": 15.0,
        "battery_capacity": 100.0,
        "reserve": 0.2,
        "interceptor_distance": 3,
    }
    res_inflight, fired_inflight = forward_chain(facts_inflight, rules)
    assert "R5" in fired_inflight
    assert "R6" in fired_inflight
    assert res_inflight["action"] == "RETURN-TO-BASE"
    assert res_inflight["mode"] == "EVADE"


def test_static_restricted_mask() -> None:
    cfg = Config(grid_n=10, max_climb_m=40.0, restricted_buffer=1)
    terrain = np.zeros((10, 10), dtype=float)
    no_fly = np.zeros((10, 10), dtype=bool)

    # Place a no-fly cell at (2, 2)
    no_fly[2, 2] = True
    # Place a steep elevation peak at (6, 6) climbing 50m above its neighbours (0m)
    terrain[6, 6] = 50.0

    world = World(terrain=terrain, no_fly=no_fly, wind=(0.0, 0.0), base=(0, 0))
    mask = static_restricted(world, cfg)

    # No-fly cell (2, 2) and buffer cell (2, 3) must be restricted
    assert mask[2, 2] is np.True_ or mask[2, 2] == True
    assert mask[2, 3] is np.True_ or mask[2, 3] == True

    # Steep peak (6, 6) must be restricted
    assert mask[6, 6] is np.True_ or mask[6, 6] == True

    # Flat unbuffered cell (0, 0) must be free
    assert mask[0, 0] is np.False_ or mask[0, 0] == False


def test_dynamic_restricted_mask() -> None:
    cfg = Config(grid_n=10, interceptor_zone=2)
    world = World(
        terrain=np.zeros((10, 10)),
        no_fly=np.zeros((10, 10), dtype=bool),
        wind=(0.0, 0.0),
        base=(0, 0),
    )

    mask = dynamic_restricted(
        world=world,
        interceptor_last_seen=(5, 5),
        other_drone_cells=[(1, 1), (8, 8)],
        cfg=cfg,
    )

    # Interceptor zone within Chebyshev distance 2
    assert mask[5, 5] == True
    assert mask[3, 3] == True
    assert mask[7, 7] == True
    assert mask[2, 5] == False

    # Other drone cells
    assert mask[1, 1] == True
    assert mask[8, 8] == True
    assert mask[0, 0] == False


def test_forward_chain_uniqueness_and_termination() -> None:
    rule1 = Rule("DUP", "Rule 1", lambda f: True, lambda f: f.setdefault("x", 1))
    rule2 = Rule("DUP", "Rule 2", lambda f: True, lambda f: f.setdefault("x", 2))
    with pytest.raises(ValueError, match="unique"):
        forward_chain({}, [rule1, rule2])
