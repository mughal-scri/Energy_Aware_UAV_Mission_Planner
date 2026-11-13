"""Patrol, chase, recapture, and range-loss behavior."""

from uavplanner.config import Config
from uavplanner.interceptor import step_interceptor
from uavplanner.models import Interceptor


def test_patrol_advances_waypoints_and_cycles() -> None:
    cfg = Config(grid_n=10, interceptor_detection=0)
    threat = Interceptor(pos=(1, 1), waypoints=[(1, 2), (2, 2)])
    assert step_interceptor(threat, [], cfg) == []
    assert threat.pos == (1, 2)
    assert threat.wp_index == 0
    step_interceptor(threat, [], cfg)
    assert threat.pos == (2, 2)
    assert threat.wp_index == 1
    step_interceptor(threat, [], cfg)
    assert threat.pos == (1, 2)
    assert threat.wp_index == 0


def test_interceptor_chases_captures_and_returns_to_patrol_after_lost_ticks() -> None:
    cfg = Config(grid_n=12, interceptor_detection=3, interceptor_capture=1, interceptor_lost_ticks=2)
    threat = Interceptor(pos=(5, 5), waypoints=[(0, 0)])
    captured = step_interceptor(threat, [(5, 7)], cfg)
    assert threat.mode == "CHASE"
    assert threat.last_seen == (5, 6)
    assert threat.target_last_seen == (5, 7)
    assert threat.pos == (5, 6)
    assert captured == [0]

    step_interceptor(threat, [(11, 11)], cfg)
    assert threat.mode == "CHASE"
    assert threat.lost_ticks == 1
    step_interceptor(threat, [(11, 11)], cfg)
    assert threat.mode == "PATROL"
    assert threat.last_seen is not None
    assert threat.target_last_seen is None
    assert threat.lost_ticks == 0
