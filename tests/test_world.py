"""World generation and neighbour geometry."""

import numpy as np

from uavplanner.config import Config
from uavplanner.world import (
    chebyshev,
    load_scenario,
    make_targets,
    make_world,
    neighbors,
    nofly_buffer_mask,
    save_scenario,
)


def test_same_seed_same_world() -> None:
    cfg = Config(seed=42)
    a = make_world(cfg, np.random.default_rng(42))
    b = make_world(cfg, np.random.default_rng(42))
    assert np.allclose(a.terrain, b.terrain)
    assert np.array_equal(a.no_fly, b.no_fly)
    assert a.base == b.base
    assert a.wind == b.wind


def test_base_not_in_nofly_and_targets_separated() -> None:
    cfg = Config(seed=7)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    assert not world.no_fly[world.base]
    assert cfg.n_targets_min <= len(targets) <= cfg.n_targets_max
    for i, t in enumerate(targets):
        for u in targets[i + 1 :]:
            if world.no_fly[t.pos] or world.no_fly[u.pos]:
                continue
            assert chebyshev(t.pos, u.pos) >= cfg.target_min_separation


def test_neighbors_stay_inside_grid() -> None:
    n = 5
    cells = list(neighbors((0, 0), n))
    assert all(0 <= r < n and 0 <= c < n for (r, c), _d in cells)
    assert {(cell, round(d, 3)) for cell, d in cells} == {
        ((0, 1), 1.0),
        ((1, 0), 1.0),
        ((1, 1), 1.414),
    }


def test_demo_rejections_creates_enclosed_and_nofly_targets() -> None:
    cfg = Config(seed=11, demo_rejections=True)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    n = world.terrain.shape[0]
    enclosed = [
        t
        for t in targets
        if not world.no_fly[t.pos]
        and all(world.no_fly[nb] for nb, _ in neighbors(t.pos, n))
    ]
    inside = [t for t in targets if world.no_fly[t.pos]]
    assert enclosed
    assert inside


def test_nofly_buffer_is_chebyshev() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    dilated = nofly_buffer_mask(mask, 1)
    assert dilated[3, 3] and dilated[2, 2] and dilated[4, 4]
    assert not dilated[1, 1]


def test_save_load_roundtrip(tmp_path) -> None:
    cfg = Config(seed=3)
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    save_scenario(tmp_path, "demo", world, targets, cfg)
    loaded_world, loaded_targets = load_scenario(tmp_path, "demo")
    assert np.allclose(loaded_world.terrain, world.terrain)
    assert loaded_world.base == world.base
    assert [t.pos for t in loaded_targets] == [t.pos for t in targets]
