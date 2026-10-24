"""PLOTS: Colours and sizes live in ``STYLE``"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from uavplanner.config import Config
from uavplanner.energy import path_energy
from uavplanner.models import Cell, EnergyCoef, SearchResult, Target, World
from uavplanner.world import nofly_buffer_mask

STYLE = {
    "figsize_world": (8.4, 7.2),
    "figsize_search": (10.5, 9.2),
    "figsize_compare": (10.0, 4.8),
    "dpi": 130,
    "font_size": 11,
    "title_size": 12,
    "label_size": 10,
    "cmap_terrain": "cividis",
    "cmap_explored": "YlGnBu",
    "contour_color": "#4d4d4d",
    "contour_width": 0.55,
    "contour_levels": 8,
    "nofly_edge": "#d55e00",
    "nofly_face": "#d55e00",
    "nofly_alpha": 0.22,
    "nofly_linewidth": 1.4,
    "base_color": "#0072b2",
    "base_marker": "s",
    "base_size": 70,
    "target_color": "#000000",
    "target_marker": "*",
    "target_size": 90,
    "wind_color": "#56b4e9",
    "wind_alpha": 0.85,
    "wind_stride": 6,
    "wind_scale": 2.5,
    "wind_width": 0.004,
    "restricted_cmap": "Reds",
    "restricted_alpha": 0.28,
    "path_color": "#e69f00",
    "path_style": "-",
    "path_width": 2.0,
    "start_marker": "o",
    "goal_marker": "X",
    "start_color": "#009e73",
    "goal_color": "#cc79a7",
    "marker_edge": "#111111",
    "legend_fontsize": 8.5,
    "title_pad": 8,
    "search_colors": ["#0072b2", "#009e73", "#d55e00", "#cc79a7"],
    "bar_edge": "#333333",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "axes.labelsize": STYLE["label_size"],
            "figure.dpi": STYLE["dpi"],
        }
    )


def plot_world(
    ax,
    world: World,
    targets: list[Target],
    title: str = "World",
    buffer: int | None = None,
) -> None:
    """Terrain heatmap, contours, no-fly circles, buffer overlay, wind, base, targets."""
    n = int(world.terrain.shape[0])
    if buffer is None:
        buffer = Config.restricted_buffer
    im = ax.imshow(
        world.terrain,
        origin="upper",
        cmap=STYLE["cmap_terrain"],
        interpolation="nearest",
    )
    ax.contour(
        world.terrain,
        levels=STYLE["contour_levels"],
        colors=STYLE["contour_color"],
        linewidths=STYLE["contour_width"],
        origin="upper",
    )
    for cr, cc, radius in world.nofly_circles:
        circle = Circle(
            (cc, cr),
            radius,
            facecolor=STYLE["nofly_face"],
            edgecolor=STYLE["nofly_edge"],
            alpha=STYLE["nofly_alpha"],
            linewidth=STYLE["nofly_linewidth"],
            linestyle="--",
        )
        ax.add_patch(circle)
    restricted = nofly_buffer_mask(world.no_fly, int(buffer))
    overlay = np.where(restricted, 1.0, np.nan)
    ax.imshow(
        overlay,
        origin="upper",
        cmap=STYLE["restricted_cmap"],
        alpha=STYLE["restricted_alpha"],
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    stride = int(STYLE["wind_stride"])
    ys, xs = np.mgrid[stride // 2 : n : stride, stride // 2 : n : stride]
    u = np.full(xs.shape, world.wind[1])
    v = np.full(ys.shape, world.wind[0])
    ax.quiver(
        xs,
        ys,
        u,
        v,
        color=STYLE["wind_color"],
        alpha=STYLE["wind_alpha"],
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=STYLE["wind_scale"],
        width=STYLE["wind_width"],
    )
    ax.scatter(
        [world.base[1]],
        [world.base[0]],
        c=STYLE["base_color"],
        marker=STYLE["base_marker"],
        s=STYLE["base_size"],
        edgecolors=STYLE["marker_edge"],
        zorder=5,
        label="Base",
    )
    if targets:
        ax.scatter(
            [t.pos[1] for t in targets],
            [t.pos[0] for t in targets],
            c=STYLE["target_color"],
            marker=STYLE["target_marker"],
            s=STYLE["target_size"],
            edgecolors=STYLE["marker_edge"],
            zorder=5,
            label="Target",
        )
    ax.set_title(title, pad=STYLE["title_pad"])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    handles = [
        Patch(
            facecolor=STYLE["nofly_face"],
            edgecolor=STYLE["nofly_edge"],
            alpha=STYLE["nofly_alpha"],
            linestyle="--",
            label="No-fly",
        ),
        Patch(
            facecolor="#c44e52",
            alpha=STYLE["restricted_alpha"],
            label="Restricted buffer",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["wind_color"],
            marker=">",
            linestyle="None",
            label="Wind",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["base_color"],
            marker=STYLE["base_marker"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edge"],
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["target_color"],
            marker=STYLE["target_marker"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edge"],
            label="Target",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=STYLE["legend_fontsize"],
        framealpha=0.92,
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("height (m)")


def plot_search(
    ax,
    blocked: np.ndarray,
    result: SearchResult,
    start: Cell,
    goal: Cell,
    title: str,
) -> None:
    """Show expansion order, path, start, and goal. Marker shape is not colour-only."""
    n = int(blocked.shape[0])
    order = np.full((n, n), np.nan)
    for i, cell in enumerate(result.explored):
        order[cell] = i
    ax.imshow(blocked.astype(float), origin="upper", cmap="Greys", vmin=0, vmax=1, alpha=0.35)
    ax.imshow(order, origin="upper", cmap=STYLE["cmap_explored"], interpolation="nearest")
    if len(result.path) >= 2:
        ys = [p[0] for p in result.path]
        xs = [p[1] for p in result.path]
        ax.plot(
            xs,
            ys,
            color=STYLE["path_color"],
            linestyle=STYLE["path_style"],
            linewidth=STYLE["path_width"],
            zorder=4,
        )
    ax.scatter(
        [start[1]],
        [start[0]],
        c=STYLE["start_color"],
        marker=STYLE["start_marker"],
        s=55,
        edgecolors=STYLE["marker_edge"],
        zorder=5,
    )
    ax.scatter(
        [goal[1]],
        [goal[0]],
        c=STYLE["goal_color"],
        marker=STYLE["goal_marker"],
        s=70,
        edgecolors=STYLE["marker_edge"],
        zorder=5,
    )
    ax.set_title(title, pad=STYLE["title_pad"])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")


def fig_world(world: World, targets: list[Target], path: str | Path) -> None:
    """Save ``fig_world.png``."""
    _apply_style()
    fig, ax = plt.subplots(figsize=STYLE["figsize_world"])
    plot_world(
        ax,
        world,
        targets,
        title="Terrain, wind, no-fly zones, restricted buffer, base, targets",
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_world_figure(out_dir: str | Path = "results", seed: int | None = None) -> Path:
    """Build a default scenario and save ``fig_world.png`` (no search required)."""
    from numpy.random import default_rng

    from uavplanner.world import make_targets, make_world

    cfg = Config() if seed is None else Config(seed=seed)
    rng = default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    out = Path(out_dir) / "fig_world.png"
    fig_world(world, targets, out)
    return out


def fig_search_explored(
    blocked: np.ndarray,
    panels: list[tuple[str, SearchResult, Cell, Cell]],
    path: str | Path,
) -> None:
    """Save a 2x2 search exploration figure for four search results."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=STYLE["figsize_search"])
    for ax, (title, result, start, goal) in zip(axes.ravel(), panels):
        steps = max(len(result.path) - 1, 0) if result.path else 0
        label = f"{title}\nexpanded={result.expanded}, steps={steps}"
        plot_search(ax, blocked, result, start, goal, label)
    handles = [
        Line2D(
            [0],
            [0],
            color=STYLE["path_color"],
            lw=STYLE["path_width"],
            label="Path",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["start_color"],
            marker=STYLE["start_marker"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edge"],
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["goal_color"],
            marker=STYLE["goal_marker"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edge"],
            label="Goal",
        ),
        Patch(facecolor="#777777", alpha=0.35, label="Blocked"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=STYLE["legend_fontsize"],
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Explored cells: " + ", ".join(title for title, _result, _start, _goal in panels),
        y=1.01,
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures(out_dir: str | Path = "results") -> tuple[Path, Path]:
    """Build a default scenario and write the figures."""
    from numpy.random import default_rng

    from uavplanner.config import Config
    from uavplanner.informed import greedy
    from uavplanner.models import EnergyCoef
    from uavplanner.uninformed import bfs, dfs, ucs
    from uavplanner.world import make_targets, make_world

    cfg = Config()
    rng = default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    blocked = world.no_fly
    start = world.base
    goal = next(t.pos for t in targets if not blocked[t.pos])
    coef = EnergyCoef()
    payload = 1.0
    panels = [
        ("BFS", bfs(blocked, start, goal), start, goal),
        ("DFS", dfs(blocked, start, goal), start, goal),
        ("UCS", ucs(world, blocked, start, goal, coef, payload), start, goal),
        ("Greedy", greedy(world, blocked, start, goal, coef, payload), start, goal),
    ]
    out = Path(out_dir)
    world_path = out / "fig_world.png"
    search_path = out / "fig_search_explored.png"
    fig_world(world, targets, world_path)
    fig_search_explored(blocked, panels, search_path)
    return world_path, search_path


def compare_searches(
    world: World,
    blocked: np.ndarray,
    pairs: list[tuple[Cell, Cell]],
    coef: EnergyCoef,
    payload: float,
    max_wind: float,
) -> dict[str, dict[str, float]]:
    """Average energy and expanded nodes for BFS, UCS, Greedy, and A*.

    Every pair must be reachable. BFS is priced afterward with the same energy
    model as the energy-aware searches, because its own ``cost`` is step count.
    """
    from uavplanner.informed import astar, greedy
    from uavplanner.uninformed import bfs, ucs

    if not pairs:
        raise ValueError("at least one start-goal pair is required")
    totals = {
        name: {"energy": 0.0, "expanded": 0.0}
        for name in ("BFS", "UCS", "Greedy", "A*")
    }
    for start, goal in pairs:
        results = {
            "BFS": bfs(blocked, start, goal),
            "UCS": ucs(world, blocked, start, goal, coef, payload),
            "Greedy": greedy(world, blocked, start, goal, coef, payload),
            "A*": astar(world, blocked, start, goal, coef, payload, max_wind),
        }
        for name, result in results.items():
            if not result.path and start != goal:
                raise ValueError(f"unreachable comparison pair: {start} -> {goal}")
            totals[name]["energy"] += path_energy(world, result.path, coef, payload)
            totals[name]["expanded"] += result.expanded
    return {
        name: {metric: value / len(pairs) for metric, value in values.items()}
        for name, values in totals.items()
    }


def fig_search_compare(
    metrics: dict[str, dict[str, float]], path: str | Path
) -> None:
    """Save mean path energy and expanded-node bars for the four searches."""
    _apply_style()
    names = list(metrics)
    colors = STYLE["search_colors"][: len(names)]
    energies = [metrics[name]["energy"] for name in names]
    expanded = [metrics[name]["expanded"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_compare"])
    for ax, values, title, ylabel in (
        (axes[0], energies, "Mean path energy", "energy (EU)"),
        (axes[1], expanded, "Mean nodes expanded", "nodes"),
    ):
        ax.bar(names, values, color=colors, edgecolor=STYLE["bar_edge"])
        ax.set_title(title, pad=STYLE["title_pad"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Search comparison over 30 reachable pairs", y=1.02)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures2(
    out_dir: str | Path = "results", n_pairs: int = 30
) -> tuple[Path, Path, Path]:
    """Build the reproducible exploration and comparison figures."""
    from numpy.random import default_rng

    from uavplanner.informed import astar, greedy
    from uavplanner.uninformed import bfs, dfs_reachable, ucs
    from uavplanner.world import make_targets, make_world

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    cfg = Config()
    rng = default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    blocked = world.no_fly
    start = world.base
    goal = next(target.pos for target in targets if not blocked[target.pos])
    coef = EnergyCoef()
    payload = 1.0
    panels = [
        ("BFS", bfs(blocked, start, goal), start, goal),
        ("UCS", ucs(world, blocked, start, goal, coef, payload), start, goal),
        ("Greedy", greedy(world, blocked, start, goal, coef, payload), start, goal),
        (
            "A*",
            astar(world, blocked, start, goal, coef, payload, cfg.wind_limit),
            start,
            goal,
        ),
    ]
    reachable = [tuple(cell) for cell in np.argwhere(dfs_reachable(blocked, world.base))]
    if len(reachable) < 2:
        raise ValueError("world has fewer than two reachable cells")
    pairs: list[tuple[Cell, Cell]] = []
    for _ in range(n_pairs):
        start_index, goal_index = rng.choice(len(reachable), size=2, replace=False)
        pairs.append((reachable[int(start_index)], reachable[int(goal_index)]))
    metrics = compare_searches(world, blocked, pairs, coef, payload, cfg.wind_limit)
    out = Path(out_dir)
    world_path = out / "fig_world.png"
    explored_path = out / "fig_search_explored.png"
    comparison_path = out / "fig_search_compare.png"
    fig_world(world, targets, world_path)
    fig_search_explored(blocked, panels, explored_path)
    fig_search_compare(metrics, comparison_path)
    return world_path, explored_path, comparison_path