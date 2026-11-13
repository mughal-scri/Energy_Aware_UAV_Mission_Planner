"""Plots. Colours and sizes live in ``STYLE``; do not hard-code them."""

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
    "figsize_ordering": (10.5, 5.0),
    "figsize_csp": (11.2, 5.4),
    "figsize_minimax": (8.2, 4.8),
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
    "random_color": "#7f7f7f",
    "hc_color": "#009e73",
    "ga_color": "#cc79a7",
    "base_route_color": "#0072b2",
    "route_linewidth": 1.6,
    "drone_colors": ["#0072b2", "#009e73", "#d55e00"],
    "drone_markers": ["o", "s", "D"],
    "unassigned_color": "#7f7f7f",
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


def write_figures_1(out_dir: str | Path = "results") -> tuple[Path, Path]:
    """Build a default scenario and write the two initial figures."""
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


def write_figures_2(
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


def synthetic_ordering_matrix(
    n_stops: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible directed synthetic matrix for ordering experiments.

    It supplies the 12--20 stops needed for scaling, which exceed a generated
    world scenario's usual six to eight targets. The direction multiplier
    models a steady wind, so costs remain asymmetric like the A* matrix.
    """
    if n_stops < 1:
        raise ValueError("n_stops must be positive")
    points = rng.uniform(0.0, 1.0, size=(n_stops + 1, 2))
    wind = rng.normal(size=2)
    wind /= np.linalg.norm(wind)
    matrix = np.zeros((n_stops + 1, n_stops + 1), dtype=float)
    for start in range(n_stops + 1):
        for goal in range(n_stops + 1):
            if start == goal:
                continue
            delta = points[goal] - points[start]
            distance = float(np.linalg.norm(delta))
            direction = delta / distance
            matrix[start, goal] = distance * (1.0 + 0.20 * float(np.dot(direction, wind)))
    return matrix, points


def ordering_scaling_experiment(
    sizes: tuple[int, ...] = (12, 16, 20), seed: int = Config.seed
) -> list[dict[str, float]]:
    """Compare random, Hill Climbing, and GA tours at practical problem sizes."""
    from uavplanner.genetic import genetic_order
    from uavplanner.hill_climbing import hill_climb, tour_cost

    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("sizes must contain values of at least 2")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for size in sizes:
        matrix, _points = synthetic_ordering_matrix(size, rng)
        stops = list(range(1, size + 1))
        random_order = list(rng.permutation(stops))
        _hc_order, hc_cost, _hc_history = hill_climb(
            matrix, stops, restarts=Config.hc_restarts, rng=rng
        )
        _ga_order, ga_cost, _ga_history = genetic_order(
            matrix,
            stops,
            pop=Config.ga_pop,
            gens=Config.ga_gens,
            pc=Config.ga_pc,
            pm=Config.ga_pm,
            tournament=Config.ga_tournament,
            elite=Config.ga_elite,
            rng=rng,
        )
        rows.append(
            {
                "stops": float(size),
                "random": tour_cost(random_order, matrix),
                "hill_climb": hc_cost,
                "genetic": ga_cost,
            }
        )
    return rows


def _plot_tour(ax, points: np.ndarray, order: list[int], title: str, color: str) -> None:
    """Plot a base-returning synthetic tour without relying on colour alone."""
    route = [0, *order, 0]
    ax.plot(
        points[route, 0],
        points[route, 1],
        color=color,
        linewidth=STYLE["route_linewidth"],
        marker="o",
        markersize=3.5,
    )
    ax.scatter(
        points[0, 0], points[0, 1], marker="s", s=STYLE["base_size"],
        color=STYLE["base_route_color"], edgecolors=STYLE["marker_edge"], label="Base", zorder=3
    )
    ax.scatter(
        points[1:, 0], points[1:, 1], marker=STYLE["target_marker"], s=STYLE["target_size"],
        color=STYLE["target_color"], edgecolors=STYLE["marker_edge"], label="Stop", zorder=3
    )
    ax.set_title(title, pad=STYLE["title_pad"])
    ax.set_xlabel("synthetic east-west position")
    ax.set_ylabel("synthetic north-south position")
    ax.set_aspect("equal")


def fig_tour_hc_ga(
    points: np.ndarray,
    hc_order: list[int],
    hc_cost: float,
    ga_order: list[int],
    ga_cost: float,
    path: str | Path,
) -> None:
    """Save side-by-side synthetic tours returned by Hill Climbing and GA."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_ordering"])
    _plot_tour(axes[0], points, hc_order, f"Hill Climbing\ncost={hc_cost:.2f}", STYLE["hc_color"])
    _plot_tour(axes[1], points, ga_order, f"Genetic Algorithm\ncost={ga_cost:.2f}", STYLE["ga_color"])
    axes[1].legend(loc="best", fontsize=STYLE["legend_fontsize"])
    fig.suptitle("Directed tour order on a synthetic 12-stop instance", y=1.02)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def fig_convergence(hc_history: list[float], ga_history: list[float], path: str | Path) -> None:
    """Save the best-cost histories for the two ordering methods."""
    _apply_style()
    fig, ax = plt.subplots(figsize=STYLE["figsize_ordering"])
    ax.plot(hc_history, color=STYLE["hc_color"], label="Hill Climbing", linewidth=2.0)
    ax.plot(ga_history, color=STYLE["ga_color"], label="Genetic Algorithm", linewidth=2.0)
    ax.set_title("Best tour cost during optimization", pad=STYLE["title_pad"])
    ax.set_xlabel("improvement step / generation")
    ax.set_ylabel("tour energy (synthetic units)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=STYLE["legend_fontsize"])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def fig_ordering_scaling(rows: list[dict[str, float]], path: str | Path) -> None:
    """Save random, Hill Climbing, and GA costs for 12--20 stop instances."""
    _apply_style()
    labels = [str(int(row["stops"])) for row in rows]
    locations = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=STYLE["figsize_ordering"])
    for offset, key, label, color in (
        (-width, "random", "Random", STYLE["random_color"]),
        (0.0, "hill_climb", "Hill Climbing", STYLE["hc_color"]),
        (width, "genetic", "Genetic Algorithm", STYLE["ga_color"]),
    ):
        ax.bar(
            locations + offset,
            [row[key] for row in rows],
            width,
            label=label,
            color=color,
            edgecolor=STYLE["bar_edge"],
        )
    ax.set_title("Ordering cost as stop count grows", pad=STYLE["title_pad"])
    ax.set_xlabel("number of stops")
    ax.set_ylabel("tour energy (synthetic units)")
    ax.set_xticks(locations, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=STYLE["legend_fontsize"])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures_3(out_dir: str | Path = "results") -> tuple[Path, Path, Path]:
    """Generate all reproducible ordering figures from the default seed."""
    from uavplanner.genetic import genetic_order
    from uavplanner.hill_climbing import hill_climb

    rng = np.random.default_rng(Config.seed)
    matrix, points = synthetic_ordering_matrix(12, rng)
    stops = list(range(1, 13))
    hc_order, hc_cost, hc_history = hill_climb(
        matrix, stops, restarts=Config.hc_restarts, rng=rng
    )
    ga_order, ga_cost, ga_history = genetic_order(
        matrix,
        stops,
        pop=Config.ga_pop,
        gens=Config.ga_gens,
        pc=Config.ga_pc,
        pm=Config.ga_pm,
        tournament=Config.ga_tournament,
        elite=Config.ga_elite,
        rng=rng,
    )
    rows = ordering_scaling_experiment()
    out = Path(out_dir)
    tour_path = out / "fig_tour_hc_ga.png"
    convergence_path = out / "fig_convergence.png"
    scaling_path = out / "fig_ordering_scaling.png"
    fig_tour_hc_ga(points, hc_order, hc_cost, ga_order, ga_cost, tour_path)
    fig_convergence(hc_history, ga_history, convergence_path)
    fig_ordering_scaling(rows, scaling_path)
    return tour_path, convergence_path, scaling_path


def fig_csp_assignment(
    world: World,
    targets: list[Target],
    assignment: dict[int, int],
    plain_backtracks: int,
    heuristic_backtracks: int,
    path: str | Path,
) -> None:
    """Draw targets coloured by assigned drone beside a backtrack comparison."""
    _apply_style()
    n = int(world.terrain.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_csp"], gridspec_kw={"width_ratios": [1.35, 1]})
    ax, bars = axes
    image = ax.imshow(
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
        ax.add_patch(
            Circle(
                (cc, cr),
                radius,
                facecolor=STYLE["nofly_face"],
                edgecolor=STYLE["nofly_edge"],
                alpha=STYLE["nofly_alpha"],
                linewidth=STYLE["nofly_linewidth"],
                linestyle="--",
            )
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
    drone_ids = sorted({drone_id for drone_id in assignment.values()})
    legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color=STYLE["base_color"],
            marker=STYLE["base_marker"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edge"],
            label="Base",
        )
    ]
    for drone_id in drone_ids:
        color = STYLE["drone_colors"][(drone_id - 1) % len(STYLE["drone_colors"])]
        marker = STYLE["drone_markers"][(drone_id - 1) % len(STYLE["drone_markers"])]
        chosen = [target for target in targets if assignment.get(target.id) == drone_id]
        ax.scatter(
            [target.pos[1] for target in chosen],
            [target.pos[0] for target in chosen],
            c=color,
            marker=marker,
            s=STYLE["target_size"],
            edgecolors=STYLE["marker_edge"],
            zorder=6,
            label=f"Drone {drone_id}",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="None",
                markeredgecolor=STYLE["marker_edge"],
                label=f"Drone {drone_id}",
            )
        )
    unassigned = [target for target in targets if target.id not in assignment]
    if unassigned:
        ax.scatter(
            [target.pos[1] for target in unassigned],
            [target.pos[0] for target in unassigned],
            c=STYLE["unassigned_color"],
            marker=STYLE["target_marker"],
            s=STYLE["target_size"],
            edgecolors=STYLE["marker_edge"],
            zorder=6,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=STYLE["unassigned_color"],
                marker=STYLE["target_marker"],
                linestyle="None",
                markeredgecolor=STYLE["marker_edge"],
                label="Unassigned",
            )
        )
    ax.set_title("Targets coloured by assigned drone", pad=STYLE["title_pad"])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=STYLE["legend_fontsize"], framealpha=0.92)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("height (m)")

    labels = ["Plain\nbacktracking", "MRV +\nforward checking"]
    counts = [plain_backtracks, heuristic_backtracks]
    bar_colors = [STYLE["random_color"], STYLE["hc_color"]]
    positions = np.arange(len(labels))
    drawn = bars.bar(
        positions,
        counts,
        color=bar_colors,
        edgecolor=STYLE["bar_edge"],
        width=0.62,
    )
    bars.bar_label(drawn, padding=3)
    bars.set_xticks(positions, labels)
    bars.set_ylabel("backtracks")
    bars.set_title("Assignment search effort", pad=STYLE["title_pad"])
    bars.set_ylim(0, max(counts + [1]) * 1.25)
    bars.grid(axis="y", alpha=0.25)
    bars.set_axisbelow(True)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures_4(out_dir: str | Path = "results") -> Path:
    """Generate the seeded CSP assignment figure and backtrack comparison."""
    from uavplanner.csp import avg_cell_cost, csp_assign, estimate_target_cost
    from uavplanner.models import Drone, EnergyCoef, Target
    from uavplanner.world import make_targets, make_world

    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    targets = make_targets(world, cfg, rng)
    drones = [
        Drone(id=drone_id, battery=cfg.battery, capacity_kg=cfg.capacity_kg, pos=world.base)
        for drone_id in range(1, cfg.n_drones + 1)
    ]
    average = avg_cell_cost(world, EnergyCoef(), rng, samples=cfg.csp_samples)
    assignment = None
    # Keep the seeded figure informative when the full random scenario is
    # infeasible under the deliberately conservative per-target CSP estimate.
    for count in range(len(targets), 0, -1):
        selected = targets[:count]
        estimates = {
            target.id: estimate_target_cost(world.base, target, average)
            for target in selected
        }
        candidate, heuristic_stats = csp_assign(
            selected,
            drones,
            estimates,
            reserve=cfg.reserve,
            use_mrv=True,
            use_fc=True,
        )
        if candidate is not None:
            assignment = candidate
            break
    # A compact constrained example makes the MRV/forward-checking contrast
    # visible even when the seeded map's feasible subset needs no backtracking.
    effort_targets = [
        Target(id=1, pos=(1, 1), mass_kg=1.0),
        Target(id=2, pos=(2, 2), mass_kg=3.0),
    ]
    effort_drones = [
        Drone(id=1, battery=100.0, capacity_kg=3.0),
        Drone(id=2, battery=100.0, capacity_kg=1.0),
    ]
    effort_costs = {1: 1.0, 2: 1.0}
    _plain, plain_stats = csp_assign(
        effort_targets, effort_drones, effort_costs, use_mrv=False, use_fc=False
    )
    _smart, heuristic_stats = csp_assign(
        effort_targets, effort_drones, effort_costs, use_mrv=True, use_fc=True
    )
    path = Path(out_dir) / "fig_csp_assignment.png"
    fig_csp_assignment(
        world,
        targets,
        {} if assignment is None else assignment,
        plain_stats["backtracks"],
        heuristic_stats["backtracks"],
        path,
    )
    return path


def fig_minimax_nodes(plain_nodes: int, alpha_beta_nodes: int, path: str | Path) -> None:
    """Compare total visited nodes with Minimax and Alpha-Beta pruning."""
    _apply_style()
    fig, ax = plt.subplots(figsize=STYLE["figsize_minimax"])
    labels = ["Minimax", "Alpha-Beta"]
    values = [plain_nodes, alpha_beta_nodes]
    bars = ax.bar(
        labels,
        values,
        color=[STYLE["random_color"], STYLE["hc_color"]],
        edgecolor=STYLE["bar_edge"],
        width=0.62,
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("nodes visited across seeded positions")
    ax.set_title("Alpha-Beta reduces search work", pad=STYLE["title_pad"])
    ax.set_ylim(0, max(values + [1]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures_5(out_dir: str | Path = "results") -> tuple[Path, Path, Path]:
    """Run a seeded three-drone mission and regenerate summary figures."""
    from uavplanner.adversarial import choose_move
    from uavplanner.console import Console
    from uavplanner.models import Drone, Target, World
    from uavplanner.planner import Plan
    from uavplanner.simulator import run_mission

    cfg = Config(grid_n=11, max_ticks=80, takeoff_stagger=2, noise_sd=0.0)
    world = World(
        terrain=np.zeros((cfg.grid_n, cfg.grid_n)),
        no_fly=np.zeros((cfg.grid_n, cfg.grid_n), dtype=bool),
        wind=(0.0, 0.0),
        base=(5, 5),
    )
    targets = [
        Target(id=1, pos=(2, 5), mass_kg=0.5),
        Target(id=2, pos=(5, 8), mass_kg=0.5),
        Target(id=3, pos=(8, 5), mass_kg=0.5),
    ]
    drones = [
        Drone(id=drone_id, battery=cfg.battery, capacity_kg=cfg.capacity_kg, pos=world.base)
        for drone_id in range(1, 4)
    ]
    routes = {
        1: [(5, 5), (4, 5), (3, 5), (2, 5), (3, 5), (4, 5), (5, 5)],
        2: [(5, 5), (5, 6), (5, 7), (5, 8), (5, 7), (5, 6), (5, 5)],
        3: [(5, 5), (6, 5), (7, 5), (8, 5), (7, 5), (6, 5), (5, 5)],
    }
    plan = Plan(
        verdict="GO",
        reasons=[],
        rejected={},
        assignment={1: 1, 2: 2, 3: 3},
        order={1: [1], 2: [2], 3: [3]},
        routes=routes,
        est_energy={1: 8.0, 2: 8.0, 3: 8.0},
    )
    console = Console()
    summary = run_mission(
        world, plan, drones, None, targets, cfg.true_coef, cfg.true_coef, cfg,
        np.random.default_rng(cfg.seed), console, [],
    )
    if (
        summary["collisions"] != 0
        or summary["missions_completed"] != len(drones)
        or summary["targets_confirmed"] != len(targets)
    ):
        raise RuntimeError("Figure mission did not complete cleanly")

    battery_path = Path(out_dir) / "fig_battery.png"
    timeline_path = Path(out_dir) / "fig_timeline.png"
    _apply_style()
    for output, plotter in (
        (battery_path, console.plot_battery),
        (timeline_path, console.plot_timeline),
    ):
        fig, ax = plt.subplots(figsize=STYLE["figsize_compare"])
        plotter(ax)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=STYLE["dpi"], bbox_inches="tight")
        plt.close(fig)

    rng = np.random.default_rng(cfg.seed)
    plain_total = 0
    pruned_total = 0
    mask = np.zeros((9, 9), dtype=bool)
    minimax_cfg = Config()
    for _ in range(20):
        drone, interceptor, waypoint = [
            tuple(int(value) for value in rng.integers(0, 9, size=2))
            for _ in range(3)
        ]
        plain_total += choose_move(
            drone, interceptor, waypoint, mask, cfg=minimax_cfg, use_alpha_beta=False
        )[2]
        pruned_total += choose_move(
            drone, interceptor, waypoint, mask, cfg=minimax_cfg, use_alpha_beta=True
        )[2]
    minimax_path = Path(out_dir) / "fig_minimax_nodes.png"
    fig_minimax_nodes(plain_total, pruned_total, minimax_path)
    return battery_path, timeline_path, minimax_path
