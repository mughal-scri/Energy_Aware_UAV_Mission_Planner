"""In-memory operator log with CSV export and Matplotlib plotting helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes

from uavplanner.models import Cell
from uavplanner.viz import STYLE


@dataclass(frozen=True)
class ConsoleRow:
    """One drone status sample recorded for one simulation tick."""

    tick: int
    drone: int
    pos: Cell
    battery_pct: float
    mode: str
    event: str = ""


class Console:
    """Collect flight rows and expose CSV and plot views for the operator."""

    def __init__(self) -> None:
        self._rows: list[ConsoleRow] = []

    def log(
        self,
        tick: int,
        drone: int,
        pos: Cell,
        battery_pct: float,
        mode: str,
        event: str = "",
    ) -> None:
        """Append one status or event row to the in-memory log."""
        if tick < 0 or drone < 1:
            raise ValueError("tick must be non-negative and drone id must be positive")
        if not 0.0 <= battery_pct <= 100.0:
            raise ValueError("battery_pct must be between 0 and 100")
        self._rows.append(ConsoleRow(tick, drone, pos, float(battery_pct), mode, event))

    def rows(self) -> list[ConsoleRow]:
        """Return a copy of the logged rows in insertion order."""
        return list(self._rows)

    def to_csv(self, path: str | Path) -> None:
        """Write the complete console log to a CSV file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("tick", "drone", "row", "column", "battery_pct", "mode", "event"))
            for row in self._rows:
                writer.writerow((row.tick, row.drone, row.pos[0], row.pos[1], row.battery_pct, row.mode, row.event))

    def plot_battery(self, ax: Axes) -> None:
        """Draw battery percentage over time, with one line per drone."""
        drone_ids = sorted({row.drone for row in self._rows})
        battery_values = [row.battery_pct for row in self._rows]
        for index, drone_id in enumerate(drone_ids):
            rows = [row for row in self._rows if row.drone == drone_id]
            color = STYLE["drone_colors"][index % len(STYLE["drone_colors"])]
            marker = STYLE["drone_markers"][index % len(STYLE["drone_markers"])]
            ax.plot([row.tick for row in rows], [row.battery_pct for row in rows],
                    color=color, marker=marker, label=f"Drone {drone_id}")
        ax.set_xlabel("tick")
        ax.set_ylabel("battery (%)")
        ax.set_ylim(max(0.0, min(battery_values, default=0.0) - 3.0), 100)
        ax.set_title("Battery over mission time")
        if drone_ids:
            ax.legend()
        ax.grid(True, alpha=0.25)

    def plot_timeline(self, ax: Axes) -> None:
        """Draw event markers along the tick axis, grouped by drone."""
        events = [row for row in self._rows if row.event]
        drone_ids = sorted({row.drone for row in self._rows})
        for index, drone_id in enumerate(drone_ids):
            chosen = [row for row in events if row.drone == drone_id]
            if not chosen:
                continue
            color = STYLE["drone_colors"][index % len(STYLE["drone_colors"])]
            marker = STYLE["drone_markers"][index % len(STYLE["drone_markers"])]
            ax.scatter([row.tick for row in chosen], [row.drone for row in chosen],
                       color=color, marker=marker, label=f"Drone {drone_id}")
            for row in chosen:
                ax.annotate(row.event, (row.tick, row.drone), xytext=(0, 6),
                            textcoords="offset points", ha="center")
        ax.set_xlabel("tick")
        ax.set_ylabel("drone")
        ax.set_title("Mission events")
        if drone_ids:
            ax.set_yticks(drone_ids)
            ax.set_ylim(0.5, max(drone_ids) + 0.9)
        if events:
            ax.legend()
        ax.grid(True, axis="x", alpha=0.25)
