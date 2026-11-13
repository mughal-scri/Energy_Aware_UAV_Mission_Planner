"""Target sensing and operator console tests."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from uavplanner.console import Console
from uavplanner.sensor import detect_prob, look


def test_sensor_probabilities_and_seeded_detection_rate() -> None:
    assert [detect_prob(distance) for distance in range(5)] == [0.9, 0.7, 0.5, 0.3, 0.0]
    with pytest.raises(ValueError):
        detect_prob(-1)
    rng = np.random.default_rng(23)
    successes = sum(look((4, 4), (4, 4), rng) for _ in range(10_000))
    assert successes / 10_000 == pytest.approx(0.9, abs=0.015)


def test_console_csv_rows_and_plot_helpers(tmp_path) -> None:
    console = Console()
    console.log(0, 1, (2, 3), 100.0, "TRANSIT", "takeoff")
    console.log(1, 1, (2, 4), 95.0, "HOVER", "target confirmed")
    assert len(console.rows()) == 2
    console.rows().clear()
    assert len(console.rows()) == 2

    csv_path = tmp_path / "logs" / "mission.csv"
    console.to_csv(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["event"] == "target confirmed"
    assert rows[1]["column"] == "4"

    fig, axes = plt.subplots(1, 2)
    console.plot_battery(axes[0])
    console.plot_timeline(axes[1])
    assert axes[0].get_ylabel() == "battery (%)"
    assert axes[1].get_xlabel() == "tick"
    plt.close(fig)


def test_console_rejects_bad_values() -> None:
    console = Console()
    with pytest.raises(ValueError):
        console.log(-1, 1, (0, 0), 50.0, "WAIT")
    with pytest.raises(ValueError):
        console.log(0, 1, (0, 0), 120.0, "WAIT")
