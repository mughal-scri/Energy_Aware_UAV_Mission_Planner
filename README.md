# Energy-Aware UAV Mission Planner

Python simulation of a small drone fleet (up to 3 drones) that plans and flies
missions over hilly terrain. It applies the algorithms from an AI course
(search, local search, GA, CSP, expert systems, adversarial search, and a
learning cost model) in one coherent system.

## Requirements

- Python 3.10 or newer
- Runtime: NumPy, Matplotlib
- Development: pytest, Jupyter

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

```bash
python -c "from uavplanner.viz import write_figures_2; write_figures_2('results')"
```

Generate the ordering figures:

```bash
python -c "from uavplanner.viz import write_figures_3; write_figures_3('results')"
```

Generate the assignment figures:

```bash
python -c "from uavplanner.viz import write_figures_4; write_figures_4('results')"
```

Generate the simulation summary figures:

```bash
python -c "from uavplanner.viz import write_figures_5; write_figures_5('results')"
```

A one-command scenario runner will land later (`python -m uavplanner.cli`).
