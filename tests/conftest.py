"""Headless Matplotlib so tests do not need a display."""

import matplotlib

matplotlib.use("Agg")
