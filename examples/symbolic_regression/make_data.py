#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate data/train.csv: a driven oscillator with cubic stiffness and quadratic drag.

    dv/dt = F0 sin(t) - c v|v| - omega0^2 x - alpha x^3

The observations follow one trajectory from x = 2, v = 0 over t in [0, 40],
sampled at uniformly random times, in the style of LLM-SRBench's LSR-Synth
oscillators: the target is the exact acceleration at each sampled state.
This file holds the ground truth; the model never sees it.

Usage: uv run examples/symbolic_regression/make_data.py
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

F0, C, OMEGA0_SQ, ALPHA = 0.8, 0.3, 1.5, 0.5
SAMPLES = 1000


def acceleration(x, t, v):
    return F0 * np.sin(t) - C * v * np.abs(v) - OMEGA0_SQ * x - ALPHA * x**3


def main():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 40, SAMPLES))
    solution = solve_ivp(
        lambda t, s: [s[1], acceleration(s[0], t, s[1])],
        (0, 40),
        [2.0, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    x, v = solution.y
    rows = np.column_stack([acceleration(x, t, v), x, t, v])
    path = Path(__file__).parent / "data/train.csv"
    np.savetxt(path, rows, delimiter=",", header="dv_dt,x,t,v", comments="", fmt="%.17g")
    print(f"wrote {SAMPLES} observations to {path}")


if __name__ == "__main__":
    main()
