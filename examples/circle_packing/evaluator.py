"""Evaluator: validate the packing and score the sum of radii.

2.635 is the reference result reported by AlphaEvolve for n=26.

Metrics:
- combined_score: sum_radii / 2.635, not capped, 0.0 if invalid.
- sum_radii, validity.
- radius_gini: Gini coefficient of the radii, 0 for uniform sizes, higher
  for mixed sizes.
"""

import importlib.util

import numpy as np

N = 26
TARGET = 2.635
TOL = 1e-9  # numerical slack only; exploiting it gains at most 13e-9 in sum_radii


def _validate(centers, radii):
    if centers.shape != (N, 2) or radii.shape != (N,):
        return False
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()):
        return False
    if (radii < 0).any():
        return False
    if (centers - radii[:, None] < -TOL).any() or (centers + radii[:, None] > 1 + TOL).any():
        return False
    dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    gap = dist - (radii[:, None] + radii[None, :])
    i, j = np.triu_indices(N, 1)
    return not (gap[i, j] < -TOL).any()


def _gini(radii):
    r = np.sort(radii)
    total = r.sum()
    if total == 0:
        return 0.0
    n = len(r)
    return max(0.0, float((2 * np.arange(1, n + 1) - n - 1) @ r / (n * total)))


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    centers, radii = program.pack()
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if not _validate(centers, radii):
        return {"combined_score": 0.0, "sum_radii": 0.0, "validity": 0.0, "radius_gini": 0.0}
    total = float(radii.sum())
    return {
        "combined_score": total / TARGET,
        "sum_radii": total,
        "validity": 1.0,
        "radius_gini": _gini(radii),
    }
