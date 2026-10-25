"""Evaluator: check that the circles lie inside the unit square without
overlapping, then score the sum of their radii.

Metrics:
- sum_radii: the fitness.
- radius_gini: Gini coefficient of the radii, 0 when all circles have the same
  size; the MAP-Elites feature.
"""

import importlib.util

import numpy as np

N = 26
TOL = 1e-9


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    centers, radii = (np.asarray(a, dtype=float) for a in program.pack())

    if centers.shape != (N, 2) or radii.shape != (N,):
        raise ValueError(
            f"expected shapes ({N}, 2) and ({N},), got {centers.shape} and {radii.shape}"
        )
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()) or (radii < 0).any():
        raise ValueError("centers and radii must be finite, radii non-negative")
    margin = np.minimum(centers, 1 - centers).min(axis=1) - radii
    if margin.min() < -TOL:
        raise ValueError(f"circle {margin.argmin()} extends {-margin.min():.3g} beyond the square")
    i, j = np.triu_indices(N, 1)
    gap = np.linalg.norm(centers[i] - centers[j], axis=1) - radii[i] - radii[j]
    if gap.min() < -TOL:
        k = gap.argmin()
        raise ValueError(f"circles {i[k]} and {j[k]} overlap by {-gap[k]:.3g}")

    r = np.sort(radii)
    gini = (2 * np.arange(1, N + 1) - N - 1) @ r / (N * r.sum()) if r.sum() > 0 else 0.0
    return {"sum_radii": float(r.sum()), "radius_gini": float(gini)}
