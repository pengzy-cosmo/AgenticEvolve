"""Circle packing: evolve pack().

pack() returns centers, shape (26, 2), and radii, shape (26,), of
non-overlapping circles inside the unit square, maximizing the sum of radii.
"""

import numpy as np

N = 26


def max_radii(centers):
    """Conservative valid radii for fixed centers: half the nearest-neighbor
    distance, capped by the distance to the walls."""
    walls = np.minimum(centers, 1 - centers).min(axis=1)
    dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.minimum(walls, dist.min(axis=1) / 2)


def pack():
    """Baseline: one center circle and two concentric rings."""
    centers = np.zeros((N, 2))
    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.24 * np.cos(angle), 0.5 + 0.24 * np.sin(angle)]
    for i in range(17):
        angle = 2 * np.pi * i / 17
        centers[i + 9] = [0.5 + 0.44 * np.cos(angle), 0.5 + 0.44 * np.sin(angle)]
    return centers, max_radii(centers)
