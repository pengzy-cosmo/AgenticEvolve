"""Symbolic regression: evolve predict() and PARAMETER_BOUNDS.

predict(x, params) maps inputs x, shape (n, 3), to outputs, shape (n,).
The evaluator fits params within PARAMETER_BOUNDS by least squares before
scoring, so express the model structure here and leave constants to params.
"""

import numpy as np

PARAMETER_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]


def predict(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Linear baseline."""
    return params[0] * x[:, 0] + params[1] * x[:, 1] + params[2] * x[:, 2] + params[3]
