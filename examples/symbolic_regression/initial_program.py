"""Driven nonlinear oscillator: evolve equation()."""

import numpy as np


def equation(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Mathematical function for Acceleration in a driven nonlinear oscillator.

    Args:
        x: observations of Position at time t.
        t: observations of Time.
        v: observations of Velocity at time t.
        params: 10 numeric constants, fitted by the evaluator.

    Returns:
        Acceleration in a driven nonlinear oscillator for each observation.
    """
    return params[0] * x + params[1] * t + params[2] * v + params[3]
