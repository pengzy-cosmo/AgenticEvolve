"""Evaluator: trace the candidate into an expression (expression.py), check it,
fit its parameters on the training observations and score the fit. The
LLM-SRBench experiments (experiments/llm_srbench) use it for every method.

The fit is scipy least_squares on the 10 params from all ones. LLM-SR minimizes
the mean squared error with BFGS instead; on CRK21 that fit stops at train NMSE
7e-7 for the generating equation itself, where least squares reaches 1e-15.

Metrics: mse; nmse = mse / var(target); combined_score = -nmse, the fitness;
complexity = size / MAX_SIZE. A non-finite mse fails the evaluation.
data/train.csv next to this file holds the observations, target column first.
"""

from pathlib import Path

import numpy as np
from expression import MAX_SIZE, N_PARAMS, check, compile, load, size, trace
from scipy.optimize import least_squares


def fit(equation, x, y):
    """Least-squares fit of params from all ones. x has one column per input; returns (params, mse)."""

    def residual(params):
        return equation(*x.T, params=params) - y

    with np.errstate(all="ignore"):
        result = least_squares(residual, np.ones(N_PARAMS))
    return result.x, float(np.mean(result.fun**2))


def evaluate(program_path):
    data = Path(__file__).parent / "data/train.csv"
    inputs = data.read_text().split("\n", 1)[0].split(",")[1:]
    rows = np.loadtxt(data, delimiter=",", skiprows=1)
    y, x = rows[:, 0], rows[:, 1:]

    equation = load(Path(program_path).read_text(), program_path)
    expr = check(trace(equation, inputs))
    _, mse = fit(compile(expr, inputs), x, y)
    if not np.isfinite(mse):
        raise ValueError(f"fit produced a non-finite mse: {mse}")
    nmse = mse / float(np.var(y))
    return {
        "combined_score": -nmse,
        "mse": mse,
        "nmse": nmse,
        "complexity": size(expr) / MAX_SIZE,
    }
