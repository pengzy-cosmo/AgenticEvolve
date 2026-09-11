"""Evaluator: fit parameters on the observed samples, score hidden interpolation and extrapolation.

This file holds the ground-truth formula; the model never sees it. The
observed samples are regenerated here exactly as exported to
data/observations.csv (seed 7, x0 in [0, 4], x1 in [-3, 3], noise 0.05,
rounded to 4 decimals). The committed program plus this fit IS the recovered
equation.

Metrics:
- interpolation_score / extrapolation_score: 1 / (1 + NMSE) on hidden samples
  inside / outside the observed input box.
- combined_score: 0.7 * interpolation_score + 0.3 * extrapolation_score.
- complexity: operation nodes in predict and the local functions it calls,
  plus the parameter count, normalized by 40, capped at 1.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _truth(x):
    return 1.8 * np.exp(-0.4 * x[:, 0]) * np.cos(2.2 * x[:, 1]) + 0.6 * x[:, 0] * x[:, 1]


def _public_samples():
    rng = np.random.default_rng(7)
    x = np.column_stack([rng.uniform(0, 4, 150), rng.uniform(-3, 3, 150)])
    y = _truth(x) + rng.normal(0, 0.05, 150)
    return np.round(x, 4), np.round(y, 4)


def _hidden_samples(rng, n, x0_range, x1_range):
    x = np.column_stack([rng.uniform(*x0_range, n), rng.uniform(*x1_range, n)])
    return x, _truth(x) + rng.normal(0, 0.05, n)


X, y = _public_samples()
_rng = np.random.default_rng(42)
INTERP = _hidden_samples(_rng, 200, (0, 4), (-3, 3))
EXTRAP = _hidden_samples(_rng, 200, (4, 6), (-4.5, 4.5))


def _fit(predict, bounds, starts=16, seed=0):
    """Deterministic multi-start bounded least squares on the observed samples."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    best, last_error = None, None
    for _ in range(starts):
        start = rng.uniform(lo, hi)
        try:
            result = least_squares(lambda p: predict(X, p) - y, start, bounds=(lo, hi))
        except Exception as exc:
            last_error = exc
            continue
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        raise ValueError(f"all {starts} fit starts failed, last error: {last_error}")
    return best.x


_OP_NODES = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.BoolOp, ast.IfExp)


def _complexity(source, n_params):
    """Operation count of predict plus n_params, normalized by 40, capped at 1.

    Follows calls into functions defined in the program: the call itself is
    not counted, the callee body is counted once.
    """
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "predict" not in functions:
        raise ValueError("program must define predict as a top-level def")
    queue, visited, ops = ["predict"], set(), 0
    while queue:
        name = queue.pop()
        if name in visited or name not in functions:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
            ):
                queue.append(node.func.id)
            elif isinstance(node, _OP_NODES):
                ops += 1
    return min((ops + n_params) / 40, 1.0)


def _score(predict, params, split):
    x_split, y_split = split
    try:
        pred = np.asarray(predict(x_split, params))
    except Exception:
        return 0.0
    if pred.shape != y_split.shape or not np.all(np.isfinite(pred)):
        return 0.0
    nmse = float(np.mean((pred - y_split) ** 2) / np.var(y_split))
    return 1.0 / (1.0 + nmse)


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "predict") or not hasattr(program, "PARAMETER_BOUNDS"):
        raise ValueError("program must define predict(x, params) and PARAMETER_BOUNDS")

    params = _fit(program.predict, program.PARAMETER_BOUNDS)
    interpolation = _score(program.predict, params, INTERP)
    extrapolation = _score(program.predict, params, EXTRAP)

    return {
        "combined_score": 0.7 * interpolation + 0.3 * extrapolation,
        "interpolation_score": interpolation,
        "extrapolation_score": extrapolation,
        "complexity": _complexity(Path(program_path).read_text(), len(program.PARAMETER_BOUNDS)),
    }
