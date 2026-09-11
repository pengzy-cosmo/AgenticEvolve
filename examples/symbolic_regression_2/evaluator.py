"""Evaluator: fit parameters on the observed samples, score hidden interpolation and extrapolation.

This file holds the ground-truth formula; the model never sees it. The
observed samples are regenerated here exactly as exported to
data/observations.csv (seed 11, x in [-2, 2]^3, noise 0.1, rounded to 4
decimals).

Metrics:
- interpolation_score / extrapolation_score: 1 / (1 + NMSE) on hidden samples
  inside / outside the observed input box.
- combined_score: 0.5 * interpolation_score + 0.5 * extrapolation_score.
- complexity: operation nodes in predict and the local functions it calls,
  plus the parameter count, normalized by 40, capped at 1.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _truth(x):
    return 2.0 * x[:, 0] / (1 + x[:, 2] ** 2) + 1.5 * np.sin(1.3 * x[:, 0] * x[:, 1])


def _public_samples():
    rng = np.random.default_rng(11)
    x = rng.uniform(-2, 2, size=(120, 3))
    y = _truth(x) + rng.normal(0, 0.1, 120)
    return np.round(x, 4), np.round(y, 4)


def _interp_samples(rng, n):
    x = rng.uniform(-2, 2, size=(n, 3))
    return x, _truth(x) + rng.normal(0, 0.1, n)


def _extrap_samples(rng, n):
    x = rng.uniform(-4, 4, size=(3 * n, 3))
    x = x[np.max(np.abs(x), axis=1) > 2][:n]
    return x, _truth(x) + rng.normal(0, 0.1, n)


X, y = _public_samples()
_rng = np.random.default_rng(42)
INTERP = _interp_samples(_rng, 200)
EXTRAP = _extrap_samples(_rng, 200)


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
        "combined_score": 0.5 * interpolation + 0.5 * extrapolation,
        "interpolation_score": interpolation,
        "extrapolation_score": extrapolation,
        "complexity": _complexity(Path(program_path).read_text(), len(program.PARAMETER_BOUNDS)),
    }
