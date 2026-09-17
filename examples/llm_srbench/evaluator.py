"""Evaluator: fit the parameters on the training observations and score the fit.

The fit is LLM-SR's (Shojaee et al., ICLR 2025): scipy BFGS from all ones
minimizing the mean squared error, 10 parameters. The same function runs in
the LLM-SR baseline and in score.py, so every method's equation is fitted
identically. data/train.csv next to this file holds the observations, target
column first.

Metrics:
- mse, nmse: mean squared error of the fitted equation and mse / var(target).
- combined_score: -nmse, the fitness.
- complexity: operation nodes reachable from equation plus the distinct params
  entries it uses, / 40, capped at 1.

A non-finite mse fails the evaluation.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

N_PARAMS = 10


def fit(equation, x, y):
    """LLM-SR's parameter fit. x has one column per input; returns (params, mse)."""

    def loss(params):
        return np.mean((equation(*x.T, params) - y) ** 2)

    with np.errstate(all="ignore"):
        result = minimize(loss, np.ones(N_PARAMS), method="BFGS")
    return result.x, float(result.fun)


_OP_NODES = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.BoolOp, ast.IfExp)


def _complexity(source):
    """Operations reachable from equation plus distinct params entries, / 40, capped at 1.

    Follows calls into functions defined in the program: the call itself is
    not counted, the callee body is counted once.
    """
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "equation" not in functions:
        raise ValueError("program must define equation as a top-level def")
    queue, visited, ops, params = ["equation"], set(), 0, set()
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
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "params"
            ):
                params.add(ast.dump(node.slice))
    return min((ops + len(params)) / 40, 1.0)


def evaluate(program_path):
    rows = np.loadtxt(Path(__file__).parent / "data/train.csv", delimiter=",", skiprows=1)
    y, x = rows[:, 0], rows[:, 1:]

    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "equation"):
        raise ValueError("program must define equation(..., params)")

    _, mse = fit(program.equation, x, y)
    if not np.isfinite(mse):
        raise ValueError(f"fit produced a non-finite mse: {mse}")
    nmse = mse / float(np.var(y))
    return {
        "combined_score": -nmse,
        "mse": mse,
        "nmse": nmse,
        "complexity": _complexity(Path(program_path).read_text()),
    }
