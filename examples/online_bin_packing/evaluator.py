"""Evaluator: online bin packing on hidden uniform sequences.

FunSearch's online bin packing setting (Romera-Paredes et al., Nature 2024):
the program scores the feasible bins with priority(item, bins) and the
harness places each arriving item into the highest-scoring one. The hidden
sequences in sequences.txt are OR1-sized, generated once with
numpy.random.default_rng(864202469): 100 sequences of 120 items, integer
sizes uniform in [20, 100], bin capacity 150.

Metrics:
- mean_excess: (total used bins - total L2) / total L2, the paper's
  reporting metric; L2 is the Martello-Toth lower bound of the offline
  optimum.
- combined_score: 1 / (1 + mean_excess).
- complexity: operation nodes reachable from priority, / 40, capped at 1.
- bf_divergence: fraction of placements into a bin looser than Best Fit's
  (tightest feasible) choice, / 0.2, capped at 1.

A NaN score fails the evaluation; +-inf is allowed.
"""

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np

CAPACITY = 150
SEQUENCES = [
    np.array(line.split(), dtype=np.int64)
    for line in (Path(__file__).parent / "sequences.txt").read_text().splitlines()
]


def _l2_bound(items):
    """Martello-Toth L2 lower bound on the offline optimal bin count."""
    best = 0
    for alpha in {0} | {int(s) for s in items if s <= CAPACITY / 2}:
        large = medium = medium_sum = small_sum = 0
        for s in items:
            if s > CAPACITY - alpha:
                large += 1
            elif s > CAPACITY / 2:
                medium += 1
                medium_sum += s
            elif s >= alpha:
                small_sum += s
        free = medium * CAPACITY - medium_sum
        best = max(best, large + medium + max(0, math.ceil((small_sum - free) / CAPACITY)))
    return best


L2S = [_l2_bound(items) for items in SEQUENCES]


def _pack(priority, items):
    """Runs the online loop; returns bins used and non-Best-Fit placements."""
    bins = np.full(len(items), CAPACITY, dtype=np.int64)
    divergent = 0
    for item in items:
        feasible = np.nonzero(bins >= item)[0]
        scores = np.asarray(priority(int(item), bins[feasible]))
        if scores.shape != feasible.shape:
            raise ValueError("priority must return one score per feasible bin")
        if np.isnan(scores).any():
            raise ValueError("priority returned NaN")
        choice = feasible[np.argmax(scores)]
        divergent += int(bins[choice] > bins[feasible].min())
        bins[choice] -= item
    return int((bins < CAPACITY).sum()), divergent


_OP_NODES = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.BoolOp, ast.IfExp)


def _complexity(source):
    """Operation count reachable from priority, normalized by 40, capped at 1.

    Follows calls into functions defined in the program: the call itself is
    not counted, the callee body is counted once.
    """
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "priority" not in functions:
        raise ValueError("program must define priority as a top-level def")
    queue, visited, ops = ["priority"], set(), 0
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
    return min(ops / 40, 1.0)


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "priority"):
        raise ValueError("program must define priority(item, bins) -> np.ndarray")

    used_total = l2_total = divergent_total = 0
    for items, l2 in zip(SEQUENCES, L2S, strict=True):
        used, divergent = _pack(program.priority, items)
        used_total += used
        l2_total += l2
        divergent_total += divergent
    mean_excess = (used_total - l2_total) / l2_total

    return {
        "combined_score": 1.0 / (1.0 + mean_excess),
        "mean_excess": mean_excess,
        "complexity": _complexity(Path(program_path).read_text()),
        "bf_divergence": min(divergent_total / sum(len(items) for items in SEQUENCES) / 0.2, 1.0),
    }
