"""Evaluator: dynamic job shop dispatching on hidden arrival streams.

The program's priority(queue, now) picks the next operation on every machine
with a choice (see simulator.py). Formal evaluation runs 5 scenarios x 10
hidden seeds. Per scenario the mean tardiness over seeds is divided by the
frozen reference value of rule R1 (Ferreira et al. 2021) on the same seeds;
the geometric mean of these ratios across scenarios is the reported
tardiness_ratio. Zero tardiness is floored at 1e-4 before the ratio.

Metrics:
- tardiness_ratio: geometric mean over scenarios of mean tardiness / R1's.
- combined_score: 1 / tardiness_ratio, the fitness; 1.0 matches R1.
- complexity: operation nodes reachable from priority, / 40, capped at 1.
- spt_divergence: fraction of decisions that did not pick a shortest waiting
  operation (SPT's choice).

Running this file regenerates reference.json (R1 on the hidden seeds); it must
be regenerated whenever the simulator or the hidden set changes.
"""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
from simulator import Scenario, simulate

SCENARIOS = [Scenario(u, a) for u, a in ((0.85, 2), (0.85, 4), (0.95, 2), (0.95, 4), (0.95, 6))]
SEEDS = [int(s) for s in np.random.default_rng(6180339).integers(1 << 31, size=10)]
EPS = 1e-4
REFERENCE_PATH = Path(__file__).parent / "reference.json"


def r1(queue, now):
    """Ferreira et al. 2021 rule R1: 2 PT + PT * S+ / RPT + WINQ, minimized."""
    slack = np.maximum(queue.due - now - queue.remaining_work, 0)
    return -(2 * queue.pt + queue.pt * slack / queue.remaining_work + queue.winq)


def run_scenarios(load_priority):
    """Per-scenario mean tardiness over the hidden seeds, plus decision counts.

    load_priority is called before every simulation run, so a program's module
    state lives for one run only.
    """
    means, decisions, divergent = [], 0, 0
    for scenario in SCENARIOS:
        results = [simulate(load_priority(), scenario, seed) for seed in SEEDS]
        means.append(float(np.mean([r[0] for r in results])))
        decisions += sum(r[1] for r in results)
        divergent += sum(r[2] for r in results)
    return means, decisions, divergent


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
    def load_priority():
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "priority"):
            raise ValueError("program must define priority(queue, now) -> np.ndarray")
        return program.priority

    reference = json.loads(REFERENCE_PATH.read_text())
    means, decisions, divergent = run_scenarios(load_priority)
    ratio = float(
        np.exp(
            np.mean(
                [np.log(max(m, EPS) / max(r, EPS)) for m, r in zip(means, reference, strict=True)]
            )
        )
    )
    return {
        "combined_score": 1.0 / ratio,
        "tardiness_ratio": ratio,
        "complexity": _complexity(Path(program_path).read_text()),
        "spt_divergence": divergent / decisions,
    }


if __name__ == "__main__":
    means, _, _ = run_scenarios(lambda: r1)
    REFERENCE_PATH.write_text(json.dumps(means) + "\n")
    print("R1 mean tardiness per scenario:", means)
