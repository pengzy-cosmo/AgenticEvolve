#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "pyyaml", "scipy", "sympy"]
# ///
"""Generate one task directory per LSR-Synth problem of LLM-SRBench.

tasks/<name>/ receives config.yaml, initial_program.py (LLM-SR's linear
skeleton), copies of evaluator.py and expression.py, and data/train.csv with the 4000 training
observations, target column first. The test sets stay in the dataset; score.py
reads them from there. The generated directories are not tracked.

The description gives the model what LLM-SR's prompt gives: the target and
input variables with their meanings, plus how the evaluator fits and scores.

Usage: uv run examples/llm_srbench/make_tasks.py [--out DIR]
"""

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import Problem, problems  # noqa: E402
from expression import MAX_SIZE, RULE  # noqa: E402

BUDGET = {"num_islands": 3, "max_commits": 333, "max_requests": 333, "max_idle_turns": 5}


class _Dumper(yaml.SafeDumper):
    """Multi-line strings as literal blocks, so the description stays readable."""

    def represent_scalar(self, tag, value, style=None):
        if style is None and isinstance(value, str) and "\n" in value:
            style = "|"
        return super().represent_scalar(tag, value, style)


def _description(p: Problem) -> str:
    target, inputs = p.descriptions[0], p.descriptions[1:]
    on = ", ".join(inputs[:-1]) + f", and {inputs[-1]}" if len(inputs) > 1 else inputs[0]
    signature = ", ".join(p.symbols[1:])
    columns = ", ".join(p.symbols)
    return (
        f"Find the mathematical function skeleton that represents {target}, given data on {on}.\n"
        "\n"
        f"Complete equation({signature}, params) in program.py. The inputs are numpy\n"
        "arrays of observations ("
        + "; ".join(f"{s}: {d}" for s, d in zip(p.symbols[1:], inputs, strict=True))
        + "),\n"
        "params is a numpy array of 10 numeric constants. Return a numpy array with\n"
        f"one value of {target} per observation.\n"
        "\n"
        f"The {len(p.train)} observations have columns {columns}, the target first;\n"
        "with the analysis tool they are in data/train.csv.\n"
        "\n"
        "The formal evaluation fits params on these observations by least squares from\n"
        "all ones and reports mse and nmse (mse / variance of the target);\n"
        f"combined_score = -nmse is the fitness, complexity = nodes / {MAX_SIZE}. A non-finite\n"
        "mse fails the evaluation.\n"
        "\n" + textwrap.fill(RULE, 80) + "\n"
    )


def _initial_program(p: Problem) -> str:
    inputs = p.symbols[1:]
    args = "".join(
        f"        {s}: observations of {d}.\n"
        for s, d in zip(inputs, p.descriptions[1:], strict=True)
    )
    body = (
        " + ".join(f"params[{i}] * {s}" for i, s in enumerate(inputs)) + f" + params[{len(inputs)}]"
    )
    return (
        f'"""LLM-SRBench {p.name}: evolve equation()."""\n'
        "\n"
        "import numpy as np\n"
        "\n"
        "\n"
        f"def equation({', '.join(f'{s}: np.ndarray' for s in inputs)}, params: np.ndarray) -> np.ndarray:\n"
        f'    """Mathematical function for {p.descriptions[0]}.\n'
        "\n"
        "    Args:\n"
        f"{args}"
        "        params: 10 numeric constants, fitted by the evaluator.\n"
        "\n"
        "    Returns:\n"
        f"        {p.descriptions[0]} for each observation.\n"
        '    """\n'
        f"    return {body}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "tasks")
    args = parser.parse_args()
    for p in problems():
        task = args.out / p.name
        (task / "data").mkdir(parents=True, exist_ok=True)
        np.savetxt(
            task / "data/train.csv",
            p.train,
            delimiter=",",
            header=",".join(p.symbols),
            comments="",
            fmt="%.17g",  # round-trips the float64 values every method receives
        )
        (task / "initial_program.py").write_text(_initial_program(p))
        for name in ("evaluator.py", "expression.py"):
            shutil.copy(HERE / name, task / name)
        config = {
            "task": {
                "initial_program": "initial_program.py",
                "evaluator": "evaluator.py",
                "data": "data",
                "description": _description(p),
            },
            "llm": {"model": "deepseek-flash"},
            "evolution": BUDGET,
            "map_elites": {"features": ["complexity"]},
            "executor": {"timeout": 30},  # LLM-SR's evaluation time limit, for all executions
        }
        (task / "config.yaml").write_text(yaml.dump(config, Dumper=_Dumper, sort_keys=False))
    print(f"wrote {len(list(args.out.iterdir()))} tasks to {args.out}")


if __name__ == "__main__":
    main()
