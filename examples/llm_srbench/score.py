#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "scipy", "sympy"]
# ///
"""Score final equations on the hidden test sets of one LSR-Synth problem.

Each program goes through the shared trace, check and fit on the training
observations (what every search method used), then is evaluated on the
in-domain and out-of-domain test sets:
- nmse: mean squared error / variance of the test targets;
- acc_0.01, acc_0.001: 1 if at least 95% of the test points with a non-zero
  target have relative error below the tolerance, else 0 (LLM-SRBench's
  accuracy to tolerance with SR-Scientist's 5% allowance).
The traced expression is printed for the structure check against the truth.

Usage: uv run examples/llm_srbench/score.py PROBLEM PROGRAM.py [PROGRAM.py ...]
Prints CSV: program,split,size,nmse,acc_0.01,acc_0.001,error,expression
with split in train, id, ood.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import problems  # noqa: E402
from evaluator import fit, load  # noqa: E402
from expression import check, compile, size, trace  # noqa: E402

TOLERANCES = (0.01, 0.001)


def _metrics(equation, params, rows):
    y, x = rows[:, 0], rows[:, 1:]
    with np.errstate(all="ignore"):
        pred = equation(*x.T, params=params)
    if not np.all(np.isfinite(pred)):
        raise ValueError("prediction is not finite")
    nmse = float(np.mean((pred - y) ** 2) / np.var(y))
    nonzero = y != 0
    relative = np.abs((pred[nonzero] - y[nonzero]) / y[nonzero])
    accuracies = [int(np.mean(relative < tau) >= 0.95) for tau in TOLERANCES]
    return nmse, *accuracies


def _csv(text):
    return str(text).replace(",", ";").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("programs", nargs="+")
    args = parser.parse_args()
    problem = next(p for p in problems() if p.name == args.problem)
    inputs = problem.symbols[1:]
    splits = {"train": problem.train, "id": problem.test, "ood": problem.ood_test}

    print("program,split,size,nmse,acc_0.01,acc_0.001,error,expression")
    for program in args.programs:
        try:
            expr = check(trace(load(program), inputs))
            equation = compile(expr, inputs)
            params, _ = fit(equation, problem.train[:, 1:], problem.train[:, 0])
            for name, rows in splits.items():
                nmse, *acc = _metrics(equation, params, rows)
                print(
                    f"{program},{name},{size(expr)},{nmse:.6e},{acc[0]},{acc[1]},,{_csv(sp.sstr(expr))}"
                )
        except Exception as e:  # the program's failure is the result
            print(f"{program},,,,,,{_csv(f'{type(e).__name__}: {e}')},")


if __name__ == "__main__":
    main()
