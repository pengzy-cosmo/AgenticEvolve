#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "scipy"]
# ///
"""Score final equations on the hidden test sets of one LSR-Synth problem.

Each program is refitted on the training observations with the evaluator's
fit (the fit every search method used), then evaluated on the in-domain and
out-of-domain test sets:
- nmse: mean squared error / variance of the test targets;
- acc_0.1, acc_0.01: 1 if at least 95% of the test points with a non-zero
  target have relative error below the tolerance, else 0 (the accuracy to
  tolerance of LLM-SRBench with SR-Scientist's 5% allowance).

Usage: uv run examples/llm_srbench/score.py PROBLEM PROGRAM.py [PROGRAM.py ...]
Prints CSV: program,split,nmse,acc_0.1,acc_0.01,error with split in train, id, ood.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import problems  # noqa: E402
from evaluator import fit  # noqa: E402

TOLERANCES = (0.1, 0.01)


def _load(program):
    spec = importlib.util.spec_from_file_location("program", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.equation


def _metrics(equation, params, rows):
    y, x = rows[:, 0], rows[:, 1:]
    with np.errstate(all="ignore"):
        pred = np.asarray(equation(*x.T, params), dtype=float)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        raise ValueError("prediction is not a finite array of the target's shape")
    nmse = float(np.mean((pred - y) ** 2) / np.var(y))
    nonzero = y != 0
    relative = np.abs((pred[nonzero] - y[nonzero]) / y[nonzero])
    accuracies = [int(np.mean(relative < tau) >= 0.95) for tau in TOLERANCES]
    return nmse, *accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("programs", nargs="+")
    args = parser.parse_args()
    problem = next(p for p in problems() if p.name == args.problem)
    splits = {"train": problem.train, "id": problem.test, "ood": problem.ood_test}

    print("program,split,nmse,acc_0.1,acc_0.01,error")
    for program in args.programs:
        try:
            equation = _load(program)
            params, _ = fit(equation, problem.train[:, 1:], problem.train[:, 0])
            for name, rows in splits.items():
                nmse, *acc = _metrics(equation, params, rows)
                print(f"{program},{name},{nmse:.6e},{acc[0]},{acc[1]},")
        except Exception as e:  # the program's failure is the result
            message = f"{type(e).__name__}: {e}".replace(",", ";").replace("\n", " ")
            print(f"{program},,,,,{message}")


if __name__ == "__main__":
    main()
