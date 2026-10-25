#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "pysr>=2.4"]
# ///
"""Run PySR on one LSR-Synth problem, with the search settings SR-Scientist
published for this benchmark (Xia et al., ICLR 2026, Appendix A): 125
iterations of 550 cycles, 15 populations of 33, maxsize 30, operators
+ - * / pow and exp log sqrt sin cos with their nesting constraints. Their
benchmark-wide cap of 100,000 candidates per problem is not applied: with
these settings it stops the search inside the first iteration.

The run directory receives equations.csv, PySR's Pareto front with its own
fitted constants, from which score.py selects and scores the final equation
under the shared count and rules, and run.json.

Usage: uv run experiments/llm_srbench/run_pysr.py --problem CRK21 --seed 1 [--runs-dir runs/pysr]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pysr import PySRRegressor

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import problems  # noqa: E402

CONFIG = {
    "niterations": 125,
    "ncycles_per_iteration": 550,
    "populations": 15,
    "population_size": 33,
    "maxsize": 30,
    "binary_operators": ["+", "-", "*", "/", "pow"],
    "unary_operators": ["exp", "log", "sqrt", "sin", "cos"],
    "nested_constraints": {
        "sin": {"sin": 0, "cos": 0},
        "cos": {"sin": 0, "cos": 0},
        "exp": {"exp": 0, "log": 0},
        "log": {"exp": 0, "log": 0},
        "sqrt": {"sqrt": 0},
    },
    "constraints": {"sin": 10, "cos": 10, "exp": 20, "log": 20, "sqrt": 20, "pow": (-1, 20)},
    "weight_randomize": 0.1,
    "model_selection": "accuracy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/pysr"))
    args = parser.parse_args()

    problem = next(p for p in problems() if p.name == args.problem)
    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{problem.name}-seed{args.seed}"
    run_dir.mkdir(parents=True)
    started = datetime.now(UTC)

    inputs = problem.symbols[1:]
    model = PySRRegressor(
        **CONFIG,
        random_state=args.seed,
        output_directory=str(run_dir / "pysr"),
        progress=False,
        verbosity=0,
    )
    model.fit(problem.train[:, 1:], problem.train[:, 0], variable_names=inputs)

    model.equations_.to_csv(run_dir / "equations.csv", index=False)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "problem": problem.name,
                "subset": problem.subset,
                "seed": args.seed,
                "config": CONFIG,
                "started": started.isoformat(),
                "finished": datetime.now(UTC).isoformat(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
