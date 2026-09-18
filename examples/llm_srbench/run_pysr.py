#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "pysr>=2.4", "sympy"]
# ///
"""Run PySR on one LSR-Synth problem, with the search settings SR-Scientist
published for this benchmark (Xia et al., ICLR 2026, Appendix A): 125
iterations of 550 cycles, 15 populations of 33, maxsize 30, operators
+ - * / pow and exp log sqrt sin cos with their nesting constraints. Their
benchmark-wide cap of 100,000 candidates per problem is not applied: with
these settings it stops the search inside the first iteration.

The final equation is the lowest-loss member of PySR's Pareto front whose
size under the shared count (expression.py) is within MAX_SIZE (the two counts
agree except for rare operator-specific cases). PySR fits its own constants;
best.py carries them as literals and ignores params, so score.py's refit
leaves it unchanged.

Usage: uv run examples/llm_srbench/run_pysr.py --problem CRK21 --seed 1 [--runs-dir runs/pysr]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import sympy as sp
from pysr import PySRRegressor
from sympy.printing.numpy import NumPyPrinter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import problems  # noqa: E402
from expression import MAX_SIZE, size  # noqa: E402

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

    symbols = {name: sp.Symbol(name) for name in inputs}
    front = model.equations_.sort_values("loss")
    for row in front.itertuples():
        # as PySR wrote it: sympy's own simplification would change the node count
        expression = sp.sympify(row.equation.replace("^", "**"), locals=symbols, evaluate=False)
        if size(expression) <= MAX_SIZE:
            break
    else:
        raise RuntimeError(f"no equation on the Pareto front is within {MAX_SIZE} nodes")
    code = NumPyPrinter().doprint(expression).replace("numpy.", "np.")
    (run_dir / "best.py").write_text(
        f'"""LLM-SRBench {problem.name}: PySR\'s equation, constants fitted by PySR."""\n\n'
        "import numpy as np\n\n\n"
        f"def equation({', '.join(inputs)}, params):\n"
        f"    return {code}\n"
    )
    model.equations_.to_csv(run_dir / "equations.csv", index=False)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "problem": problem.name,
                "subset": problem.subset,
                "seed": args.seed,
                "config": CONFIG,
                "best_equation": sp.sstr(expression),
                "best_train_loss": float(row.loss),
                "size": size(expression),
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
