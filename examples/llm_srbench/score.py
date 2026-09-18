#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "scipy", "sympy"]
# ///
"""Select and score the final equations of runs on one LSR-Synth problem.

Every candidate a run evaluated (Full: archive.jsonl commits; LLM-SR: the
searcher's samples; PySR: equations.csv) goes through the shared trace, check
and fit on the training observations, giving its size and train NMSE; the
run's candidates.csv lists them with the Pareto front (best train NMSE per
size, dominating smaller sizes) and PySR's score, -dlog(loss)/dsize between
front neighbours. The final equation is PySR's model_selection="best": the
highest score among front members within 1.5x the lowest train NMSE, from
training data alone.

The final equation is evaluated on the in-domain and out-of-domain test
sets: nmse = mean squared error / variance of the test targets; acc_0.01 and
acc_0.001 = 1 if at least 95% of the test points with a non-zero target have
relative error below the tolerance (LLM-SRBench's accuracy to tolerance with
SR-Scientist's 5% allowance). The traced expression and, per top-level
summand, its largest training-set magnitude relative to the target's standard
deviation are printed for the structure check against the truth.

Usage: uv run examples/llm_srbench/score.py PROBLEM RUN_DIR_OR_PROGRAM.py ...
Prints CSV: run,id,size,split,nmse,acc_0.01,acc_0.001,error,expression,terms
with split in train, id, ood; candidates.csv holds every candidate's size,
train nmse, front membership, score, fitted params and whether it was
selected. A program file is scored as a single candidate.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dataset import problems  # noqa: E402
from evaluator import fit  # noqa: E402
from expression import check, compile, size, trace  # noqa: E402

TOLERANCES = (0.01, 0.001)
BEST_WITHIN = 1.5


def _equation(code):
    namespace = {}
    exec(code, namespace)
    return namespace["equation"]


def candidates(run, inputs):
    """(id, program code or expression) of every candidate the run evaluated successfully."""
    if run.is_file():
        yield "", run.read_text()
    elif (run / "archive.jsonl").exists():  # Full
        for line in (run / "archive.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["status"] == "ok":
                yield record["id"], record["code"]
    elif (run / "equations.csv").exists():  # PySR
        symbols = {name: sp.Symbol(name) for name in inputs}
        with (run / "equations.csv").open() as f:
            for row in csv.DictReader(f):
                equation = row["equation"].replace("^", "**")
                yield row["complexity"], sp.sympify(equation, locals=symbols, evaluate=False)
    else:  # LLM-SR
        samples = sorted(
            run.glob("llmsr/*/samples/*.json"), key=lambda p: int(p.stem.split("_")[1])
        )
        for path in samples:
            sample = json.load(path.open())
            if sample["score"] is not None:
                yield str(sample["sample_order"]), "import numpy as np\n\n" + sample["function"]


def evaluate(run, inputs, x, y):
    """Every candidate with its expression, size, fitted params and train nmse, or its error."""
    rows = []
    for candidate, program in candidates(run, inputs):
        row = {"id": candidate}
        try:
            if isinstance(program, str):
                program = trace(_equation(program), inputs)
            row["expression"] = program
            expression = check(program)
            params, mse = fit(compile(expression, inputs), x, y)
            if not np.isfinite(mse):
                raise ValueError(f"fit produced a non-finite mse: {mse}")
            row |= {"size": size(expression), "params": params, "nmse": mse / float(np.var(y))}
        except Exception as exc:  # the candidate's failure is its result
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def front(rows):
    """Pareto front by size, sorted; each member gets PySR's score."""
    by_size = {}
    for row in rows:
        if "nmse" in row and row["nmse"] < by_size.get(row["size"], {"nmse": math.inf})["nmse"]:
            by_size[row["size"]] = row
    members = []
    for row in (by_size[s] for s in sorted(by_size)):
        if members and row["nmse"] >= members[-1]["nmse"]:
            continue  # dominated by a smaller member
        if not members:
            row["score"] = 0.0
        elif row["nmse"] == 0:
            row["score"] = math.inf
        else:
            last = members[-1]
            row["score"] = -math.log(row["nmse"] / last["nmse"]) / (row["size"] - last["size"])
        members.append(row)
    return members


def select(rows):
    """The final equation, or None when no candidate was evaluable."""
    members = front(rows)
    if not members:
        return None
    lowest = min(row["nmse"] for row in members)
    return max(
        (row for row in members if row["nmse"] <= BEST_WITHIN * lowest), key=lambda r: r["score"]
    )


def _summands(expression):
    if isinstance(expression, sp.Add):
        return [term for arg in expression.args for term in _summands(arg)]
    return [expression]


def _metrics(equation, params, rows):
    y, x = rows[:, 0], rows[:, 1:]
    with np.errstate(all="ignore"):
        pred = equation(*x.T, params=params)
    if not np.all(np.isfinite(pred)):
        raise ValueError("prediction is not finite")
    nmse = float(np.mean((pred - y) ** 2) / np.var(y))
    nonzero = y != 0
    relative = np.abs((pred[nonzero] - y[nonzero]) / y[nonzero])
    return nmse, *(int(np.mean(relative < tau) >= 0.95) for tau in TOLERANCES)


def _csv(text):
    return str(text).replace(",", ";").replace("\n", " ")


def _write_candidates(path, rows, selected):
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "size", "nmse", "front", "score", "selected", "expression", "params", "error"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["id"],
                    row.get("size", ""),
                    row.get("nmse", ""),
                    int("score" in row),
                    row.get("score", ""),
                    int(row is selected),
                    sp.sstr(row["expression"]) if "expression" in row else "",
                    ";".join(f"{v:.12g}" for v in row["params"]) if "params" in row else "",
                    row.get("error", ""),
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    problem = next(p for p in problems() if p.name == args.problem)
    inputs = problem.symbols[1:]
    x, y = problem.train[:, 1:], problem.train[:, 0]
    splits = {"train": problem.train, "id": problem.test, "ood": problem.ood_test}

    print("run,id,size,split,nmse,acc_0.01,acc_0.001,error,expression,terms")
    for run in args.runs:
        rows = evaluate(run, inputs, x, y)
        selected = select(rows)
        if run.is_dir():
            _write_candidates(run / "candidates.csv", rows, selected)
        if selected is None:
            print(f"{run},,,,,,,{_csv(rows[0]['error'] if rows else 'no candidates')},,")
            continue
        expression, params = selected["expression"], selected["params"]
        equation = compile(expression, inputs)
        with np.errstate(all="ignore"):
            terms = ";".join(
                f"{np.max(np.abs(compile(term, inputs)(*x.T, params=params))) / np.std(y):.1e}"
                for term in _summands(expression)
            )
        for split, data in splits.items():
            try:
                nmse, *acc = _metrics(equation, params, data)
                result = f"{nmse:.6e},{acc[0]},{acc[1]},"
            except ValueError as exc:  # non-finite predictions on this split
                result = f",,,{_csv(exc)}"
            print(
                f"{run},{selected['id']},{selected['size']},{split},{result},"
                f"{_csv(sp.sstr(expression))},{terms}"
            )


if __name__ == "__main__":
    main()
