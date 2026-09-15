"""Generalization check: evolved priority functions on sequences they never saw.

Every set matches the description (120 items, sizes uniform in [20, 100],
capacity 150):
- hidden: the formal evaluation set, to confirm a program's archived score;
- or1: the 20 u120 instances of OR-Library in their original item order
  (or1.txt, licence in or1_LICENSE.txt);
- fresh: 100 sequences drawn like the hidden set from a different seed.

Best Fit and the heuristic FunSearch published for the OR datasets
(Romera-Paredes et al. 2024) are reported as references. As in the formal
evaluation, a program is loaded once per set and runs its sequences in order.
excess is (bins used - L2) / L2 over the set; combined_score = 1 / (1 + excess).

Usage: uv run examples/online_bin_packing/generalize.py [--raw FILE] PROGRAM.py ...
Prints CSV: program,set,excess,combined_score,error. --raw writes every sequence:
program,set,sequence,bins,l2
"""

import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import SEQUENCES, _l2_bound, _pack

HERE = Path(__file__).parent


def _read_or1():
    tokens = (HERE / "or1.txt").read_text().split()
    instances, i = [], 1
    for _ in range(int(tokens[0])):
        capacity, count = int(tokens[i + 1]), int(tokens[i + 2])
        if capacity != 150:
            raise ValueError(f"unexpected capacity {capacity}")
        instances.append(np.array(tokens[i + 4 : i + 4 + count], dtype=np.int64))
        i += 4 + count
    return instances


SETS = {
    "hidden": SEQUENCES,
    "or1": _read_or1(),
    "fresh": list(np.random.default_rng(271828182).integers(20, 101, size=(100, 120))),
}


def best_fit(item, bins):
    return -(bins - item)


def funsearch(item, bins):
    """FunSearch's published heuristic for the OR datasets, verbatim in effect."""

    def score(bin_):
        rest = bin_ - item
        for limit, value in (
            (2, 4), (3, 3), (5, 2), (7, 1), (9, 0.9), (12, 0.95),
            (15, 0.97), (18, 0.98), (20, 0.98), (21, 0.98),
        ):  # fmt: skip
            if rest <= limit:
                return value
        return 0.99

    return np.array([score(b) for b in bins])


REFERENCES = {"best_fit": best_fit, "funsearch": funsearch}


def _load(program):
    spec = importlib.util.spec_from_file_location("program", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.priority


def run(job):
    """Bins used per sequence of the set, or the error message."""
    program, name = job
    try:
        priority = REFERENCES[program] if program in REFERENCES else _load(program)
        return job, [_pack(priority, items)[0] for items in SETS[name]], ""
    except Exception as e:  # the program's failure is the result
        return job, None, f"{type(e).__name__}: {e}"


def _csv(*fields):
    return ",".join(str(f).replace(",", ";").replace("\n", " ") for f in fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("programs", nargs="+")
    args = parser.parse_args()
    programs = [*REFERENCES, *args.programs]
    jobs = [(program, name) for program in programs for name in SETS]
    l2 = {name: [_l2_bound(items) for items in sets] for name, sets in SETS.items()}

    raw = ["program,set,sequence,bins,l2"]
    print("program,set,excess,combined_score,error")
    with Pool() as pool:
        for (program, name), bins, error in pool.imap(run, jobs):
            if error:
                print(_csv(program, name, "", "", error), flush=True)
                continue
            excess = (sum(bins) - sum(l2[name])) / sum(l2[name])
            print(_csv(program, name, f"{excess:.6f}", f"{1 / (1 + excess):.6f}", ""), flush=True)
            for i, (used, bound) in enumerate(zip(bins, l2[name], strict=True)):
                raw.append(_csv(program, name, i, used, bound))
    if args.raw:
        args.raw.write_text("\n".join(raw) + "\n")


if __name__ == "__main__":
    main()
