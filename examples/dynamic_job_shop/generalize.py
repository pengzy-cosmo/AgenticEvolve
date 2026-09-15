"""Generalization check: evolved dispatching rules on arrival streams they never saw.

Runs each program on the five formal scenarios with two seed sets:
- hidden: the formal seeds, to confirm a program's archived score;
- fresh: 10 seeds drawn from a different generator.

The metric is the formal one: per scenario, mean tardiness over the seeds
divided by R1's on the same seeds; combined_score is the inverse geometric
mean across scenarios. R1 is simulated here on both sets. As in the formal
evaluation, the program is reloaded before every simulation.

Usage: uv run examples/dynamic_job_shop/generalize.py [--raw FILE] PROGRAM.py ...
Prints CSV: program,set,combined_score,error. --raw writes every simulation:
program,set,utilization,due_factor,seed,tardiness,error
"""

import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import EPS, SCENARIOS, SEEDS, r1
from simulator import simulate

SEED_SETS = {
    "hidden": SEEDS,
    "fresh": [int(s) for s in np.random.default_rng(2718281).integers(1 << 31, size=10)],
}
if set(SEED_SETS["hidden"]) & set(SEED_SETS["fresh"]):
    raise SystemExit("fresh seeds overlap the hidden seeds")
R1 = "r1"


def _load(program):
    spec = importlib.util.spec_from_file_location("program", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.priority


def run(job):
    program, _, scenario, seed = job
    try:
        priority = r1 if program == R1 else _load(program)
        return job, simulate(priority, SCENARIOS[scenario], seed)[0], ""
    except Exception as e:  # the program's failure is the result
        return job, None, f"{type(e).__name__}: {e}"


def _csv(*fields):
    return ",".join(str(f).replace(",", ";").replace("\n", " ") for f in fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("programs", nargs="+")
    args = parser.parse_args()
    programs = [R1, *args.programs]
    jobs = [
        (program, name, scenario, seed)
        for program in programs
        for name, seeds in SEED_SETS.items()
        for scenario in range(len(SCENARIOS))
        for seed in seeds
    ]
    with Pool() as pool:
        results = pool.map(run, jobs)
    tardiness = {job: value for job, value, _ in results}
    errors = {}
    for (program, name, _, _), _, error in results:
        if error:
            errors.setdefault((program, name), error)
    if args.raw:
        rows = ["program,set,utilization,due_factor,seed,tardiness,error"]
        for (program, name, scenario, seed), value, error in results:
            s = SCENARIOS[scenario]
            rows.append(_csv(program, name, s.utilization, s.due_factor, seed, value, error))
        args.raw.write_text("\n".join(rows) + "\n")

    def scenario_means(program, name):
        return [
            np.mean([tardiness[program, name, scenario, seed] for seed in SEED_SETS[name]])
            for scenario in range(len(SCENARIOS))
        ]

    print("program,set,combined_score,error")
    for program in programs:
        for name in SEED_SETS:
            if (program, name) in errors:
                print(_csv(program, name, "", errors[program, name]))
                continue
            means = np.maximum(scenario_means(program, name), EPS)
            reference = np.maximum(scenario_means(R1, name), EPS)
            print(_csv(program, name, f"{1 / np.exp(np.mean(np.log(means / reference))):.6f}", ""))


if __name__ == "__main__":
    main()
