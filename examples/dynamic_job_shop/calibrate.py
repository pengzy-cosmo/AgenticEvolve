"""Calibration without an LLM: classical and published dispatching rules on the
scenario grid. Prints per-scenario mean tardiness with its seed-to-seed
variation, each rule's ratio to R1 (the intended fixed reference) and the
geometric mean across scenarios, a paired bootstrap over seeds of that
geometric mean between the strong rules, and wall time per rule.

Usage: uv run examples/dynamic_job_shop/calibrate.py [--seeds 10] [--first-seed 1000]
Rule parameters (ATC and COVERT k, RR utilization) are our choices; the
literature does not fix them.
"""

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np
from simulator import Scenario, simulate

SCENARIOS = [Scenario(u, a) for u in (0.85, 0.95) for a in (2, 4, 6)]
K = 2.0  # ATC and COVERT look-ahead parameter


def slack(q, now):
    return q.due - now - q.remaining_work


RULES = {
    "FIFO": lambda q, now: -q.queued_since,
    "SPT": lambda q, now: -q.pt,
    "EDD": lambda q, now: -q.due,
    "MOD": lambda q, now: (
        -np.maximum(
            q.arrival
            + (q.due - q.arrival) / q.total_work * (q.total_work - q.remaining_work + q.pt),
            now + q.pt,
        )
    ),
    "ATC": lambda q, now: np.exp(-np.maximum(slack(q, now), 0) / (K * q.pt.mean())) / q.pt,
    "COVERT": lambda q, now: np.clip(1 - slack(q, now) / (K * q.remaining_work), 0, 1) / q.pt,
    "PT+WINQ": lambda q, now: -(q.pt + q.winq),
    "2PT+WINQ+NPT": lambda q, now: -(2 * q.pt + q.winq + q.next_pt),
    "RR": lambda q, now: (
        -(
            q.pt * np.exp(q.utilization)
            + q.pt * slack(q, now) / q.remaining_work * np.exp(-q.utilization)
            + q.winq
        )
    ),
    "R1": lambda q, now: (
        -(2 * q.pt + q.pt * np.maximum(slack(q, now), 0) / q.remaining_work + q.winq)
    ),
    "R2": lambda q, now: np.where(
        slack(q, now) > 0,
        -(q.pt + q.pt * np.maximum((q.due - now) / q.remaining_work, 0) + q.winq),
        -(3 * q.pt + q.winq - q.winq / q.pt),
    ),
}
STRONG = ["ATC", "COVERT", "2PT+WINQ+NPT", "RR", "R1", "R2"]
REFERENCE = "R1"
EPS = 1e-4  # positive floor before taking ratios (Ferreira et al. 2021)


def run(job):
    name, scenario, seed = job
    start = time.perf_counter()
    tardiness, decisions, divergent = simulate(RULES[name], scenario, seed)
    return name, scenario, seed, tardiness, divergent / decisions, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1000)
    args = parser.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.seeds)

    jobs = list(itertools.product(RULES, SCENARIOS, seeds))
    with Pool() as pool:
        results = pool.map(run, jobs)
    tard = {(n, s, seed): t for n, s, seed, t, _, _ in results}
    diverg = {(n, s, seed): d for n, s, seed, _, d, _ in results}
    secs = {
        n: sum(t for m, _, _, _, _, t in results if m == n) / len(SCENARIOS) / args.seeds
        for n in RULES
    }

    means = {
        (n, s): np.mean([tard[n, s, seed] for seed in seeds]) for n in RULES for s in SCENARIOS
    }
    cv = {
        (n, s): np.std([tard[n, s, seed] for seed in seeds], ddof=1) / means[n, s]
        for n in RULES
        for s in SCENARIOS
    }

    def geo_vs_reference(name, seed_sample, reference=REFERENCE):
        # Per scenario: mean over the sampled seeds, ratio to the reference rule's
        # mean on the same seeds; then the geometric mean across scenarios.
        logs = [
            np.log(
                max(np.mean([tard[name, s, seed] for seed in seed_sample]), EPS)
                / max(np.mean([tard[reference, s, seed] for seed in seed_sample]), EPS)
            )
            for s in SCENARIOS
        ]
        return float(np.exp(np.mean(logs)))

    geo = {n: geo_vs_reference(n, list(seeds)) for n in RULES}

    head = "".join(f"u={s.utilization} a={s.due_factor}".rjust(17) for s in SCENARIOS)
    print(
        f"mean tardiness over {args.seeds} seeds [seed-to-seed CV]; geo-mean of ratio to {REFERENCE}"
    )
    print(f"{'rule':14s}{head}{'geo':>7s}{'spt-div':>9s}{'s/sim':>7s}")
    for n in sorted(RULES, key=geo.get):
        cells = "".join(f"{means[n, s]:9.1f}[{cv[n, s]:4.2f}]" for s in SCENARIOS)
        sd = np.mean([diverg[n, s, seed] for s in SCENARIOS for seed in seeds])
        print(f"{n:14s}{cells}{geo[n]:7.3f}{sd:9.3f}{secs[n]:7.2f}")

    rng = np.random.default_rng(0)
    samples = [list(rng.choice(list(seeds), size=args.seeds, replace=True)) for _ in range(2000)]
    print(f"\npaired bootstrap over seeds ({len(samples)} resamples): geo(a)/geo(b), 95% interval")
    for a, b in itertools.combinations(STRONG, 2):
        r = np.array([geo_vs_reference(a, sm) / geo_vs_reference(b, sm) for sm in samples])
        lo, hi = np.percentile(r, [2.5, 97.5])
        print(f"  {a:14s} / {b:14s}: {geo[a] / geo[b]:5.3f}  [{lo:5.3f}, {hi:5.3f}]")

    print(
        "\nbest rule per scenario:",
        {
            f"u={s.utilization} a={s.due_factor}": min(RULES, key=lambda n: means[n, s])
            for s in SCENARIOS
        },
    )


if __name__ == "__main__":
    main()
