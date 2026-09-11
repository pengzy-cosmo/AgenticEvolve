"""Evaluator: pack a fixed instance set across four distributions.

Self-contained: the instances come from fixed private seeds.

Metrics:
- combined_score: mean of lower_bound / used_bins over all instances;
  invalid or crashing instances count as 0.
- reliability: fraction of instances packed validly.
- speed_score: 1 / (1 + mean seconds per instance / 0.002).
"""

import importlib.util
import math
import random
import time

CAPACITY = 100
SIZES = (100, 200, 500)
KINDS = ("uniform", "halves", "thirds", "pairs")
SEEDS = range(1000, 1003)


def _instance(seed, size, kind):
    rng = random.Random(f"{kind}:{size}:{seed}")
    if kind == "uniform":
        return [rng.randint(1, 99) for _ in range(size)]
    if kind == "halves":
        return [rng.randint(40, 60) for _ in range(size)]
    if kind == "thirds":
        return [rng.randint(25, 40) for _ in range(size)]
    half = [rng.randint(1, 99) for _ in range(size // 2)]
    items = half + [CAPACITY - item for item in half]
    if size % 2:
        items.append(rng.randint(1, 99))
    rng.shuffle(items)
    return items


def _quality(items, assignment):
    if len(assignment) != len(items):
        return 0.0
    loads = {}
    for item, bin_id in zip(items, assignment, strict=True):
        if not isinstance(bin_id, int):
            return 0.0
        loads[bin_id] = loads.get(bin_id, 0) + item
    if any(load > CAPACITY for load in loads.values()):
        return 0.0
    return math.ceil(sum(items) / CAPACITY) / len(loads)


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "pack"):
        raise ValueError("program must define pack(items, capacity) -> list[int]")

    qualities, times = [], []
    for kind in KINDS:
        for size in SIZES:
            for seed in SEEDS:
                items = _instance(seed, size, kind)
                start = time.perf_counter()
                try:
                    assignment = program.pack(list(items), CAPACITY)
                    qualities.append(_quality(items, assignment))
                except Exception:
                    qualities.append(0.0)
                times.append(time.perf_counter() - start)

    mean_time = sum(times) / len(times)
    return {
        "combined_score": sum(qualities) / len(qualities),
        "reliability": sum(q > 0 for q in qualities) / len(qualities),
        "speed_score": 1.0 / (1.0 + mean_time / 0.002),
    }
