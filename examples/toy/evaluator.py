"""Evaluator: score the search algorithm over repeated trials.

Runs run_search() several times and scores the returned points against the
known global minimum, using the evaluator's own copy of the objective.

Metrics:
- value_score: 1 / (1 + mean value gap to the global minimum).
- distance_score: 1 / (1 + mean distance to the global minimum).
- reliability: fraction of trials returning a valid point. [0, 1]
- speed_score: 1 / (1 + mean trial seconds / 0.05). [0, 1]
- combined_score: 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability.
"""

import importlib.util
import math
import time

TRIALS = 10
GLOBAL_MIN_X = -1.7041
GLOBAL_MIN_Y = 0.6775
GLOBAL_MIN_VALUE = -1.5187


def objective(x, y):
    return math.sin(x) * math.cos(y) + math.sin(x * y) + (x * x + y * y) / 20


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "run_search"):
        raise ValueError("program must define run_search() -> (x, y, value)")

    values, distances, times = [], [], []
    for _ in range(TRIALS):
        try:
            start = time.perf_counter()
            result = program.run_search()
            elapsed = time.perf_counter() - start
            x, y = float(result[0]), float(result[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
        except Exception:
            continue
        values.append(objective(x, y))
        distances.append(math.dist((x, y), (GLOBAL_MIN_X, GLOBAL_MIN_Y)))
        times.append(elapsed)

    if not values:
        raise ValueError(
            "all trials failed: run_search() must return (x, y, value) without crashing"
        )

    mean_value = sum(values) / len(values)
    mean_distance = sum(distances) / len(distances)
    mean_time = sum(times) / len(times)

    value_score = 1.0 / (1.0 + max(mean_value - GLOBAL_MIN_VALUE, 0.0))
    distance_score = 1.0 / (1.0 + mean_distance)
    reliability = len(values) / TRIALS
    return {
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability": reliability,
        "speed_score": 1.0 / (1.0 + mean_time / 0.05),
        "combined_score": 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability,
    }
