"""Function minimization: evolve a search algorithm.

The evaluator calls run_search() repeatedly and scores the returned points
against the global minimum of evaluate_function. The evaluator uses its own
copy of evaluate_function, so modifying it here only misleads your search.
"""

import math
import random


def evaluate_function(x, y):
    """The multimodal 2D objective being minimized. Do not modify."""
    return math.sin(x) * math.cos(y) + math.sin(x * y) + (x * x + y * y) / 20


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """A simple random search that often gets stuck in local minima."""
    best_x = random.uniform(*bounds)
    best_y = random.uniform(*bounds)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        x = random.uniform(*bounds)
        y = random.uniform(*bounds)
        value = evaluate_function(x, y)
        if value < best_value:
            best_x, best_y, best_value = x, y, value

    return best_x, best_y, best_value


def run_search():
    return search_algorithm()
