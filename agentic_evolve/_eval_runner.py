"""Subprocess entry point for formal evaluation.

Usage: python _eval_runner.py <evaluator_path> <program_path> <metrics_path>

Loads the evaluator module, calls ``evaluate(program_path)``, and writes the
returned metrics dict as JSON to ``metrics_path``. Any exception propagates to
stderr with a nonzero exit code.
"""

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    evaluator_path, program_path, metrics_path = sys.argv[1:4]

    # Let the evaluator import siblings from its own directory.
    sys.path.insert(0, str(Path(evaluator_path).parent))

    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics = module.evaluate(program_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
