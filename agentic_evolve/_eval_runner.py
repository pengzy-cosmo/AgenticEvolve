"""Subprocess entry point for formal evaluation.

Usage: python _eval_runner.py <evaluator_path> <program_path> <metrics_path> <error_path>

Write evaluate(program_path)'s metrics to metrics_path. On an exception, write
its type, message and traceback frames to error_path, then re-raise for stderr
logging and a nonzero exit code. Feedback uses this structured record because
candidate-controlled stderr text can imitate a traceback.
"""

import builtins
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    evaluator_path, program_path, metrics_path, error_path = sys.argv[1:5]
    try:
        # Let the evaluator import siblings from its own directory.
        sys.path.insert(0, str(Path(evaluator_path).parent))
        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load evaluator: {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.evaluate(program_path)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "builtin": getattr(builtins, type(exc).__name__, None) is type(exc),
            "message": str(exc),
            "frames": [(f.filename, f.lineno) for f in traceback.extract_tb(exc.__traceback__)],
        }
        Path(error_path).write_text(json.dumps(error))
        raise
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
