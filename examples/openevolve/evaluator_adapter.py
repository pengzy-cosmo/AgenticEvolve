"""OpenEvolve evaluator for a shared task.

Runs the same evaluator.py through the same subprocess Executor as our
framework, so both systems face identical execution constraints: fresh temp
dir, pinned BLAS threads, 60 s hard kill of the process group. Every call is
appended to a per-process evals-<pid>.jsonl in $EVOLVE_EVAL_LOG_DIR.

Failures return zero fitness with placeholder features so OpenEvolve can
store and bin the program; the error text reaches the next prompt through
its artifacts channel. The log keeps the real status.
"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

from agentic_evolve.config import load_config
from agentic_evolve.executor import EvaluationError, Executor

# run.py points this at the run's snapshot so evaluation is immune to
# workspace edits made while the run is in flight.
_cfg = load_config(Path(os.environ["EVOLVE_TASK_DIR"]) / "config.yaml")
_executor = Executor(_cfg.executor, Path(_cfg.task.evaluator))

# Zero fitness and zero features, so OpenEvolve can still store and bin the program.
FAILURE_METRICS = {_cfg.map_elites.fitness: 0.0} | {f: 0.0 for f in _cfg.map_elites.features}


def evaluate(program_path: str):
    program = Path(program_path).read_text()
    start = time.perf_counter()
    try:
        metrics = asyncio.run(_executor.evaluate(program))
        status, error = "ok", None
    except EvaluationError as exc:
        metrics, status, error = None, exc.status, str(exc)
    _log(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "duration": round(time.perf_counter() - start, 3),
            "status": status,
            "metrics": metrics,
            "error": error,
            "program": program,
        }
    )
    if metrics is None:
        return EvaluationResult(metrics=dict(FAILURE_METRICS), artifacts={"stderr": error})
    return metrics


def _log(record: dict) -> None:
    log_dir = os.environ.get("EVOLVE_EVAL_LOG_DIR")
    if log_dir:
        with (Path(log_dir) / f"evals-{os.getpid()}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
