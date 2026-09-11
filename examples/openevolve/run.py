"""Run OpenEvolve on one of our example tasks.

Usage:
    uv run --group openevolve examples/openevolve/run.py --task examples/online_bin_packing \
        --llm llm.yaml --iterations 180 --seed 0 [--runs-dir runs/openevolve]

The task directory holds config.yaml (task description, evaluator, initial
program), the files those refer to, and openevolve.yaml with the OpenEvolve
settings that do not depend on the model. The llm file names the model for
OpenEvolve's OpenAI-compatible client:

    model: deepseek-flash
    api_base: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    max_tokens: 384000              # the model's output limit, same value as our side
    max_tokens_field: max_tokens    # the API's field for a whole response incl. reasoning
    reasoning_effort: high          # optional; omitted means the provider default

Mirrors openevolve-run.py's assembly (load_config -> OpenEvolve -> run) and
adds: the task description as the system message, our client (output-limit
field, usage records), per-run log directories, a snapshot of the task
directory (the run executes from the snapshot), and run.json with the
effective request parameters. Evaluation and request records land in evals/
and usage/ inside the run directory; OpenEvolve's own output goes to
openevolve_output/.
"""

import argparse
import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from llm_client import make_client
from openevolve import OpenEvolve
from openevolve import __version__ as openevolve_version
from openevolve.config import load_config

from agentic_evolve.config import load_config as load_task

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/openevolve"))
    args = parser.parse_args()

    # Snapshot first, then run everything from the snapshot: evaluation and
    # the initial program are immune to workspace edits during the run.
    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-seed{args.seed}"
    snapshot = run_dir / "snapshot"
    shutil.copytree(args.task, snapshot, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    (snapshot / "openevolve.yaml").rename(snapshot / "openevolve_config.yaml")
    for name in ("evaluator_adapter.py", "llm_client.py"):
        shutil.copy(HERE / name, snapshot / name)

    task = load_task(snapshot / "config.yaml")
    llm = yaml.safe_load(args.llm.read_text())
    (run_dir / "evals").mkdir()
    (run_dir / "usage").mkdir()
    # The adapter and the client read these; workers inherit the environment.
    os.environ["EVOLVE_TASK_DIR"] = str(snapshot)
    os.environ["EVOLVE_EVAL_LOG_DIR"] = str(run_dir / "evals")
    os.environ["EVOLVE_USAGE_LOG_DIR"] = str(run_dir / "usage")
    os.environ["EVOLVE_MAX_TOKENS_FIELD"] = llm["max_tokens_field"]

    config = load_config(snapshot / "openevolve_config.yaml")
    config.random_seed = args.seed
    config.prompt.system_message = task.task.description
    config.llm.primary_model = llm["model"]
    config.llm.api_base = llm["api_base"]
    config.llm.api_key = os.environ[llm["api_key_env"]]
    config.llm.max_tokens = llm["max_tokens"]
    config.llm.reasoning_effort = llm.get("reasoning_effort")
    config.llm.rebuild_models()
    config.llm.update_model_params(
        {"system_message": config.prompt.system_message, "init_client": make_client},
        overwrite=True,
    )

    controller = OpenEvolve(
        initial_program_path=task.task.initial_program,
        evaluation_file=str(snapshot / "evaluator_adapter.py"),
        config=config,
        output_dir=str(run_dir / "openevolve_output"),
    )
    # Effective request parameters, read back after the controller derived the LLM seed.
    model = config.llm.models[0]
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "task": str(args.task),
                "fitness": task.map_elites.fitness,
                "llm": llm,
                "seed": args.seed,
                "iterations": args.iterations,
                "openevolve": openevolve_version,
                "request": {
                    "model": model.name,
                    "api_base": model.api_base,
                    "temperature": model.temperature,
                    llm["max_tokens_field"]: model.max_tokens,
                    "reasoning_effort": model.reasoning_effort,
                    "timeout": model.timeout,
                    "retries": model.retries,
                    "seed": model.random_seed,
                },
            },
            indent=2,
        )
    )
    asyncio.run(controller.run(iterations=args.iterations))


if __name__ == "__main__":
    main()
