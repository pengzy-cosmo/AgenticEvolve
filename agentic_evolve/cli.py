"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from agentic_evolve.config import load_config
from agentic_evolve.controller import Controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic LLM evolution runner")
    parser.add_argument("config", help="path to the run config YAML")
    parser.add_argument("--override", help="YAML file merged over the run config")
    parser.add_argument("--runs-dir", default="runs", help="directory to create the run under")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_dir) / run_id

    # Snapshot the whole task directory and run from it: the written config
    # points into task/, and reloading it resolves all paths there.
    task_src = Path(args.config).resolve().parent
    shutil.copytree(
        task_src, run_dir / "task", ignore=shutil.ignore_patterns("__pycache__", ".DS_Store")
    )
    for field in ("initial_program", "evaluator", "data"):
        src = getattr(config.task, field)
        if src is not None:
            setattr(config.task, field, str(Path("task") / Path(src).relative_to(task_src)))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    config = load_config(run_dir / "config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")],
    )
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(__name__).info("run directory: %s", run_dir)

    asyncio.run(Controller(config, run_dir).run())


if __name__ == "__main__":
    main()
