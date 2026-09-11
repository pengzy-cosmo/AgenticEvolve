"""Controller: assemble the run and drive all islands concurrently.

Islands share one event loop. The TaskGroup cancels the remaining islands as
soon as any island raises, which ends the run; there is no checkpoint resume.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from agentic_evolve.archive import Archive
from agentic_evolve.config import Config
from agentic_evolve.executor import Executor
from agentic_evolve.island import Island

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: Config, run_dir: Path):
        self.cfg = config
        self.run_dir = run_dir

    async def run(self) -> None:
        session_dir = self.run_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        archive = Archive(self.cfg.map_elites, self.cfg.evolution.num_islands, self.run_dir)
        task = self.cfg.task
        executor = Executor(
            self.cfg.executor, Path(task.evaluator), Path(task.data) if task.data else None
        )
        initial_program = Path(self.cfg.task.initial_program).read_text()
        started = perf_counter()
        initial_metrics = await executor.evaluate(initial_program)
        (self.run_dir / "initial.json").write_text(
            json.dumps(
                {
                    "metrics": initial_metrics,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "duration": round(perf_counter() - started, 3),
                },
                indent=2,
            )
        )
        logger.info("initial program evaluated: %s", initial_metrics)
        islands = [
            Island(i, self.cfg, archive, executor, session_dir, initial_program, initial_metrics)
            for i in range(self.cfg.evolution.num_islands)
        ]
        agent = islands[0].agent
        logger.info(
            "llm: provider=%s model=%s base_url=%s context_window=%d max_tokens=%d reasoning_effort=%s",
            agent.provider,
            agent.model,
            agent.api_base or "provider default",
            agent.context_window,
            agent.max_tokens,
            agent.reasoning_effort or "provider default",
        )

        try:
            async with asyncio.TaskGroup() as tg:
                for island in islands:
                    tg.create_task(island.run())
        finally:
            if archive.global_best is not None:
                best = archive.records[archive.global_best]
                logger.info(
                    "global best: %s | %s | %s",
                    best.id,
                    best.metrics,
                    best.description,
                )
            else:
                logger.info("no successful commit recorded")
