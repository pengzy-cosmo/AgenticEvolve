"""Functional test harness with a scripted model boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from mycode import Event

from agentic_evolve.config import load_config
from agentic_evolve.controller import Controller


class ScriptedAgent:
    scenario: ClassVar[Callable[[ScriptedAgent, str], AsyncIterator[Event]]]

    def __init__(self, **kwargs):
        self.session_id = kwargs["session_id"]
        self.system = kwargs["system"]
        self.tools = {spec.name: spec for spec in kwargs["tools"]}
        self.provider = kwargs.get("provider") or "scripted"
        self.model = kwargs["model"]
        self.api_base = kwargs.get("api_base")
        self.context_window = kwargs.get("context_window") or 128_000
        self.max_tokens = kwargs.get("max_tokens") or 16_384
        self.reasoning_effort = kwargs.get("reasoning_effort")
        self.cancelled = False
        self.last_metadata = None

    async def achat(self, prompt: str, **kwargs):
        self.cancelled = False
        async for event in type(self).scenario(self, prompt):
            yield event
            if self.cancelled:
                yield Event("cancelled", {})
                return

    async def call(self, name: str, **args) -> str:
        result = await self.tools[name].runner(None, args)
        self.last_metadata = result.metadata
        return result.output

    def cancel(self) -> None:
        self.cancelled = True


def usage(tokens: int) -> Event:
    return Event("usage", {"context_tokens": tokens})


DEFAULT_EVALUATOR = """\
def evaluate(program_path):
    namespace = {}
    exec(open(program_path).read(), namespace)
    return {
        "score": float(namespace["SCORE"]),
        "feature": float(namespace["FEATURE"]),
    }
"""


@pytest.fixture
def run_task(monkeypatch, tmp_path):
    monkeypatch.setattr("agentic_evolve.island.Agent", ScriptedAgent)

    async def run(
        scenario,
        *,
        islands: int = 1,
        max_commits: int = 1,
        max_total_tokens: int | None = None,
        max_requests: int | None = None,
        migration_interval: int = 10,
        migration_rate: float = 1.0,
        max_analyses_per_commit: int = 10,
        context_window: int | None = None,
        max_tokens: int | None = None,
        compact_threshold: float = 0.7,
        safety_threshold: float = 0.9,
        timeout: float = 2.0,
        initial_program: str = "SCORE = 0\nFEATURE = 0\n",
        evaluator: str = DEFAULT_EVALUATOR,
        data: dict[str, str] | None = None,
    ) -> Path:
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "initial_program.py").write_text(initial_program)
        (task_dir / "evaluator.py").write_text(evaluator)
        if data is not None:
            (task_dir / "data").mkdir()
            for filename, content in data.items():
                (task_dir / "data" / filename).write_text(content)

        llm = {"model": "deepseek-flash"}
        if context_window is not None:
            llm["context_window"] = context_window
        if max_tokens is not None:
            llm["max_tokens"] = max_tokens
        evolution = {
            "num_islands": islands,
            "max_commits": max_commits,
            "migration_interval": migration_interval,
            "migration_rate": migration_rate,
            "max_analyses_per_commit": max_analyses_per_commit,
        }
        if max_total_tokens is not None:
            evolution["max_total_tokens"] = max_total_tokens
        if max_requests is not None:
            evolution["max_requests"] = max_requests

        task = {
            "initial_program": "initial_program.py",
            "evaluator": "evaluator.py",
            "description": "Improve SCORE while preserving diverse FEATURE values.",
        }
        if data is not None:
            task["data"] = "data"

        config_path = task_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "task": task,
                    "llm": llm,
                    "evolution": evolution,
                    "map_elites": {
                        "fitness": "score",
                        "features": ["feature"],
                        "feature_bins": 10,
                    },
                    "executor": {"timeout": timeout, "output_limit": 10_000},
                    "compact": {
                        "compact_threshold": compact_threshold,
                        "safety_threshold": safety_threshold,
                    },
                },
                sort_keys=False,
            )
        )

        ScriptedAgent.scenario = scenario
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        await Controller(load_config(config_path), run_dir).run()
        return run_dir

    return run
