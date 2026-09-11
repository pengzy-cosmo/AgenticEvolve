"""Dataclass configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TaskConfig:
    initial_program: str
    evaluator: str
    description: str
    data: str | None = None


@dataclass
class LLMConfig:
    model: str
    provider: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None


@dataclass
class EvolutionConfig:
    num_islands: int = 3
    max_commits: int = 30
    max_analyses_per_commit: int = 10
    migration_interval: int = 3
    migration_rate: float = 0.3
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_analyses_per_commit < 0:
            raise ValueError("max_analyses_per_commit must be >= 0")


@dataclass
class MapElitesConfig:
    features: list[str]
    fitness: str = "combined_score"
    feature_bins: int = 10


@dataclass
class ExecutorConfig:
    timeout: float = 120.0
    output_limit: int = 10_000


@dataclass
class CompactConfig:
    compact_threshold: float = 0.7
    safety_threshold: float = 0.9


@dataclass
class Config:
    task: TaskConfig
    llm: LLMConfig
    evolution: EvolutionConfig
    map_elites: MapElitesConfig
    executor: ExecutorConfig
    compact: CompactConfig


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, override: str | Path | None = None) -> Config:
    """Load and optionally override a run config."""

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if override is not None:
        override_raw = yaml.safe_load(Path(override).read_text()) or {}
        _merge(raw, override_raw)

    task = TaskConfig(**raw["task"])
    base = path.resolve().parent
    for field in ("initial_program", "evaluator", "data"):
        if (value := getattr(task, field)) is not None:
            setattr(task, field, str(base / value))

    return Config(
        task=task,
        llm=LLMConfig(**raw["llm"]),
        evolution=EvolutionConfig(**raw.get("evolution", {})),
        map_elites=MapElitesConfig(**raw["map_elites"]),
        executor=ExecutorConfig(**raw.get("executor", {})),
        compact=CompactConfig(**raw.get("compact", {})),
    )
