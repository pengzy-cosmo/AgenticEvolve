"""Archive: full commit history, per-island MAP-Elites, best tracking, migration.

All mutating methods are synchronous by design: they run inside the no-await
section after a commit evaluation, so the single event loop serializes every
global state update without locks.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_evolve.config import MapElitesConfig

logger = logging.getLogger(__name__)


@dataclass
class Behavior:
    """Behavior observed since the previous commit."""

    run_analysis_count: int = 0
    read_commit_ids: list[str] = field(default_factory=list)
    context_tokens: int | None = None
    model_calls: int = 0


@dataclass
class CommitRecord:
    """One formal evaluation. status is "ok" | "error" | "timeout"; successful
    records carry metrics, failed records carry error."""

    id: str
    island_id: int
    code: str
    description: str
    behavior: Behavior
    timestamp: str
    duration: float
    status: str
    metrics: dict | None = None
    error: str | None = None


class Archive:
    def __init__(self, cfg: MapElitesConfig, num_islands: int, run_dir: Path):
        self.cfg = cfg
        self.num_islands = num_islands
        self.records: dict[str, CommitRecord] = {}
        self.maps: dict[int, dict[tuple[int, ...], str]] = {i: {} for i in range(num_islands)}
        self.island_best: dict[int, str | None] = {i: None for i in range(num_islands)}
        self.global_best: str | None = None
        self.archive_path = run_dir / "archive.jsonl"
        self.best_py_path = run_dir / "best.py"
        self.best_json_path = run_dir / "best.json"

    def fitness(self, commit_id: str) -> float:
        metrics = self.records[commit_id].metrics
        assert metrics is not None  # maps and bests hold only successful commits
        return float(metrics[self.cfg.fitness])

    def cell(self, metrics: dict) -> tuple[int, ...]:
        """Bucket normalized feature values uniformly; out-of-range values clamp to the border."""

        coords = []
        for name in self.cfg.features:
            value = min(max(float(metrics[name]), 0.0), 1.0)
            coords.append(min(int(value * self.cfg.feature_bins), self.cfg.feature_bins - 1))
        return tuple(coords)

    def add(self, record: CommitRecord) -> CommitRecord | None:
        """Record a commit. A successful evaluation updates MAP-Elites and bests
        and returns the island's current best; a failure is only persisted."""

        self.records[record.id] = record
        # Absent field = not applicable to this status; only metrics/error can be None here.
        row = {key: value for key, value in asdict(record).items() if value is not None}
        with open(self.archive_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if record.metrics is None:
            logger.warning(
                "island %d: commit %s evaluation %s: %s",
                record.island_id,
                record.id,
                record.status,
                (record.error or "").partition("\n")[0],
            )
            return None

        fitness = self.fitness(record.id)
        island_map = self.maps[record.island_id]
        cell = self.cell(record.metrics)
        incumbent = island_map.get(cell)
        map_changed = incumbent is None or fitness > self.fitness(incumbent)
        if map_changed:
            island_map[cell] = record.id

        best_id = self.island_best[record.island_id]
        if best_id is None or fitness > self.fitness(best_id):
            best_id = record.id
            self.island_best[record.island_id] = best_id
            new_island_best = True
        else:
            new_island_best = False

        new_global_best = self.global_best is None or fitness > self.fitness(self.global_best)
        if new_global_best:
            self.global_best = record.id
            self._write_best(record)

        metrics = " ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in record.metrics.items()
        )
        description = " ".join(record.description.split())
        logger.info(
            "island %d: commit %s | %s | analyses=%d model_calls=%d context_tokens=%s | %s",
            record.island_id,
            record.id,
            metrics,
            record.behavior.run_analysis_count,
            record.behavior.model_calls,
            record.behavior.context_tokens,
            description,
        )

        if map_changed:
            coords = " ".join(
                f"{name}={index}" for name, index in zip(self.cfg.features, cell, strict=True)
            )
            if incumbent is None:
                logger.info(
                    "island %d: MAP-Elites new cell %s -> %s",
                    record.island_id,
                    coords,
                    record.id,
                )
            else:
                logger.info(
                    "island %d: MAP-Elites cell %s improved %s -> %s",
                    record.island_id,
                    coords,
                    incumbent,
                    record.id,
                )

        if new_island_best:
            logger.info(
                "island %d: new best %s (%s=%.4g)",
                record.island_id,
                record.id,
                self.cfg.fitness,
                fitness,
            )
        if new_global_best:
            logger.info("new global best %s (%s=%.4g)", record.id, self.cfg.fitness, fitness)

        return self.records[best_id]

    def _write_best(self, record: CommitRecord) -> None:
        tmp = self.best_py_path.with_name("best.py.tmp")
        tmp.write_text(record.code)
        os.replace(tmp, self.best_py_path)

        info = {"id": record.id, "metrics": record.metrics, "description": record.description}
        tmp = self.best_json_path.with_name("best.json.tmp")
        tmp.write_text(json.dumps(info, indent=2))
        os.replace(tmp, self.best_json_path)

    def island_elites(self, island_id: int) -> list[CommitRecord]:
        """Current MAP-Elites occupants of one island, best fitness first."""

        ids = sorted(self.maps[island_id].values(), key=self.fitness, reverse=True)
        return [self.records[cid] for cid in ids]

    def migration_candidates(self, island_id: int, rate: float) -> list[CommitRecord]:
        """Top ``rate`` fraction by fitness from the ring neighbors' current maps."""

        neighbors = {(island_id - 1) % self.num_islands, (island_id + 1) % self.num_islands}
        neighbors.discard(island_id)
        ids = {cid for n in neighbors for cid in self.maps[n].values()}
        if not ids:
            return []
        ranked = sorted(ids, key=self.fitness, reverse=True)
        return [self.records[cid] for cid in ranked[: math.ceil(len(ranked) * rate)]]
